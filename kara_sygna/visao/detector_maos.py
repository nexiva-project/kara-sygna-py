"""Detecção das mãos com MediaPipe e desenho do esqueleto."""

from __future__ import annotations

import logging
from typing import Any, Self

import cv2
import numpy as np

from kara_sygna import config
from kara_sygna.visao.caracteristicas import NOMES_LADO, indices_dos_slots

logger = logging.getLogger(__name__)


def _cor_bgr(valor: Any, padrao: tuple[int, int, int]) -> tuple[int, int, int]:
    """Converte uma cor hexadecimal para BGR, formato usado pelo OpenCV."""
    if not isinstance(valor, str):
        return padrao

    texto = valor.lstrip("#")
    if len(texto) != 6:
        return padrao

    try:
        vermelho = int(texto[0:2], 16)
        verde = int(texto[2:4], 16)
        azul = int(texto[4:6], 16)
        return azul, verde, vermelho
    except ValueError:
        return padrao


class DetectorMaos:
    """Encapsula o MediaPipe Hands."""

    def __init__(
        self,
        max_maos: int = config.MAX_MAOS,
        confianca_deteccao: float = config.CONFIANCA_DETECCAO,
        confianca_rastreio: float = config.CONFIANCA_RASTREIO,
    ) -> None:
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Esta versão do MediaPipe não tem 'mp.solutions'. "
                "Instale mediapipe==0.10.21."
            )

        self._hands_mp = mp.solutions.hands
        self._desenho = mp.solutions.drawing_utils
        self._hands = self._hands_mp.Hands(
            max_num_hands=max_maos,
            min_detection_confidence=confianca_deteccao,
            min_tracking_confidence=confianca_rastreio,
        )
        logger.info("MediaPipe Hands inicializado.")

    def detectar(self, quadro_bgr: np.ndarray) -> Any:
        """Detecta mãos sem alterar o quadro recebido."""
        rgb = cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB)
        return self._hands.process(rgb)

    def desenhar(
        self,
        quadro_bgr: np.ndarray,
        resultado: Any,
        estilo: dict[str, Any] | None = None,
    ) -> None:
        """Desenha as mãos. O parâmetro estilo controla apenas a aparência."""
        estilo = estilo or {}

        listas = getattr(resultado, "multi_hand_landmarks", None)
        lateralidades = getattr(resultado, "multi_handedness", None)

        if not listas or not lateralidades:
            return

        altura, largura = quadro_bgr.shape[:2]

        espessura = max(1, min(8, int(estilo.get("maos_espessura_linha", 2))))
        raio = max(1, min(8, int(estilo.get("maos_tamanho_ponto", 3))))
        cor_texto = _cor_bgr(
            estilo.get("maos_cor_texto", "#FFFFFF"),
            (255, 255, 255),
        )
        espessura_texto = max(1, min(8, int(estilo.get("maos_espessura_texto", 2))))

        # O mesmo lado ("esq"/"dir") que a IA usa para cada mão.
        lado_do_indice = {
            indice: lado
            for lado, indice in indices_dos_slots(resultado).items()
            if indice is not None
        }
        for indice, landmarks in enumerate(listas):
            lado = lado_do_indice.get(indice)
            if lado is None:
                continue
            sufixo = "esquerda" if lado == "esq" else "direita"

            cor_linha = _cor_bgr(
                estilo.get(f"maos_cor_linha_{sufixo}"),
                config.CORES_MAOS_BGR.get(lado, (0, 255, 0)),
            )
            cor_ponto = _cor_bgr(
                estilo.get(f"maos_cor_ponto_{sufixo}"),
                cor_linha,
            )

            self._desenho.draw_landmarks(
                quadro_bgr,
                landmarks,
                self._hands_mp.HAND_CONNECTIONS,
                self._desenho.DrawingSpec(
                    color=cor_ponto,
                    thickness=espessura,
                    circle_radius=raio,
                ),
                self._desenho.DrawingSpec(
                    color=cor_linha,
                    thickness=espessura,
                ),
            )

            pulso = landmarks.landmark[0]
            x = int(pulso.x * largura)
            y = max(25, int(pulso.y * altura) - 20)

            cv2.putText(
                quadro_bgr,
                NOMES_LADO.get(lado, str(lado).capitalize()),
                (x - 25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                cor_texto,
                espessura_texto,
                cv2.LINE_AA,
            )

    def fechar(self) -> None:
        """Libera os recursos do MediaPipe."""
        hands = getattr(self, "_hands", None)
        if hands is not None:
            hands.close()
            self._hands = None
            logger.info("MediaPipe Hands encerrado.")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_excecao) -> None:
        self.fechar()


def tem_maos(resultado: Any) -> bool:
    """Retorna True quando o resultado contém ao menos uma mão."""
    return bool(getattr(resultado, "multi_hand_landmarks", None))
