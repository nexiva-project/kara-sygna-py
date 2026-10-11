"""Detecção das mãos com MediaPipe e desenho do esqueleto sobre o vídeo."""

from __future__ import annotations

import logging
from typing import Any, Self

import cv2
import numpy as np

from kara_sygna import config
from kara_sygna.visao.caracteristicas import NOMES_LADO, slot_do_rotulo

logger = logging.getLogger(__name__)


class DetectorMaos:
    """Encapsula o ``mediapipe.solutions.hands``.

    O MediaPipe só é importado ao criar o detector. Assim o resto do projeto
    (e os testes) funciona mesmo sem ele instalado.

    Raises:
        RuntimeError: Na criação, se a versão do MediaPipe instalada não tiver
            ``mp.solutions`` (use ``mediapipe==0.10.21``).
    """

    def __init__(
        self,
        max_maos: int = config.MAX_MAOS,
        confianca_deteccao: float = config.CONFIANCA_DETECCAO,
        confianca_rastreio: float = config.CONFIANCA_RASTREIO,
    ) -> None:
        """Inicializa o MediaPipe Hands.

        Args:
            max_maos: Máximo de mãos detectadas por quadro.
            confianca_deteccao: Confiança mínima para detectar uma mão (0 a 1).
            confianca_rastreio: Confiança mínima para continuar rastreando.
        """
        import mediapipe as mp

        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Esta versão do MediaPipe não tem 'mp.solutions'. "
                "Instale a versão certa: pip install mediapipe==0.10.21 (Python 3.12 ou anterior)."
            )
        self._hands_mp = mp.solutions.hands
        self._desenho = mp.solutions.drawing_utils
        self._hands = self._hands_mp.Hands(
            max_num_hands=max_maos,
            min_detection_confidence=confianca_deteccao,
            min_tracking_confidence=confianca_rastreio,
        )
        logger.info("MediaPipe inicializado.")

    def detectar(self, quadro_bgr: np.ndarray) -> Any:
        """Procura mãos em um quadro.

        Args:
            quadro_bgr: Imagem BGR **já espelhada** (efeito espelho).

        Returns:
            Resultado do MediaPipe (``multi_hand_landmarks`` e
            ``multi_handedness``).
        """
        rgb = cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB)
        return self._hands.process(rgb)

    def desenhar(self, quadro_bgr: np.ndarray, resultado: Any) -> None:
        """Desenha o esqueleto e o nome do lado (Esquerda/Direita) em cada mão.

        Args:
            quadro_bgr: Imagem onde desenhar (modificada no lugar).
            resultado: Retorno de :meth:`detectar`.
        """
        listas = getattr(resultado, "multi_hand_landmarks", None)
        lateralidades = getattr(resultado, "multi_handedness", None)
        if not (listas and lateralidades):
            return

        altura, largura = quadro_bgr.shape[:2]
        for landmarks, info in zip(listas, lateralidades):
            lado = slot_do_rotulo(info.classification[0].label)
            cor = config.CORES_MAOS_BGR[lado]
            self._desenho.draw_landmarks(
                quadro_bgr,
                landmarks,
                self._hands_mp.HAND_CONNECTIONS,
                self._desenho.DrawingSpec(color=cor, thickness=2, circle_radius=3),
                self._desenho.DrawingSpec(color=cor, thickness=2),
            )
            pulso = landmarks.landmark[0]
            x = int(pulso.x * largura)
            y = max(25, int(pulso.y * altura) - 20)
            cv2.putText(
                quadro_bgr,
                NOMES_LADO[lado],
                (x - 25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                cor,
                2,
                cv2.LINE_AA,
            )

    def fechar(self) -> None:
        """Libera os recursos do MediaPipe. Pode ser chamado mais de uma vez."""
        hands: Any | None = getattr(self, "_hands", None)
        if hands is not None:
            hands.close()
            self._hands = None
            logger.info("MediaPipe encerrado.")

    def __enter__(self) -> Self:
        """Entra no bloco ``with`` devolvendo o próprio detector."""
        return self

    def __exit__(self, *_excecao) -> None:
        """Fecha o MediaPipe ao sair do bloco ``with``."""
        self.fechar()


def tem_maos(resultado: Any) -> bool:
    """Diz se o resultado do MediaPipe contém ao menos uma mão."""
    return bool(getattr(resultado, "multi_hand_landmarks", None))
