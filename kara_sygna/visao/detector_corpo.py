"""Detecção do corpo (MediaPipe Pose): ombros, cotovelos e pulsos.

Por que existe
--------------
Em sinais com movimento rápido, as mãos borram e o detector de mãos as perde
(no vídeo do sinal "carro", as duas mãos apareciam juntas em só ~20% dos
quadros). O braço é grande e continua visível mesmo borrado: no mesmo vídeo,
o Pose achou os dois pulsos e os dois cotovelos em 100% dos quadros. Por isso
o MOVIMENTO dos sinais é medido pelos braços, e a FORMA das mãos continua
vindo do detector de mãos.
"""

from __future__ import annotations

import logging
from typing import Any, Self

import cv2
import numpy as np

from kara_sygna import config

logger = logging.getLogger(__name__)

# Desenho na câmera (a IA continua usando só ombro, cotovelo e pulso).
# Cores em BGR, na mesma ordem da legenda da janela.
# Braços pedem 0,5. O quadril, com a pessoa perto, sai com confiança baixa
# e muitas vezes um pouco abaixo da imagem; 0,2 ainda desenha o tronco.
_VISIBILIDADE_DESENHO = 0.5
_VISIBILIDADE_TRONCO = 0.2
COR_OMBRO_BGR = (255, 255, 255)
COR_BRACO_BGR = (60, 200, 70)
COR_ANTEBRACO_BGR = (180, 80, 230)
COR_POSTURA_BGR = (180, 180, 180)
LEGENDA_CORPO = (
    ("OMBRO", "#FFFFFF"),
    ("BRAÇO", "#46C83C"),
    ("ANTEBRAÇO", "#E650B4"),
    ("POSTURA", "#B4B4B4"),
)

# MediaPipe Pose: ombro, cotovelo, pulso e quadril de cada lado da imagem.
_OMBRO = (11, 12)
_COTOVELO = (13, 14)
_PULSO = (15, 16)
_QUADRIL = (23, 24)
_JOELHO = (25, 26)


def desenhar_corpo(quadro_bgr: np.ndarray, resultado: Any) -> None:
    """Desenha ombro, braço, antebraço e a postura do tronco.

    O braço é o segmento ombro–cotovelo; o antebraço, cotovelo–pulso. A
    postura é a linha dos ombros, a do quadril, os lados do tronco e a
    coluna entre o meio dos ombros e o meio do quadril. O quadril entra
    mesmo com pouca confiança e um pouco fora da imagem (pessoa perto).
    Joelho entra quando está visível.
    """
    pontos = getattr(getattr(resultado, "pose_landmarks", None), "landmark", None)
    if pontos is None or len(pontos) < 25:
        return

    altura, largura = quadro_bgr.shape[:2]

    def pixel(indice: int, minimo: float) -> tuple[int, int] | None:
        if indice >= len(pontos):
            return None
        ponto = pontos[indice]
        if getattr(ponto, "visibility", 1.0) < minimo:
            return None
        # Quadril logo abaixo do quadro ainda vale; ponto perdido no
        # infinito, não.
        if not (-0.25 <= float(ponto.x) <= 1.25 and -0.15 <= float(ponto.y) <= 1.4):
            return None
        x = int(np.clip(ponto.x, 0.0, 1.0) * (largura - 1))
        y = int(np.clip(ponto.y, 0.0, 1.0) * (altura - 1))
        return x, y

    def linha(
        a: tuple[int, int] | None,
        b: tuple[int, int] | None,
        cor: tuple[int, int, int],
        espessura: int,
    ) -> None:
        if a is None or b is None:
            return
        cv2.line(quadro_bgr, a, b, cor, espessura, cv2.LINE_AA)

    ombros = [pixel(i, _VISIBILIDADE_DESENHO) for i in _OMBRO]
    cotovelos = [pixel(i, _VISIBILIDADE_DESENHO) for i in _COTOVELO]
    pulsos = [pixel(i, _VISIBILIDADE_DESENHO) for i in _PULSO]
    quadris = [pixel(i, _VISIBILIDADE_TRONCO) for i in _QUADRIL]
    joelhos = [pixel(i, _VISIBILIDADE_TRONCO) for i in _JOELHO]

    linha(ombros[0], ombros[1], COR_POSTURA_BGR, 2)
    linha(quadris[0], quadris[1], COR_POSTURA_BGR, 2)
    for lado in (0, 1):
        linha(ombros[lado], quadris[lado], COR_POSTURA_BGR, 2)
        linha(quadris[lado], joelhos[lado], COR_POSTURA_BGR, 2)
        linha(ombros[lado], cotovelos[lado], COR_BRACO_BGR, 3)
        linha(cotovelos[lado], pulsos[lado], COR_ANTEBRACO_BGR, 3)

    if all(ombros) and all(quadris):
        meio_ombro = (
            (ombros[0][0] + ombros[1][0]) // 2,
            (ombros[0][1] + ombros[1][1]) // 2,
        )
        meio_quadril = (
            (quadris[0][0] + quadris[1][0]) // 2,
            (quadris[0][1] + quadris[1][1]) // 2,
        )
        linha(meio_ombro, meio_quadril, COR_POSTURA_BGR, 2)

    for ponto in ombros:
        if ponto is not None:
            cv2.circle(quadro_bgr, ponto, 6, COR_OMBRO_BGR, -1, cv2.LINE_AA)
    for ponto in (*cotovelos, *pulsos, *quadris):
        if ponto is not None:
            cv2.circle(quadro_bgr, ponto, 4, COR_POSTURA_BGR, -1, cv2.LINE_AA)


class DetectorCorpo:
    """Encapsula o MediaPipe Pose (modelo leve, por padrão)."""

    def __init__(
        self,
        complexidade: int = config.COMPLEXIDADE_POSE,
        confianca_deteccao: float = config.CONFIANCA_DETECCAO_POSE,
        confianca_rastreio: float = config.CONFIANCA_RASTREIO_POSE,
    ) -> None:
        """Cria o detector.

        Args:
            complexidade: 0 (leve), 1 (completo; padrão) ou 2.
            confianca_deteccao: Confiança mínima para achar o corpo.
            confianca_rastreio: Confiança mínima para continuar seguindo.
        """
        import mediapipe as mp

        self._pose = mp.solutions.pose.Pose(
            model_complexity=complexidade,
            min_detection_confidence=confianca_deteccao,
            min_tracking_confidence=confianca_rastreio,
        )
        self._fechado = False
        logger.info("MediaPipe Pose inicializado (complexidade %d).", complexidade)

    def detectar(self, quadro_bgr: np.ndarray) -> Any:
        """Detecta o corpo; devolve o resultado bruto (``pose_landmarks``).

        O quadro ganha uma borda preta antes do Pose (``config.BORDA_POSE``).
        Os pontos voltam para as coordenadas da imagem original.
        """
        if self._fechado:
            raise RuntimeError("O DetectorCorpo já foi fechado.")
        altura, largura = quadro_bgr.shape[:2]
        borda = round(config.BORDA_POSE * max(largura, altura))
        rgb = cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB)
        if borda > 0:
            rgb = cv2.copyMakeBorder(
                rgb,
                borda,
                borda,
                borda,
                borda,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )
        rgb.flags.writeable = False
        resultado = self._pose.process(rgb)
        reposicionar_landmarks(resultado, largura, altura, borda)
        return resultado

    def fechar(self) -> None:
        """Libera o modelo. Pode ser chamado mais de uma vez."""
        if not self._fechado:
            self._pose.close()
            self._fechado = True
            logger.info("MediaPipe Pose encerrado.")

    def __enter__(self) -> Self:
        """Entra no bloco ``with`` devolvendo o próprio detector."""
        return self

    def __exit__(self, *_excecao) -> None:
        """Fecha o detector ao sair do bloco ``with``."""
        self.fechar()


class SemCorpo:
    """Substituto do :class:`DetectorCorpo` quando o Pose não pôde ser aberto.

    ``detectar`` devolve ``None``: o bloco dos braços fica em zero e as mãos
    passam sem a conferência pelos braços. O resto do app funciona.
    """

    def detectar(self, _quadro_bgr: np.ndarray) -> None:
        """Não há corpo."""
        return

    def fechar(self) -> None:
        """Nada a liberar."""

    def __enter__(self) -> Self:
        """Entra no bloco ``with``."""
        return self

    def __exit__(self, *_excecao) -> None:
        """Sai do bloco ``with``."""


def reposicionar_landmarks(
    resultado: Any, largura: int, altura: int, borda: int
) -> None:
    """Tira a borda das coordenadas normalizadas do Pose.

    O Pose mede x e y na imagem com a borda. Aqui eles voltam para o quadro
    da câmera (podem ficar um pouco fora de 0–1 quando o quadril sai da tela).
    """
    pontos = getattr(getattr(resultado, "pose_landmarks", None), "landmark", None)
    if pontos is None or borda <= 0 or largura <= 0 or altura <= 0:
        return
    largura_com_borda = largura + 2 * borda
    altura_com_borda = altura + 2 * borda
    for ponto in pontos:
        ponto.x = (ponto.x * largura_com_borda - borda) / largura
        ponto.y = (ponto.y * altura_com_borda - borda) / altura


def criar_detector_corpo() -> DetectorCorpo | SemCorpo:
    """Abre o Pose; se falhar, segue sem ele (e avisa no log).

    Tenta o modelo de ``config.COMPLEXIDADE_POSE`` e, se esse não abrir
    (download na primeira vez), o modelo leve. Sem nenhum dos dois, o app
    continua sem os braços.
    """
    tentativas = [config.COMPLEXIDADE_POSE]
    if config.COMPLEXIDADE_POSE != 0:
        tentativas.append(0)
    for complexidade in tentativas:
        try:
            return DetectorCorpo(complexidade=complexidade)
        except Exception:
            logger.exception(
                "Não foi possível abrir o Pose na complexidade %d.", complexidade
            )
    logger.error(
        "O app continua sem os braços; confira a internet e reinicie para usá-los."
    )
    return SemCorpo()
