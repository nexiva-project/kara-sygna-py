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


class DetectorCorpo:
    """Encapsula o MediaPipe Pose (modelo leve, por padrão)."""

    def __init__(
        self,
        complexidade: int = config.COMPLEXIDADE_POSE,
        confianca_deteccao: float = config.CONFIANCA_DETECCAO,
        confianca_rastreio: float = config.CONFIANCA_RASTREIO,
    ) -> None:
        """Cria o detector.

        Args:
            complexidade: 0 (leve, ~13 ms por quadro), 1 ou 2 (mais pesados).
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
        """Detecta o corpo; devolve o resultado bruto (``pose_landmarks``)."""
        if self._fechado:
            raise RuntimeError("O DetectorCorpo já foi fechado.")
        rgb = cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        return self._pose.process(rgb)

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


def criar_detector_corpo() -> DetectorCorpo | SemCorpo:
    """Abre o Pose; se falhar, segue sem ele (e avisa no log).

    Na primeira vez, o MediaPipe baixa o modelo do Pose da internet para a
    pasta de instalação. Sem internet, ou sem permissão de escrita nessa
    pasta, isso falha; antes o app inteiro deixava de abrir.
    """
    try:
        return DetectorCorpo()
    except Exception:
        logger.exception(
            "Não foi possível abrir o detector do corpo (MediaPipe Pose). O app "
            "continua sem os braços; confira a internet e reinicie para usá-los."
        )
        return SemCorpo()
