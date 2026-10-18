"""Memória curta das mãos: tapa os quadros em que uma mão "pisca".

Com a mão em movimento rápido, a imagem borra e o MediaPipe perde a mão por
1 a 3 quadros. Para a IA, isso vira "a mão sumiu" no meio do sinal. Aqui,
se uma mão vista há pouco (até ``config.MEMORIA_MAO_S``) não aparece no
quadro atual, a última posição dela continua valendo.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from kara_sygna import config

#: Distância mínima (fração da imagem) entre o pulso lembrado e o de uma mão
#: atual. Mais perto que isso, a mão lembrada é a mesma que já está na tela.
DISTANCIA_MINIMA = 0.08


@dataclass
class _MaoVista:
    landmarks: Any
    lateralidade: Any
    mundo: Any
    momento: float
    lado: str | None = None

    @property
    def pulso(self) -> tuple[float, float]:
        p = self.landmarks.landmark[0]
        return p.x, p.y


def _distancia(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class MemoriaMaos:
    """Completa o resultado do MediaPipe com mãos que sumiram há pouco.

    Args:
        memoria_s: Por quantos segundos uma mão sumida continua valendo
            (``0`` desliga). Padrão: ``config.MEMORIA_MAO_S``.
        max_maos: Quantas mãos lembrar. Padrão: ``config.MAX_MAOS``.
        relogio: Função que devolve o tempo em segundos (trocada nos testes).
    """

    def __init__(
        self,
        memoria_s: float | None = None,
        max_maos: int = config.MAX_MAOS,
        relogio: Callable[[], float] = time.monotonic,
    ) -> None:
        """Cria a memória vazia (veja a classe para os argumentos)."""
        self.memoria_s = float(config.MEMORIA_MAO_S if memoria_s is None else memoria_s)
        self.max_maos = max_maos
        self._relogio = relogio
        self._vistas: list[_MaoVista] = []
        #: Quantas mãos do último resultado vieram da memória.
        self.lembradas = 0

    def reiniciar(self) -> None:
        """Esquece todas as mãos."""
        self._vistas = []
        self.lembradas = 0

    def aplicar(self, resultado: Any) -> Any:
        """Devolve o resultado com as mãos lembradas acrescentadas.

        Args:
            resultado: Resultado bruto de ``DetectorMaos.detectar``.

        Returns:
            O próprio ``resultado`` se nada foi acrescentado; senão, um objeto
            com ``multi_hand_landmarks``, ``multi_handedness`` e
            ``multi_hand_world_landmarks`` completos.
        """
        agora = self._relogio()
        listas = list(getattr(resultado, "multi_hand_landmarks", None) or [])
        lados = list(getattr(resultado, "multi_handedness", None) or [])
        lados += [None] * (len(listas) - len(lados))
        mundos = list(getattr(resultado, "multi_hand_world_landmarks", None) or [])
        mundos += [None] * (len(listas) - len(mundos))
        # Lado de cada mão, quando os braços já decidiram (maos_pelo_corpo).
        slots = list(getattr(resultado, "lados", None) or [])
        slots += [None] * (len(listas) - len(slots))
        atuais = [
            _MaoVista(lm, lado, mundo, agora, slot)
            for lm, lado, mundo, slot in zip(listas, lados, mundos, slots, strict=True)
            if len(getattr(lm, "landmark", [])) >= 21
        ]

        # Mãos lembradas que ainda valem e não coincidem com uma mão atual.
        # A mão atual mais próxima "consome" a lembrança dela.
        restantes = [v for v in self._vistas if agora - v.momento <= self.memoria_s]
        for atual in atuais:
            if not restantes:
                break
            mais_perto = min(restantes, key=lambda v: _distancia(v.pulso, atual.pulso))
            restantes.remove(mais_perto)
        lados_atuais = {a.lado for a in atuais if a.lado}
        restantes = [
            v
            for v in restantes
            if all(_distancia(v.pulso, a.pulso) >= DISTANCIA_MINIMA for a in atuais)
            and (v.lado is None or v.lado not in lados_atuais)
        ][: max(0, self.max_maos - len(atuais))]

        self._vistas = atuais + restantes
        self.lembradas = len(restantes)
        if not restantes:
            return resultado
        todas = atuais + restantes
        return SimpleNamespace(
            multi_hand_landmarks=[v.landmarks for v in todas],
            multi_handedness=[v.lateralidade for v in todas],
            multi_hand_world_landmarks=[v.mundo for v in todas],
            lados=[v.lado for v in todas],
        )
