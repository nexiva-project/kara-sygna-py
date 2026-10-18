"""Escolhe quais quadros da câmera são usados no ensino e no reconhecimento.

Há dois modos, escolhidos em ``config.MODO_QUADROS``:

* ``"mudanca"`` (o filtro original do projeto): um quadro só é usado quando
  algo muda em relação ao último usado: a forma das mãos, o lugar das mãos
  ou uma mão que aparece/some. Com a mão parada, nada novo é gravado (no
  máximo um quadro a cada ``PARADO_USAR_A_CADA_S`` segundos), então o CSV
  cresce pouco. Com movimento, cada mudança vira um quadro novo.
* ``"fixo"``: usa 1 a cada ``USAR_1_A_CADA_QUADROS`` quadros, mude ou não.

O ensino e o reconhecimento usam o mesmo modo, para a IA ver ao vivo o mesmo
tipo de sequência que viu no treino.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from kara_sygna import config
from kara_sygna.visao.quadro import (
    IDX_MAOS,
    IDX_POSICAO,
    TOTAL_FEATURES_QUADRO,
    maos_presentes,
)

MODOS = ("mudanca", "fixo")


class FiltroQuadros:
    """Decide, quadro a quadro, se ele deve ser usado.

    Args:
        modo: ``"mudanca"`` ou ``"fixo"``. Padrão: ``config.MODO_QUADROS``.
        limiar: Mudança mínima na forma das mãos (média da diferença dos
            126 valores). Padrão: ``config.LIMIAR_MUDANCA_FORMA``.
        limiar_posicao: Mudança mínima no lugar das mãos, em larguras de
            rosto. Padrão: ``config.LIMIAR_MUDANCA_POSICAO``.
        parado_a_cada_s: Com a mão parada, usa um quadro a cada tantos
            segundos (``0`` = nunca). Padrão: ``config.PARADO_USAR_A_CADA_S``.
        a_cada: No modo ``"fixo"``, usa 1 a cada tantos quadros. Padrão:
            ``config.USAR_1_A_CADA_QUADROS``.
        relogio: Função que devolve o tempo em segundos (trocada nos testes).
    """

    def __init__(
        self,
        modo: str | None = None,
        limiar: float | None = None,
        limiar_posicao: float | None = None,
        parado_a_cada_s: float | None = None,
        a_cada: int | None = None,
        relogio: Callable[[], float] = time.monotonic,
    ) -> None:
        """Cria o filtro (veja a classe para os argumentos)."""
        self.modo = config.MODO_QUADROS if modo is None else modo
        if self.modo not in MODOS:
            raise ValueError(
                f"MODO_QUADROS deve ser {' ou '.join(map(repr, MODOS))}; "
                f"recebido {self.modo!r}."
            )
        self.limiar = float(config.LIMIAR_MUDANCA_FORMA if limiar is None else limiar)
        self.limiar_posicao = float(
            config.LIMIAR_MUDANCA_POSICAO if limiar_posicao is None else limiar_posicao
        )
        self.parado_a_cada_s = float(
            config.PARADO_USAR_A_CADA_S if parado_a_cada_s is None else parado_a_cada_s
        )
        self.a_cada = max(
            1, int(config.USAR_1_A_CADA_QUADROS if a_cada is None else a_cada)
        )
        for nome in ("limiar", "limiar_posicao", "parado_a_cada_s"):
            valor = getattr(self, nome)
            if not np.isfinite(valor) or valor < 0:
                raise ValueError(f"{nome} precisa ser finito e não negativo.")
        self._relogio = relogio
        self.reiniciar()

    def reiniciar(self) -> None:
        """Esquece o histórico (ao começar uma nova rodada ou reconhecimento)."""
        self._ultimo_aceito: np.ndarray | None = None
        self._momento_ultimo = 0.0
        self._contador = 0

    def aceitar(self, vetor: np.ndarray) -> bool:
        """Diz se o quadro deve ser usado (e, se sim, lembra dele).

        Args:
            vetor: Vetor de 223 valores do quadro.

        Raises:
            ValueError: Se o vetor não tiver 223 valores.
        """
        atual = np.asarray(vetor, dtype=np.float64).reshape(-1)
        if atual.size != TOTAL_FEATURES_QUADRO:
            raise ValueError(
                f"Esperados {TOTAL_FEATURES_QUADRO} valores; recebidos {atual.size}."
            )
        if not np.all(np.isfinite(atual)):
            return False

        if self.modo == "fixo":
            usar = self._contador % self.a_cada == 0
            self._contador += 1
            return usar

        agora = self._relogio()
        if self._mudou(atual) or (
            self.parado_a_cada_s > 0
            and agora - self._momento_ultimo >= self.parado_a_cada_s
        ):
            self._ultimo_aceito = atual.copy()
            self._momento_ultimo = agora
            return True
        return False

    def _mudou(self, atual: np.ndarray) -> bool:
        """Compara com o último quadro usado: presença, forma e lugar das mãos."""
        anterior = self._ultimo_aceito
        if anterior is None:
            return True

        presentes = maos_presentes(np.vstack([anterior, atual]))
        if not np.array_equal(presentes[0], presentes[1]):
            return True  # uma mão apareceu ou sumiu

        forma = float(np.mean(np.abs(atual[IDX_MAOS] - anterior[IDX_MAOS])))
        lugar = float(np.max(np.abs(atual[IDX_POSICAO] - anterior[IDX_POSICAO])))
        return forma >= self.limiar or lugar >= self.limiar_posicao
