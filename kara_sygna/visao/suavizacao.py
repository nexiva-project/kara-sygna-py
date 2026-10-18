"""Suavização temporal de pontos para visualização do Kara Sygna."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np


class SuavizadorPontos:
    """Suaviza tremores pequenos e acompanha movimentos maiores."""

    def __init__(
        self,
        intensidade: float = 0.65,
        zona_morta: float = 0.0,
        limiar_movimento: float = 0.012,
        resposta_movimento: float = 0.85,
    ) -> None:
        if not 0.0 < intensidade <= 1.0:
            raise ValueError("intensidade precisa estar entre 0 e 1.")

        if zona_morta < 0.0:
            raise ValueError("zona_morta não pode ser negativa.")

        if limiar_movimento <= 0.0:
            raise ValueError("limiar_movimento precisa ser positivo.")

        if not 0.0 < resposta_movimento <= 1.0:
            raise ValueError("resposta_movimento precisa estar entre 0 e 1.")

        self.intensidade = intensidade
        self.zona_morta = zona_morta
        self.limiar_movimento = limiar_movimento
        self.resposta_movimento = resposta_movimento

        self._historico: dict[Hashable, np.ndarray] = {}

    def suavizar(
        self,
        identificador: Hashable,
        pontos: np.ndarray,
    ) -> np.ndarray:
        """Retorna pontos suavizados sem modificar os dados recebidos."""

        atuais = np.asarray(pontos, dtype=np.float32)

        if not np.all(np.isfinite(atuais)):
            self._historico.pop(identificador, None)
            return atuais.copy()

        anteriores = self._historico.get(identificador)

        if anteriores is None or anteriores.shape != atuais.shape:
            resultado = atuais.copy()

        elif self.zona_morta == 0.0:
            # Com a zona morta desativada, mantém o filtro EMA original.
            resultado = (
                self.intensidade * atuais + (1.0 - self.intensidade) * anteriores
            ).astype(np.float32)

        else:
            diferenca = atuais - anteriores

            # Mede o deslocamento de cada ponto.
            distancia = np.linalg.norm(
                diferenca,
                axis=-1,
                keepdims=True,
            )

            # Elimina pequenas oscilações causadas pelo ruído.
            diferenca_filtrada = np.where(
                distancia <= self.zona_morta,
                0.0,
                diferenca,
            )

            # Movimentos pequenos recebem mais suavização.
            # Movimentos grandes recebem uma resposta mais rápida.
            fator = np.where(
                distancia >= self.limiar_movimento,
                self.resposta_movimento,
                self.intensidade,
            )

            resultado = (anteriores + fator * diferenca_filtrada).astype(np.float32)

        self._historico[identificador] = resultado.copy()
        return resultado

    def limpar(self, identificador: Hashable | None = None) -> None:
        """Limpa o histórico de um objeto ou de todos os objetos."""

        if identificador is None:
            self._historico.clear()
        else:
            self._historico.pop(identificador, None)
