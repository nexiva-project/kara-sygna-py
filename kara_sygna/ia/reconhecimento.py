"""Reconhecimento: previsão de um quadro e confirmação do sinal ao longo do tempo."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from kara_sygna import config
from kara_sygna.ia.rotacao import ROTACOES, gerar_vetores_rotacionados


@dataclass(frozen=True)
class Predicao:
    """Resultado da previsão de um quadro.

    Attributes:
        sinal: Palavra mais provável.
        confianca: Probabilidade dessa palavra (0 a 1).
        margem: Diferença para a segunda mais provável (0 a 1).
        aceito: ``True`` se confiança e margem passam dos limites.
        rotacao: Rotação ``(rx, ry, rz)`` em graus que gerou o melhor
            resultado; ``(0, 0, 0)`` se nenhuma ajudou.
    """

    sinal: str
    confianca: float
    margem: float
    aceito: bool
    rotacao: tuple[int, int, int] = (0, 0, 0)


def _preparar_entrada(modelo: Any, matriz: np.ndarray):
    """Põe os nomes de coluna esperados pelo modelo (evita avisos do scikit-learn)."""
    nomes = getattr(modelo, "feature_names_in_", None)
    return pd.DataFrame(matriz, columns=nomes) if nomes is not None else matriz


def prever(
    modelo: Any,
    vetor: Sequence[float],
    rotacoes: Sequence[tuple[int, int, int]] | None = ROTACOES,
) -> Predicao:
    """Prevê o sinal de um quadro, testando várias inclinações da mão.

    Args:
        modelo: Modelo com ``predict_proba`` e ``classes_``.
        vetor: Vetor de 126 valores.
        rotacoes: Rotações a testar. ``None`` (ou lista com só ``(0,0,0)``)
            desliga a correção de inclinação.

    Returns:
        :class:`Predicao` da rotação em que o modelo ficou mais confiante.
    """
    rotacoes = tuple(rotacoes) if rotacoes else ((0, 0, 0),)
    matriz = gerar_vetores_rotacionados(vetor, rotacoes)
    probabilidades = modelo.predict_proba(_preparar_entrada(modelo, matriz))

    melhor_por_linha = probabilidades.max(axis=1)
    linha = int(melhor_por_linha.argmax())
    p = probabilidades[linha]
    ordem = np.argsort(p)[::-1]
    confianca = float(p[ordem[0]])
    segundo = float(p[ordem[1]]) if len(p) > 1 else 0.0
    margem = confianca - segundo
    aceito = confianca >= config.CONFIANCA_MINIMA and margem >= config.MARGEM_MINIMA
    return Predicao(
        str(modelo.classes_[ordem[0]]),
        confianca,
        margem,
        aceito,
        tuple(int(a) for a in rotacoes[linha]),
    )


class Estabilizador:
    """Confirma um sinal por votação numa janela de quadros.

    * Precisa de ``votos_minimos`` entre os últimos ``janela`` quadros.
    * Depois de confirmar, o mesmo sinal só é confirmado de novo depois de
      ``quadros_para_liberar`` quadros sem ele (tirar a mão, trocar de sinal).
      Assim é possível escrever "um um", e um sinal segurado não se repete.
    """

    def __init__(
        self,
        janela: int = config.JANELA_ESTABILIZADOR,
        votos_minimos: int = config.VOTOS_MINIMOS,
        quadros_para_liberar: int = config.QUADROS_PARA_LIBERAR,
    ) -> None:
        """Cria o estabilizador (veja a classe para o significado dos números)."""
        self.janela = janela
        self.votos_minimos = votos_minimos
        self.quadros_para_liberar = quadros_para_liberar
        self.reiniciar()

    def reiniciar(self) -> None:
        """Esquece todo o histórico (use ao começar ou ao limpar o texto)."""
        self._historico: deque = deque(maxlen=self.janela)
        self._travado: str | None = None
        self._quadros_sem_travado = 0

    def atualizar(self, sinal: str | None) -> str | None:
        """Registra o resultado de um quadro.

        Args:
            sinal: Palavra aceita neste quadro, ou ``None`` (sem mão ou incerto).

        Returns:
            A palavra recém-confirmada, ou ``None`` se nada foi confirmado agora.
        """
        self._historico.append(sinal)

        if self._travado is not None:
            self._quadros_sem_travado = (
                0 if sinal == self._travado else self._quadros_sem_travado + 1
            )
            if self._quadros_sem_travado >= self.quadros_para_liberar:
                self._travado = None

        votos = Counter(s for s in self._historico if s is not None)
        if not votos:
            return None
        candidato, quantidade = votos.most_common(1)[0]
        if quantidade >= self.votos_minimos and candidato != self._travado:
            self._travado = candidato
            self._quadros_sem_travado = 0
            return candidato
        return None
