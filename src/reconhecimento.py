"""Kara Sygna — lógica de reconhecimento compartilhada (CLI e interface)."""

from collections import Counter, deque

import numpy as np
import pandas as pd

CONFIANCA_MINIMA = 0.70   # probabilidade mínima do melhor sinal
MARGEM_MINIMA = 0.15      # melhor - segundo melhor (evita empates)

# Rótulos que a IA aprende mas que NÃO viram texto. Grave "neutro" com a
# mão relaxada/em movimento para o modelo saber dizer "isto não é sinal".
IGNORAR = {"neutro", "nada"}


def prever(modelo, vetor):
    """Devolve (sinal, confiança, margem, aceito)."""
    entrada = pd.DataFrame([vetor], columns=modelo.feature_names_in_)
    p = modelo.predict_proba(entrada)[0]
    ordem = np.argsort(p)[::-1]
    melhor = ordem[0]
    segundo = float(p[ordem[1]]) if len(p) > 1 else 0.0
    confianca = float(p[melhor])
    margem = confianca - segundo
    aceito = confianca >= CONFIANCA_MINIMA and margem >= MARGEM_MINIMA
    return str(modelo.classes_[melhor]), confianca, margem, aceito


class Estabilizador:
    """Confirma um sinal por votação numa janela de quadros.

    * Precisa de ``votos_minimos`` dos últimos ``janela`` quadros.
    * Depois de confirmar, o mesmo sinal só é confirmado de novo depois
      de ``quadros_para_liberar`` quadros sem ele (tirar a mão, mudar de
      sinal ou "neutro"). Assim dá para escrever "eu eu", e um sinal
      segurado não se repete sozinho.
    """

    def __init__(self, janela=10, votos_minimos=6, quadros_para_liberar=8):
        self.janela = janela
        self.votos_minimos = votos_minimos
        self.quadros_para_liberar = quadros_para_liberar
        self.reiniciar()

    def reiniciar(self):
        self.historico = deque(maxlen=self.janela)
        self.travado = None
        self._sem_travado = 0

    def atualizar(self, sinal):
        """``sinal``: rótulo aceito neste quadro, ou None. Devolve o sinal
        recém-confirmado ou None."""
        self.historico.append(sinal)

        if self.travado is not None:
            self._sem_travado = 0 if sinal == self.travado else self._sem_travado + 1
            if self._sem_travado >= self.quadros_para_liberar:
                self.travado = None

        contagem = Counter(s for s in self.historico if s is not None)
        if not contagem:
            return None
        candidato, votos = contagem.most_common(1)[0]
        if votos >= self.votos_minimos and candidato != self.travado:
            self.travado = candidato
            self._sem_travado = 0
            return candidato
        return None
