"""Lógica da pré-visualização de um sinal gravado (sem nenhuma interface)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kara_sygna.visao.caracteristicas import espelhar_features, quantidade_de_maos

PASSO_GIRO_RADIANOS = 0.009


def completar_com_espelho(vetores: Sequence[Sequence[float]]) -> list[np.ndarray]:
    """Mostra cada exemplo de uma mão também espelhado na mão oposta.

    No treino, cada gravação também entra espelhada (veja
    :func:`kara_sygna.ia.sequencia.montar_amostras`). Na pré-visualização, um
    quadro de uma mão ganha a outra mão espelhada, para mostrar o sinal como
    será reconhecido com qualquer mão. Quadros com duas mãos (ou nenhuma)
    passam sem alteração.

    Args:
        vetores: Exemplos de 126 valores, na ordem de gravação.

    Returns:
        Lista de vetores de 126 valores.
    """
    resultado: list[np.ndarray] = []
    for vetor in vetores:
        atual = np.asarray(vetor, dtype=np.float64)
        if quantidade_de_maos(atual) == 1:
            # As mãos ficam em lados diferentes: somar junta as duas.
            atual = atual + espelhar_features(atual)
        resultado.append(atual)
    return resultado


class Reproducao:
    """Estado da animação de um sinal: qual exemplo mostrar e quanto girar.

    Args:
        nome: Palavra do sinal.
        vetores: Exemplos a reproduzir (ao menos um).

    Raises:
        ValueError: Se ``vetores`` estiver vazio.
    """

    def __init__(self, nome: str, vetores: Sequence[np.ndarray]) -> None:
        """Prepara a animação no primeiro exemplo."""
        if len(vetores) == 0:
            raise ValueError("Não há exemplos para reproduzir.")
        self.nome = nome
        self.vetores = list(vetores)
        self.indice = 0
        self.angulo = 0.0

    @property
    def vetor_atual(self) -> np.ndarray:
        """Exemplo que deve ser desenhado agora."""
        return self.vetores[self.indice]

    @property
    def legenda(self) -> str:
        """Texto como ``"oi - exemplo 3/80"`` (só ASCII, pois o OpenCV não desenha acentos)."""
        return f"{self.nome} - exemplo {self.indice + 1}/{len(self.vetores)}"

    def avancar(self) -> None:
        """Passa para o próximo exemplo (volta ao primeiro no fim) e gira um pouco."""
        self.indice = (self.indice + 1) % len(self.vetores)
        self.angulo += PASSO_GIRO_RADIANOS
