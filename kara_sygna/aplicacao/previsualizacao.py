"""Lógica da pré-visualização de um sinal gravado (sem nenhuma interface)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kara_sygna.visao.caracteristicas import (
    FEATURES_POR_MAO,
    eh_vetor_zerado,
    quantidade_de_maos,
)

PASSO_GIRO_RADIANOS = 0.009


def combinar_espelhos(vetores: Sequence[Sequence[float]]) -> list[np.ndarray]:
    """Junta cada exemplo de uma mão com o seu espelho gravado logo depois.

    Sinais de uma mão são gravados duas vezes (original e espelhada na mão
    oposta). Na pré-visualização, as duas linhas viram um único quadro com as
    duas mãos, para mostrar o sinal como será reconhecido com qualquer mão.
    Quadros com duas mãos, ou sem par, passam sem alteração.

    Args:
        vetores: Exemplos de 126 valores, na ordem de gravação.

    Returns:
        Lista de vetores de 126 valores.
    """
    linhas = [np.asarray(v, dtype=np.float64) for v in vetores]
    resultado: list[np.ndarray] = []
    i = 0
    while i < len(linhas):
        atual = linhas[i]
        proximo = linhas[i + 1] if i + 1 < len(linhas) else None
        if (
            proximo is not None
            and quantidade_de_maos(atual) == 1
            and quantidade_de_maos(proximo) == 1
        ):
            atual_tem_esq = not eh_vetor_zerado(atual[:FEATURES_POR_MAO])
            proximo_tem_esq = not eh_vetor_zerado(proximo[:FEATURES_POR_MAO])
            if atual_tem_esq != proximo_tem_esq:
                esquerda = atual if atual_tem_esq else proximo
                direita = proximo if atual_tem_esq else atual
                resultado.append(
                    np.concatenate(
                        [esquerda[:FEATURES_POR_MAO], direita[FEATURES_POR_MAO:]]
                    )
                )
                i += 2
                continue
        resultado.append(atual)
        i += 1
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
