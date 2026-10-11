"""Correção de inclinação na hora de reconhecer.

A mesma forma de mão, inclinada de leve, gera números diferentes. Para
compensar, cada quadro é testado em várias rotações e vale a mais confiante.
"""

from __future__ import annotations

import numpy as np

from kara_sygna.ia.geometria import matrizes_rotacao, rotacionar_mao
from kara_sygna.visao.caracteristicas import FEATURES_POR_MAO, TOTAL_FEATURES

#: Rotações testadas ``(rx, ry, rz)`` em graus. A primeira é "sem rotação".
ROTACOES = (
    (0, 0, 0),
    (15, 0, 0),
    (-15, 0, 0),
    (30, 0, 0),
    (-30, 0, 0),
    (45, 0, 0),
    (-45, 0, 0),
    (0, 15, 0),
    (0, -15, 0),
    (0, 30, 0),
    (0, -30, 0),
    (0, 45, 0),
    (0, -45, 0),
    (0, 0, 15),
    (0, 0, -15),
    (0, 0, 30),
    (0, 0, -30),
    (0, 0, 45),
    (0, 0, -45),
    (20, 20, 0),
    (20, -20, 0),
    (-20, 20, 0),
    (-20, -20, 0),
    (20, 0, 20),
    (20, 0, -20),
    (-20, 0, 20),
    (-20, 0, -20),
    (0, 20, 20),
    (0, 20, -20),
    (0, -20, 20),
    (0, -20, -20),
)


def gerar_vetores_rotacionados(vetor, rotacoes=ROTACOES) -> np.ndarray:
    """Gera uma cópia do vetor para cada rotação.

    Args:
        vetor: Vetor de 126 valores.
        rotacoes: Lista de ``(rx, ry, rz)`` em graus.

    Returns:
        Matriz (len(rotacoes), 126). Cada linha tem as duas mãos giradas.

    Raises:
        ValueError: Se o vetor não tiver 126 valores.
    """
    vetor = np.asarray(vetor, dtype=np.float64).reshape(-1)
    if vetor.size != TOTAL_FEATURES:
        raise ValueError(f"Esperados {TOTAL_FEATURES} valores; recebidos {vetor.size}.")

    angulos = np.asarray(rotacoes, dtype=np.float64)
    matrizes = matrizes_rotacao(angulos[:, 0], angulos[:, 1], angulos[:, 2])
    saida = np.empty((len(angulos), TOTAL_FEATURES))
    for inicio in (0, FEATURES_POR_MAO):
        mao = np.tile(vetor[inicio : inicio + FEATURES_POR_MAO], (len(angulos), 1))
        saida[:, inicio : inicio + FEATURES_POR_MAO] = rotacionar_mao(mao, matrizes)
    return saida
