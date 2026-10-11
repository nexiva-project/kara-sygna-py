"""Rotações 3D aplicadas às mãos (usadas no aumento de dados e na correção em tempo real)."""

from __future__ import annotations

import numpy as np

from kara_sygna.visao.caracteristicas import FEATURES_POR_MAO, TOTAL_LANDMARKS_MAO


def matrizes_rotacao(rx_graus, ry_graus, rz_graus) -> np.ndarray:
    """Cria matrizes de rotação 3D (ordem Z · Y · X) a partir de ângulos em graus.

    Args:
        rx_graus: Ângulo(s) em torno do eixo X (escalar ou array de tamanho n).
        ry_graus: Ângulo(s) em torno do eixo Y.
        rz_graus: Ângulo(s) em torno do eixo Z.

    Returns:
        Array (n, 3, 3), uma matriz por ângulo.
    """
    rx, ry, rz = (
        np.atleast_1d(np.radians(np.asarray(a, dtype=np.float64)))
        for a in (rx_graus, ry_graus, rz_graus)
    )
    n = len(rx)
    cx, sx, cy, sy, cz, sz = (
        np.cos(rx),
        np.sin(rx),
        np.cos(ry),
        np.sin(ry),
        np.cos(rz),
        np.sin(rz),
    )

    mx, my, mz = np.zeros((n, 3, 3)), np.zeros((n, 3, 3)), np.zeros((n, 3, 3))
    mx[:, 0, 0] = 1
    mx[:, 1, 1], mx[:, 1, 2], mx[:, 2, 1], mx[:, 2, 2] = cx, -sx, sx, cx
    my[:, 1, 1] = 1
    my[:, 0, 0], my[:, 0, 2], my[:, 2, 0], my[:, 2, 2] = cy, sy, -sy, cy
    mz[:, 2, 2] = 1
    mz[:, 0, 0], mz[:, 0, 1], mz[:, 1, 0], mz[:, 1, 1] = cz, -sz, sz, cz
    return mz @ my @ mx


def rotacionar_mao(maos: np.ndarray, matrizes: np.ndarray) -> np.ndarray:
    """Gira várias mãos, cada uma com a sua matriz.

    Uma mão ausente (só zeros) continua só com zeros, pois a rotação mantém a
    origem no lugar.

    Args:
        maos: Array (n, 63) com uma mão por linha.
        matrizes: Array (n, 3, 3) com a rotação de cada mão.

    Returns:
        Array (n, 63) com as mãos giradas.
    """
    pontos = maos.reshape(len(maos), TOTAL_LANDMARKS_MAO, 3)
    girados = np.einsum("nij,nkj->nki", matrizes, pontos)
    return girados.reshape(len(maos), FEATURES_POR_MAO)
