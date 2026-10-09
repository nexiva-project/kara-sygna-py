"""Kara Sygna — aumento de dados (só para o conjunto de TREINO).

Antes, o aumento gerava um CSV e o treino dividia treino/teste depois:
variações do mesmo quadro caíam nos dois lados e a acurácia mentia.
Agora o aumento é feito EM MEMÓRIA, depois da divisão, só no treino
(ver treinar_modelo.py). Não existe mais dados_sinais_aumentado.csv.

Variações geradas por amostra:
* rotação 3D global (como se a câmera/mão estivesse inclinada);
* pequena rotação extra independente em cada mão;
* variação de proporção da mão (dedos mais longos/curtos, mãos de
  outras pessoas);
* ruído nos landmarks, na posição relativa ao rosto e entre as mãos;
* espelhamento (esq <-> dir) para sinais de uma mão só.
"""

import numpy as np

from utils import (
    IDX_DIR, IDX_ENTRE_MAOS, IDX_ESQ, IDX_PRESENTE_DIR, IDX_PRESENTE_ESQ,
    IDX_PUNHO_ROSTO_DIR, IDX_PUNHO_ROSTO_ESQ, IDX_ROSTO, espelhar_features,
)


def matrizes_rotacao(rx, ry, rz):
    """Matrizes (n, 3, 3) a partir de ângulos em graus (arrays de tamanho n)."""
    rx, ry, rz = (np.radians(np.asarray(a, dtype=np.float64)) for a in (rx, ry, rz))
    n = len(rx)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)

    mx = np.zeros((n, 3, 3)); my = np.zeros((n, 3, 3)); mz = np.zeros((n, 3, 3))
    mx[:, 0, 0] = 1; mx[:, 1, 1] = cx; mx[:, 1, 2] = -sx; mx[:, 2, 1] = sx; mx[:, 2, 2] = cx
    my[:, 1, 1] = 1; my[:, 0, 0] = cy; my[:, 0, 2] = sy; my[:, 2, 0] = -sy; my[:, 2, 2] = cy
    mz[:, 2, 2] = 1; mz[:, 0, 0] = cz; mz[:, 0, 1] = -sz; mz[:, 1, 0] = sz; mz[:, 1, 1] = cz
    return mz @ my @ mx


def _variar(X, rng, intensidade):
    n = len(X)
    V = X.copy()

    ang = lambda graus: rng.uniform(-graus, graus, n) * intensidade
    global_R = matrizes_rotacao(ang(25), ang(25), ang(20))

    for indices, col_presente in ((IDX_ESQ, IDX_PRESENTE_ESQ), (IDX_DIR, IDX_PRESENTE_DIR)):
        presente = X[:, col_presente] > 0.5
        pontos = X[:, indices].reshape(n, 21, 3)

        jitter_R = matrizes_rotacao(
            rng.normal(0, 5, n), rng.normal(0, 5, n), rng.normal(0, 5, n)
        )
        R = jitter_R @ global_R
        pontos = np.einsum("nij,nkj->nki", R, pontos)
        pontos = pontos * rng.uniform(0.92, 1.08, (n, 1, 3))   # proporção da mão
        ruido = rng.normal(0, 0.02, pontos.shape)
        ruido[:, 0, :] = 0                                      # punho fica na origem
        pontos = pontos + ruido

        V[:, indices] = np.where(presente[:, None], pontos.reshape(n, 63), 0.0)

    # posições relativas (x, y): escala + ruído, só onde já existiam
    for indices in (IDX_PUNHO_ROSTO_ESQ, IDX_PUNHO_ROSTO_DIR, IDX_ENTRE_MAOS):
        bloco = X[:, indices]
        novo = bloco * rng.uniform(0.9, 1.1, (n, 1)) + rng.normal(0, 0.05, bloco.shape)
        novo[:, 2] = 0.0
        V[:, indices] = np.where(np.abs(bloco).sum(axis=1, keepdims=True) > 1e-9, novo, 0.0)

    rosto = X[:, IDX_ROSTO]
    novo = rosto * rng.uniform(0.95, 1.05, (n, 1)) + rng.normal(0, 0.01, rosto.shape)
    V[:, IDX_ROSTO] = np.where(np.abs(rosto).sum(axis=1, keepdims=True) > 1e-9, novo, 0.0)
    return V


def gerar_variantes(X, n_variantes=8, rng=None, intensidade=1.0):
    X = np.asarray(X, dtype=np.float64)
    rng = rng or np.random.default_rng(42)
    return np.vstack([_variar(X, rng, intensidade) for _ in range(n_variantes)])


def expandir_treino(X, y, n_variantes=8, rng=None):
    """Original + espelhados (sinais de uma mão) + variações aleatórias."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    rng = rng or np.random.default_rng(42)

    uma_mao = (X[:, IDX_PRESENTE_ESQ] + X[:, IDX_PRESENTE_DIR]) == 1
    X0 = np.vstack([X, espelhar_features(X[uma_mao])])
    y0 = np.concatenate([y, y[uma_mao]])

    if n_variantes <= 0:
        return X0, y0
    V = gerar_variantes(X0, n_variantes, rng)
    return np.vstack([X0, V]), np.concatenate([y0, np.tile(y0, n_variantes)])
