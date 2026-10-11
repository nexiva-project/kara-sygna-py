"""Aumento de dados: cria variações dos exemplos de TREINO.

O aumento é feito em memória, depois de separar treino e teste, e nunca no
conjunto de teste. Assim a acurácia da validação não é inflada por cópias do
mesmo quadro.

Variações geradas: rotação 3D global, pequena rotação extra em cada mão,
proporção da mão (dedos mais longos/curtos, mãos de outras pessoas) e ruído.
(O espelhamento de sinais de uma mão já é gravado no CSV.)
"""

from __future__ import annotations

import numpy as np

from kara_sygna import config
from kara_sygna.ia.geometria import matrizes_rotacao, rotacionar_mao
from kara_sygna.visao.caracteristicas import FEATURES_POR_MAO, IDX_DIR, IDX_ESQ


def numero_de_variantes(n_linhas: int) -> int:
    """Decide quantas variações gerar por exemplo, para manter o treino rápido.

    Args:
        n_linhas: Quantidade de exemplos de treino.

    Returns:
        Um valor entre ``MIN_VARIANTES_AUMENTO`` e ``MAX_VARIANTES_AUMENTO``.
    """
    por_orcamento = config.ORCAMENTO_LINHAS_AUMENTO // max(1, n_linhas)
    return int(
        max(
            config.MIN_VARIANTES_AUMENTO,
            min(config.MAX_VARIANTES_AUMENTO, por_orcamento),
        )
    )


def _variar(X: np.ndarray, rng: np.random.Generator, intensidade: float) -> np.ndarray:
    """Gera uma variação aleatória de cada linha de ``X``."""
    n = len(X)
    saida = X.copy()

    def angulos(graus: float) -> np.ndarray:
        """Sorteia ``n`` ângulos entre -graus e +graus, já multiplicados pela intensidade."""
        return rng.uniform(-graus, graus, n) * intensidade

    rotacao_global = matrizes_rotacao(angulos(25), angulos(25), angulos(20))

    for intervalo in (IDX_ESQ, IDX_DIR):
        maos = X[:, intervalo]
        presente = np.abs(maos).sum(axis=1) > 1e-9

        ajuste_individual = matrizes_rotacao(
            rng.normal(0, 5, n), rng.normal(0, 5, n), rng.normal(0, 5, n)
        )
        pontos = rotacionar_mao(maos, ajuste_individual @ rotacao_global).reshape(
            n, -1, 3
        )
        pontos = pontos * rng.uniform(0.92, 1.08, (n, 1, 3))  # proporção da mão
        ruido = rng.normal(0, 0.02, pontos.shape)
        ruido[:, 0, :] = 0  # o pulso fica na origem
        pontos = pontos + ruido

        saida[:, intervalo] = np.where(
            presente[:, None], pontos.reshape(n, FEATURES_POR_MAO), 0.0
        )
    return saida


def expandir_treino(
    X: np.ndarray,
    y: np.ndarray,
    n_variantes: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Devolve os exemplos originais mais ``n_variantes`` variações de cada um.

    Args:
        X: Matriz (n, 126).
        y: Rótulos (n,).
        n_variantes: Quantas variações por exemplo (0 = só os originais).
        rng: Gerador aleatório (para resultados reproduzíveis nos testes).

    Returns:
        Tupla ``(X_expandido, y_expandido)``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if n_variantes <= 0:
        return X, y
    rng = rng or np.random.default_rng(config.SEMENTE_ALEATORIA)
    variantes = [_variar(X, rng, 1.0) for _ in range(n_variantes)]
    return np.vstack([X, *variantes]), np.concatenate([y, *([y] * n_variantes)])
