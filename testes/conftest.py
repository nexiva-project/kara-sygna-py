"""Apoio compartilhado pelos testes: objetos falsos no lugar de câmera e MediaPipe."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from kara_sygna.visao.caracteristicas import FEATURES_POR_MAO, TOTAL_FEATURES


def mao_falsa(rng: np.random.Generator, deslocamento: float = 0.0):
    """Landmarks de uma mão aleatória (mesmo formato do MediaPipe)."""
    pontos = rng.random((21, 3)) * 0.2 + np.array([0.4 + deslocamento, 0.4, 0.0])
    return SimpleNamespace(
        landmark=[SimpleNamespace(x=x, y=y, z=z) for x, y, z in pontos]
    )


def lateralidade(rotulo: str):
    """Lateralidade ("Left"/"Right") no formato do MediaPipe."""
    return SimpleNamespace(classification=[SimpleNamespace(label=rotulo)])


def resultado_falso(*maos):
    """Resultado do MediaPipe com as mãos dadas, como ``(landmarks, "Left"|"Right")``."""
    if not maos:
        return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
    return SimpleNamespace(
        multi_hand_landmarks=[m for m, _ in maos],
        multi_handedness=[lateralidade(r) for _, r in maos],
    )


@pytest.fixture
def rng():
    """Gerador aleatório com semente fixa."""
    return np.random.default_rng(123)


def vetor_de_uma_mao(rng: np.random.Generator, lado: str = "dir") -> np.ndarray:
    """Vetor de 126 valores com só uma mão preenchida."""
    vetor = np.zeros(TOTAL_FEATURES)
    inicio = 0 if lado == "esq" else FEATURES_POR_MAO
    mao = rng.normal(0, 1, (21, 3))
    mao[0] = 0
    vetor[inicio : inicio + FEATURES_POR_MAO] = mao.reshape(-1)
    return vetor


def vetor_de_duas_maos(rng: np.random.Generator) -> np.ndarray:
    """Vetor de 126 valores com as duas mãos preenchidas."""
    return vetor_de_uma_mao(rng, "esq") + vetor_de_uma_mao(rng, "dir")
