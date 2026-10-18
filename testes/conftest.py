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


@pytest.fixture(autouse=True)
def diagnostico_em_pasta_temporaria(tmp_path, monkeypatch):
    """Os arquivos de diagnóstico dos testes não vão para a pasta real."""
    from kara_sygna import config

    monkeypatch.setattr(config, "PASTA_DIAGNOSTICO", tmp_path / "diagnostico")


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


def quadro_falso(
    maos: np.ndarray | None = None,
    posicao_dir: tuple[float, float] = (0.0, 0.0),
    rosto: bool = True,
    corpo: bool = True,
) -> np.ndarray:
    """Vetor de um quadro (223 valores) com a forma das mãos e a posição da direita.

    Args:
        maos: Os 126 valores da forma das mãos (``None`` = sem mãos).
        posicao_dir: Posição ``(x, y)`` do pulso direito. Vale tanto para a mão
            (em relação ao nariz) quanto para o braço do Pose (em relação aos
            ombros), para os testes do movimento ficarem simples.
        rosto: Se o rosto foi encontrado.
        corpo: Se o corpo (braços) foi encontrado.
    """
    from kara_sygna.visao.quadro import (
        COLUNA_CORPO_PRESENTE,
        COLUNA_ROSTO_PRESENTE,
        COLUNAS_POSICAO,
        COLUNAS_PULSO_BRACO,
        IDX_MAOS,
        TOTAL_FEATURES_QUADRO,
    )

    vetor = np.zeros(TOTAL_FEATURES_QUADRO)
    if maos is not None:
        vetor[IDX_MAOS] = maos
    vetor[COLUNA_ROSTO_PRESENTE] = float(rosto)
    if rosto and maos is not None:
        col_x, col_y, col_escala = COLUNAS_POSICAO["dir"]
        vetor[[col_x, col_y, col_escala]] = (*posicao_dir, 0.5)
    if corpo:
        vetor[COLUNA_CORPO_PRESENTE] = 1.0
        vetor[list(COLUNAS_PULSO_BRACO["dir"])] = posicao_dir
    return vetor


def gravacao_falsa(
    forma: np.ndarray,
    rng: np.random.Generator,
    quadros: int = 20,
    movimento: tuple[float, float] = (0.0, 0.0),
    ruido: float = 0.02,
) -> np.ndarray:
    """Sequência de quadros de um sinal: mesma forma, pulso andando em linha reta."""
    linhas = []
    for t in range(quadros):
        fracao = t / max(1, quadros - 1)
        maos = forma + rng.normal(0, ruido, forma.shape) * (forma != 0)
        posicao = (movimento[0] * fracao, movimento[1] * fracao)
        linhas.append(quadro_falso(maos, posicao))
    return np.array(linhas)
