"""Extração das características (o "vetor") de cada quadro.

Cada quadro vira exatamente 126 números::

    [ mão esquerda (63) | mão direita (63) ]

Cada mão tem 21 landmarks com (x, y, z). As coordenadas são centradas no pulso
e divididas pelo tamanho da palma, então o vetor não depende de onde a mão está
na imagem nem de quão perto ela está da câmera. Uma mão ausente vira zeros.

Este módulo é a tradução fiel do antigo ``utils.py``: os números produzidos são
idênticos, então o CSV e o modelo já treinados continuam válidos.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kara_sygna import config

TOTAL_LANDMARKS_MAO = 21
FEATURES_POR_LANDMARK = 3
FEATURES_POR_MAO = TOTAL_LANDMARKS_MAO * FEATURES_POR_LANDMARK  # 63
TOTAL_FEATURES = FEATURES_POR_MAO * 2  # 126

IDX_ESQ = slice(0, FEATURES_POR_MAO)
IDX_DIR = slice(FEATURES_POR_MAO, TOTAL_FEATURES)

#: Nome de exibição de cada lado.
NOMES_LADO = {"esq": "Esquerda", "dir": "Direita"}


def slot_do_rotulo(rotulo: str) -> str:
    """Converte o rótulo de lateralidade do MediaPipe em ``"esq"`` ou ``"dir"``.

    A imagem é espelhada antes da detecção, como o MediaPipe espera, então o
    rótulo já é a mão de verdade (``config.INVERTER_LATERALIDADE`` = False).

    Args:
        rotulo: ``"Left"`` ou ``"Right"``, como devolvido pelo MediaPipe.

    Returns:
        ``"esq"`` ou ``"dir"``.
    """
    e_left = rotulo == "Left"
    if config.INVERTER_LATERALIDADE:
        return "dir" if e_left else "esq"
    return "esq" if e_left else "dir"


def nomes_features() -> list[str]:
    """Nomes das 126 colunas: ``esq_x0, esq_y0, esq_z0, ..., dir_z20``."""
    nomes = []
    for lado in ("esq", "dir"):
        for i in range(TOTAL_LANDMARKS_MAO):
            nomes.extend([f"{lado}_x{i}", f"{lado}_y{i}", f"{lado}_z{i}"])
    return nomes


def eh_vetor_zerado(vetor: Any) -> bool:
    """Diz se um vetor está vazio ou só tem zeros (mão ausente).

    Args:
        vetor: Lista/array de números, ou ``None``.
    """
    if vetor is None:
        return True
    v = np.asarray(vetor, dtype=np.float64)
    if v.size == 0:
        return True
    return bool(np.all(np.abs(v) < 1e-9))


def dividir_maos(vetor: Any) -> tuple[np.ndarray, np.ndarray]:
    """Separa um vetor de 126 valores em (mão esquerda, mão direita).

    Raises:
        ValueError: Se o vetor não tiver 126 valores.
    """
    v = np.asarray(vetor, dtype=np.float64).reshape(-1)
    if v.size != TOTAL_FEATURES:
        raise ValueError(f"Esperados {TOTAL_FEATURES} valores; recebidos {v.size}.")
    return v[IDX_ESQ], v[IDX_DIR]


def quantidade_de_maos(vetor: Any) -> int:
    """Conta quantas mãos (0, 1 ou 2) existem em um vetor de 126 valores."""
    esquerda, direita = dividir_maos(vetor)
    return int(not eh_vetor_zerado(esquerda)) + int(not eh_vetor_zerado(direita))


# ---------------------------------------------------------------------------
# Conversão do resultado do MediaPipe
# ---------------------------------------------------------------------------


def _pontos_da_mao(landmarks: Any) -> np.ndarray:
    """Converte os landmarks de uma mão em uma matriz (21, 3) de x, y, z."""
    return np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float64)


def normalizar_mao(pontos: Any) -> np.ndarray | None:
    """Centra a mão no pulso e divide pelo tamanho médio da palma.

    Args:
        pontos: Matriz (21, 3).

    Returns:
        Matriz (21, 3) normalizada, ou ``None`` se a mão for degenerada
        (palma de tamanho zero ou valores inválidos).

    Raises:
        ValueError: Se a forma não for (21, 3).
    """
    pontos = np.asarray(pontos, dtype=np.float64)
    if pontos.shape != (TOTAL_LANDMARKS_MAO, 3):
        raise ValueError("Uma mão precisa conter 21 landmarks com 3 coordenadas.")

    relativos = pontos - pontos[0]
    norma = np.linalg.norm
    escala = np.mean(
        [
            norma(relativos[5]),
            norma(relativos[9]),
            norma(relativos[13]),
            norma(relativos[17]),
            norma(relativos[5] - relativos[17]),
        ]
    )
    if not np.isfinite(escala) or escala < 1e-6:
        return None

    normalizados = relativos / escala
    if not np.all(np.isfinite(normalizados)):
        return None
    return normalizados


def _rotulo_da_mao(lateralidades: list, indice: int) -> str | None:
    """Lê o rótulo ``"Left"``/``"Right"`` da mão ``indice`` (ou ``None``)."""
    if indice >= len(lateralidades):
        return None
    classificacoes = getattr(lateralidades[indice], "classification", [])
    return classificacoes[0].label if classificacoes else None


def indices_dos_slots(resultado: Any) -> dict[str, int | None]:
    """Diz qual mão do resultado (índice) vai para cada "slot" esquerda/direita.

    Se o resultado já traz o lado de cada mão (``lados``, dado pelos braços em
    :func:`kara_sygna.visao.maos_pelo_corpo.conferir_maos`), ele vale. Senão,
    com uma mão vale a lateralidade informada pelo MediaPipe, e com duas vale
    a posição horizontal, para que uma mão nunca sobrescreva a outra se os dois
    rótulos vierem iguais.
    """
    slots: dict[str, int | None] = {"esq": None, "dir": None}
    listas = getattr(resultado, "multi_hand_landmarks", None) or []
    if not listas:
        return slots
    lados = getattr(resultado, "lados", None)
    if lados is not None and len(lados) == len(listas) and None not in lados:
        for indice, lado in enumerate(lados):
            if slots.get(lado) is None:
                slots[lado] = indice
        return slots
    lateralidades = getattr(resultado, "multi_handedness", None) or []

    if len(listas) == 1:
        rotulo = _rotulo_da_mao(lateralidades, 0)
        slots[slot_do_rotulo(rotulo) if rotulo else "dir"] = 0
        return slots

    duas = sorted(range(2), key=lambda i: listas[i].landmark[0].x)
    lado_da_esquerda_da_imagem = slot_do_rotulo("Left")
    lado_da_direita_da_imagem = "esq" if lado_da_esquerda_da_imagem == "dir" else "dir"
    slots[lado_da_esquerda_da_imagem] = duas[0]
    slots[lado_da_direita_da_imagem] = duas[1]
    return slots


def separar_maos(resultado: Any) -> dict[str, np.ndarray | None]:
    """Distribui as mãos detectadas nos "slots" esquerda e direita.

    Returns:
        Para cada slot, os pontos da mão na imagem, matriz (21, 3), ou ``None``.
    """
    listas = getattr(resultado, "multi_hand_landmarks", None) or []
    return {
        lado: None if indice is None else _pontos_da_mao(listas[indice])
        for lado, indice in indices_dos_slots(resultado).items()
    }


def separar_maos_mundo(resultado: Any) -> dict[str, np.ndarray | None]:
    """Como :func:`separar_maos`, mas com os *world landmarks* de cada mão.

    Os *world landmarks* são os mesmos 21 pontos em 3D, em metros, centrados
    na própria mão. A profundidade deles é bem melhor que o ``z`` da imagem.

    Returns:
        Para cada slot, matriz (21, 3), ou ``None`` se não houver a mão ou o
        resultado não trouxer os *world landmarks* dela.
    """
    mundo = getattr(resultado, "multi_hand_world_landmarks", None) or []
    saida: dict[str, np.ndarray | None] = {}
    for lado, indice in indices_dos_slots(resultado).items():
        landmarks = (
            mundo[indice] if indice is not None and indice < len(mundo) else None
        )
        valido = len(getattr(landmarks, "landmark", [])) >= TOTAL_LANDMARKS_MAO
        saida[lado] = _pontos_da_mao(landmarks) if valido else None
    return saida


def montar_vetor_duas_maos(resultado: Any) -> np.ndarray:
    """Monta o vetor de 126 valores a partir do resultado do MediaPipe.

    Args:
        resultado: Objeto devolvido por ``Hands.process`` (precisa de
            ``multi_hand_landmarks`` e, de preferência, ``multi_handedness``).

    Returns:
        Array de 126 ``float64``. Mãos ausentes ou degeneradas viram zeros.
    """
    vetor = np.zeros(TOTAL_FEATURES, dtype=np.float64)
    slots = separar_maos(resultado)
    for lado, intervalo in (("esq", IDX_ESQ), ("dir", IDX_DIR)):
        pontos = slots[lado]
        if pontos is None:
            continue
        normalizados = normalizar_mao(pontos)
        if normalizados is not None:
            vetor[intervalo] = normalizados.reshape(-1)
    return vetor


# ---------------------------------------------------------------------------
# Espelhamento
# ---------------------------------------------------------------------------


def espelhar_mao(vetor: Any) -> list[float]:
    """Espelha uma mão (63 valores) no eixo X.

    Args:
        vetor: 63 valores já centrados no pulso.

    Raises:
        ValueError: Se não houver exatamente 63 valores.
    """
    pontos = np.asarray(vetor, dtype=np.float64).reshape(-1)
    if pontos.size != FEATURES_POR_MAO:
        raise ValueError(
            f"Esperados {FEATURES_POR_MAO} valores para uma mão; recebidos {pontos.size}."
        )
    pontos = pontos.reshape(TOTAL_LANDMARKS_MAO, FEATURES_POR_LANDMARK).copy()
    pontos[:, 0] *= -1.0
    return pontos.reshape(-1).tolist()


def espelhar_features(vetor: Any) -> np.ndarray:
    """Espelha um vetor (ou matriz) de 126 valores, trocando as mãos de lado.

    Args:
        vetor: Vetor de 126 valores ou matriz com 126 colunas.

    Returns:
        Array do mesmo formato, com a mão direita espelhada indo para a
        esquerda e vice-versa.
    """
    v = np.asarray(vetor, dtype=np.float64)
    if v.shape[-1] != TOTAL_FEATURES:
        raise ValueError(
            f"O último eixo deve ter {TOTAL_FEATURES} valores; recebidos {v.shape[-1]}."
        )

    saida = v.copy()
    esquerda = v[..., IDX_ESQ].reshape(-1, FEATURES_POR_MAO)
    direita = v[..., IDX_DIR].reshape(-1, FEATURES_POR_MAO)

    def _espelhar_varias(maos: np.ndarray) -> np.ndarray:
        """Espelha cada mão (linha) de uma matriz (n, 63), mantendo o formato."""
        return np.asarray([espelhar_mao(mao) for mao in maos]).reshape(maos.shape)

    saida[..., IDX_ESQ] = _espelhar_varias(direita).reshape(v[..., IDX_ESQ].shape)
    saida[..., IDX_DIR] = _espelhar_varias(esquerda).reshape(v[..., IDX_DIR].shape)
    return saida
