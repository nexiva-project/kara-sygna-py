
"""Kara Sygna — extração e engenharia de características.

Cada quadro gera exatamente 126 valores:
    [ mão esquerda (63) | mão direita (63) ]

Cada mão possui 21 landmarks, com coordenadas x, y e z.
Não utiliza características do rosto.
"""

import numpy as np


# ============================================================
# CONFIGURAÇÃO
# ============================================================

TOTAL_LANDMARKS_MAO = 21
FEATURES_POR_LANDMARK = 3
FEATURES_POR_MAO = TOTAL_LANDMARKS_MAO * FEATURES_POR_LANDMARK  # 63

TOTAL_FEATURES = FEATURES_POR_MAO * 2  # 126

IDX_ESQ = slice(0, FEATURES_POR_MAO)
IDX_DIR = slice(FEATURES_POR_MAO, TOTAL_FEATURES)

INVERTER_LATERALIDADE = True


# ============================================================
# LATERALIDADE
# ============================================================

def slot_do_rotulo(rotulo):
    """Converte o rótulo do MediaPipe em esq ou dir."""

    e_left = rotulo == "Left"

    if INVERTER_LATERALIDADE:
        return "dir" if e_left else "esq"

    return "esq" if e_left else "dir"


# ============================================================
# NOMES DAS CARACTERÍSTICAS
# ============================================================

def nomes_features():
    """Retorna os nomes das 126 características."""

    nomes = []

    for lado in ("esq", "dir"):
        for i in range(TOTAL_LANDMARKS_MAO):
            nomes.extend([
                f"{lado}_x{i}",
                f"{lado}_y{i}",
                f"{lado}_z{i}",
            ])

    assert len(nomes) == TOTAL_FEATURES

    return nomes


# ============================================================
# AUXILIARES
# ============================================================

def eh_vetor_zerado(vetor):
    """Verifica se um vetor está vazio ou contém somente zeros."""

    if vetor is None:
        return True

    v = np.asarray(vetor, dtype=np.float64)

    if v.size == 0:
        return True

    return bool(np.all(np.abs(v) < 1e-9))


def quantidade_de_maos(vetor):
    """Conta quantas mãos estão presentes em um vetor de 126 valores."""

    v = np.asarray(vetor, dtype=np.float64).reshape(-1)

    if v.size != TOTAL_FEATURES:
        raise ValueError(
            f"Esperados {TOTAL_FEATURES} valores; recebidos {v.size}."
        )

    esquerda = not eh_vetor_zerado(v[IDX_ESQ])
    direita = not eh_vetor_zerado(v[IDX_DIR])

    return int(esquerda) + int(direita)


# ============================================================
# CONVERSÃO DOS LANDMARKS
# ============================================================

def _pontos_em_pixels(landmarks, largura, altura):
    """Converte landmarks normalizados para coordenadas proporcionais."""

    return np.array(
        [
            [
                ponto.x * largura,
                ponto.y * altura,
                ponto.z * largura,
            ]
            for ponto in landmarks.landmark
        ],
        dtype=np.float64,
    )


def _normalizar_mao(pontos):
    """Centraliza os landmarks no pulso e normaliza pela palma."""

    pontos = np.asarray(pontos, dtype=np.float64)

    if pontos.shape != (TOTAL_LANDMARKS_MAO, 3):
        raise ValueError(
            "Uma mão precisa conter 21 landmarks com 3 coordenadas."
        )

    relativos = pontos - pontos[0]

    norma = np.linalg.norm

    escala = np.mean([
        norma(relativos[5]),
        norma(relativos[9]),
        norma(relativos[13]),
        norma(relativos[17]),
        norma(relativos[5] - relativos[17]),
    ])

    if not np.isfinite(escala) or escala < 1e-6:
        return None

    normalizados = relativos / escala

    if not np.all(np.isfinite(normalizados)):
        return None

    return normalizados


# ============================================================
# EXTRAÇÃO DE DUAS MÃOS
# ============================================================

def _separar_maos(resultado_hands, largura, altura):
    """Organiza os landmarks em slots de mão esquerda e direita."""

    slots = {
        "esq": None,
        "dir": None,
    }

    landmarks_lista = (
        getattr(resultado_hands, "multi_hand_landmarks", None) or []
    )

    if not landmarks_lista:
        return slots

    lateralidades = (
        getattr(resultado_hands, "multi_handedness", None) or []
    )

    maos = []

    for indice, landmarks in enumerate(landmarks_lista):
        rotulo = None

        if indice < len(lateralidades):
            classificacoes = getattr(
                lateralidades[indice],
                "classification",
                [],
            )

            if classificacoes:
                rotulo = classificacoes[0].label

        pontos = _pontos_em_pixels(
            landmarks,
            largura,
            altura,
        )

        maos.append((rotulo, pontos))

    if len(maos) == 1:
        rotulo, pontos = maos[0]

        lado = slot_do_rotulo(rotulo) if rotulo else "dir"
        slots[lado] = pontos

    else:
        # Com duas mãos, a posição horizontal ajuda a evitar
        # que uma mão sobrescreva a outra caso os rótulos coincidam.
        duas_maos = sorted(
            maos[:2],
            key=lambda item: item[1][0, 0],
        )

        lado_esquerdo_da_imagem = slot_do_rotulo("Left")
        lado_direito_da_imagem = (
            "esq"
            if lado_esquerdo_da_imagem == "dir"
            else "dir"
        )

        slots[lado_esquerdo_da_imagem] = duas_maos[0][1]
        slots[lado_direito_da_imagem] = duas_maos[1][1]

    return slots


def montar_vetor_duas_maos(resultado_hands):
    """Monta o vetor usado pela interface: 63 + 63 = 126 valores.

    A assinatura é compatível com:
        montar_vetor_duas_maos(resultado)

    O resultado deve ser retornado por MediaPipe Hands.
    """

    vetor = np.zeros(TOTAL_FEATURES, dtype=np.float64)

    # O resultado do MediaPipe contém coordenadas normalizadas.
    # A escala unitária mantém a função compatível com a chamada atual
    # da interface, que fornece somente o resultado do MediaPipe.
    slots = _separar_maos(
        resultado_hands,
        largura=1.0,
        altura=1.0,
    )

    for lado, intervalo in (
        ("esq", IDX_ESQ),
        ("dir", IDX_DIR),
    ):
        pontos = slots[lado]

        if pontos is None:
            continue

        normalizados = _normalizar_mao(pontos)

        if normalizados is None:
            continue

        vetor[intervalo] = normalizados.reshape(-1)

    return vetor.tolist()


# ============================================================
# ESPELHAMENTO DE UMA MÃO
# ============================================================

def espelhar_mao(vetor):
    """Espelha uma mão de 63 valores no eixo X."""

    pontos = np.asarray(
        vetor,
        dtype=np.float64,
    ).reshape(-1)

    if pontos.size != FEATURES_POR_MAO:
        raise ValueError(
            f"Esperados {FEATURES_POR_MAO} valores para uma mão; "
            f"recebidos {pontos.size}."
        )

    pontos = pontos.reshape(
        TOTAL_LANDMARKS_MAO,
        FEATURES_POR_LANDMARK,
    ).copy()

    # As coordenadas já estão centralizadas no pulso.
    pontos[:, 0] *= -1.0

    return pontos.reshape(-1).tolist()


# ============================================================
# ESPELHAMENTO DO VETOR COMPLETO
# ============================================================

def espelhar_features(vetor):
    """Espelha um vetor de 126 valores, trocando as mãos de lado.

    Aceita:
        - um vetor de 126 valores;
        - uma matriz com 126 colunas.

    A função mantém a estrutura de características usada pelo projeto.
    """

    v = np.asarray(vetor, dtype=np.float64)

    if v.shape[-1] != TOTAL_FEATURES:
        raise ValueError(
            f"O último eixo deve ter {TOTAL_FEATURES} valores; "
            f"recebidos {v.shape[-1]}."
        )

    saida = v.copy()

    mao_esquerda = v[..., IDX_ESQ]
    mao_direita = v[..., IDX_DIR]

    # A mão direita espelhada passa para a esquerda e vice-versa.
    saida[..., IDX_ESQ] = np.asarray([
        espelhar_mao(mao)
        for mao in mao_direita.reshape(-1, FEATURES_POR_MAO)
    ]).reshape(mao_direita.shape)

    saida[..., IDX_DIR] = np.asarray([
        espelhar_mao(mao)
        for mao in mao_esquerda.reshape(-1, FEATURES_POR_MAO)
    ]).reshape(mao_esquerda.shape)

    return saida
