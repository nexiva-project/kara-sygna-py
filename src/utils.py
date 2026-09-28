"""kara-sygna-py — utilitário compartilhado: normalização de landmarks."""

NUM_LANDMARKS = 21
FEATURES_POR_MAO = NUM_LANDMARKS * 3  # 63

PULSO = 0
BASE_DEDO_MEDIO = 9


def extrair_landmarks_normalizados(landmarks_da_mao):
    """
    Normaliza os landmarks de uma mão.

    A posição da mão na câmera é removida usando o pulso como origem.
    A escala é normalizada usando a distância entre o pulso e a base
    do dedo médio.

    Retorna:
        63 valores = 21 landmarks × (x, y, z)
    """

    pontos = landmarks_da_mao.landmark

    # ---------------------------------------------------------
    # 1. Pulso como origem
    # ---------------------------------------------------------

    origem_x = pontos[PULSO].x
    origem_y = pontos[PULSO].y
    origem_z = pontos[PULSO].z

    # ---------------------------------------------------------
    # 2. Tamanho da mão
    # ---------------------------------------------------------

    dx = pontos[BASE_DEDO_MEDIO].x - origem_x
    dy = pontos[BASE_DEDO_MEDIO].y - origem_y
    dz = pontos[BASE_DEDO_MEDIO].z - origem_z

    escala = (
        dx ** 2
        + dy ** 2
        + dz ** 2
    ) ** 0.5

    if escala < 1e-6:
        escala = 1e-6

    # ---------------------------------------------------------
    # 3. Coordenadas relativas ao pulso
    # ---------------------------------------------------------

    coordenadas = []

    for ponto in pontos:

        x_relativo = (
            ponto.x - origem_x
        ) / escala

        y_relativo = (
            ponto.y - origem_y
        ) / escala

        z_relativo = (
            ponto.z - origem_z
        ) / escala

        coordenadas.extend([
            x_relativo,
            y_relativo,
            z_relativo,
        ])

    return coordenadas


def montar_vetor_duas_maos(resultado):
    """
    Retorna um vetor fixo de 126 valores:

        63 valores = mão esquerda
        63 valores = mão direita

    Se uma mão não for detectada, seus valores ficam zerados.
    """

    vetor_esquerda = [0.0] * FEATURES_POR_MAO
    vetor_direita = [0.0] * FEATURES_POR_MAO

    if (
        resultado.multi_hand_landmarks
        and resultado.multi_handedness
    ):
        zipped = zip(
            resultado.multi_hand_landmarks,
            resultado.multi_handedness,
        )

        for landmarks_da_mao, info_da_mao in zipped:

            rotulo = info_da_mao.classification[0].label

            coordenadas = extrair_landmarks_normalizados(
                landmarks_da_mao
            )

            if rotulo == "Left":
                vetor_esquerda = coordenadas
            else:
                vetor_direita = coordenadas

    return vetor_esquerda + vetor_direita


def eh_vetor_zerado(vetor):
    """
    Verifica se uma mão não foi detectada.
    """

    return all(
        abs(valor) < 1e-9
        for valor in vetor
    )


def espelhar_mao(vetor_de_uma_mao):
    """
    Espelha uma mão invertendo o eixo X.

    Mantém Y e Z.
    """

    espelhado = list(vetor_de_uma_mao)

    for i in range(
        0,
        len(espelhado),
        3,
    ):
        espelhado[i] = -espelhado[i]

    return espelhado