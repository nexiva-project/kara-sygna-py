"""
kara-sygna-py — utilitário compartilhado: normalização de landmarks.

Por que normalizar?
    As coordenadas (x, y, z) que o MediaPipe devolve são relativas ao
    tamanho da IMAGEM inteira. Isso significa que o mesmo sinal feito
    em lugares diferentes da tela, ou a distâncias diferentes da
    câmera, gera números bem diferentes — e o classificador acaba
    aprendendo "onde a mão está" em vez de "qual é o formato do sinal".

Como resolvemos isso:
    1. Posição: subtraímos a posição do PULSO (landmark 0) de todos
       os pontos. Depois disso, o pulso vira a origem (0, 0, 0), e
       cada ponto representa sua posição RELATIVA ao pulso.
    2. Escala: dividimos todos os pontos pela distância entre o pulso
       e a base do dedo médio (landmark 9). Isso faz com que uma mão
       grande (perto da câmera) e uma mão pequena (longe da câmera)
       produzam números na mesma faixa, desde que o sinal seja igual.

Use SEMPRE esta mesma função em todos os scripts (coleta de dados e
reconhecimento em tempo real), para garantir que os dados de treino e
os dados usados na hora de prever estejam no mesmo formato.
"""

NUM_LANDMARKS = 21
FEATURES_POR_MAO = NUM_LANDMARKS * 3  # 63
PULSO = 0
BASE_DEDO_MEDIO = 9


def montar_vetor_duas_maos(resultado):
    """
    Recebe o 'resultado' bruto do MediaPipe (com multi_hand_landmarks
    e multi_handedness) e retorna um vetor de tamanho FIXO com
    2 * FEATURES_POR_MAO = 126 números: primeiro os 63 da mão
    ESQUERDA, depois os 63 da mão DIREITA.

    Se alguma das mãos não for detectada naquele frame, a parte dela
    no vetor é preenchida com zeros. Isso garante que toda amostra
    tenha sempre o mesmo tamanho, não importa quantas mãos apareceram
    -- essencial para treinar e usar o classificador depois.
    """
    vetor_esquerda = [0.0] * FEATURES_POR_MAO
    vetor_direita = [0.0] * FEATURES_POR_MAO

    if resultado.multi_hand_landmarks and resultado.multi_handedness:
        zipped = zip(resultado.multi_hand_landmarks, resultado.multi_handedness)
        for landmarks_da_mao, info_da_mao in zipped:
            rotulo = info_da_mao.classification[0].label  # "Left" ou "Right"
            coordenadas = extrair_landmarks_normalizados(landmarks_da_mao)

            if rotulo == "Left":
                vetor_esquerda = coordenadas
            else:
                vetor_direita = coordenadas

    return vetor_esquerda + vetor_direita


def extrair_landmarks_normalizados(landmarks_da_mao):
    """
    Recebe os landmarks de UMA mão (objeto do MediaPipe) e retorna uma
    lista achatada de 63 números (21 pontos x 3 coordenadas), já
    normalizados por posição e escala.
    """
    pontos = landmarks_da_mao.landmark

    origem_x = pontos[PULSO].x
    origem_y = pontos[PULSO].y
    origem_z = pontos[PULSO].z

    # Distância entre pulso e base do dedo médio, usada como
    # referência de escala (tamanho da mão naquele frame).
    dx = pontos[BASE_DEDO_MEDIO].x - origem_x
    dy = pontos[BASE_DEDO_MEDIO].y - origem_y
    escala = (dx ** 2 + dy ** 2) ** 0.5

    # Proteção: evita dividir por zero no caso raro da escala dar 0.
    if escala == 0:
        escala = 1e-6

    coordenadas = []
    for ponto in pontos:
        x_relativo = (ponto.x - origem_x) / escala
        y_relativo = (ponto.y - origem_y) / escala
        z_relativo = (ponto.z - origem_z) / escala
        coordenadas.extend([x_relativo, y_relativo, z_relativo])

    return coordenadas