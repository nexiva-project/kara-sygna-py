"""
Kara Sygna — aumento de dados com rotação 3D da mão.

Cria variações do mesmo sinal com a mão:
- inclinada para frente/trás
- inclinada para esquerda/direita
- girada no próprio eixo
- combinações dessas rotações

A rotação acontece em torno do pulso.

Uso:
    python src/aumentar_dados.py

Depois:
    python src/treinar_modelo.py
"""

import csv
import math
import os

from utils import FEATURES_POR_MAO


ARQUIVO_CSV = "dados_sinais.csv"

# ---------------------------------------------------------
# ROTAÇÕES 3D
#
# X = inclinar para frente/trás
# Y = inclinar para esquerda/direita
# Z = girar a mão no próprio plano
# ---------------------------------------------------------

ROTACOES = [
    # Pequenas rotações
    (15, 0, 0),
    (-15, 0, 0),

    (0, 15, 0),
    (0, -15, 0),

    (0, 0, 15),
    (0, 0, -15),

    # Rotações médias
    (30, 0, 0),
    (-30, 0, 0),

    (0, 30, 0),
    (0, -30, 0),

    (0, 0, 30),
    (0, 0, -30),

    # Combinações
    (20, 20, 0),
    (-20, -20, 0),

    (20, -20, 0),
    (-20, 20, 0),

    (20, 0, 20),
    (-20, 0, -20),

    (0, 20, 20),
    (0, -20, -20),
]


def vetor_zerado(vetor):
    """
    Verifica se a mão não foi detectada.
    """

    return all(
        abs(valor) < 1e-9
        for valor in vetor
    )


def matriz_rotacao(rx, ry, rz):
    """
    Cria uma matriz de rotação 3D.

    rx = rotação no eixo X
    ry = rotação no eixo Y
    rz = rotação no eixo Z
    """

    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)

    cx = math.cos(rx)
    sx = math.sin(rx)

    cy = math.cos(ry)
    sy = math.sin(ry)

    cz = math.cos(rz)
    sz = math.sin(rz)

    # Rotação X
    matriz_x = [
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx],
    ]

    # Rotação Y
    matriz_y = [
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ]

    # Rotação Z
    matriz_z = [
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1],
    ]

    # Primeiro X
    temp = multiplicar_matrizes(
        matriz_y,
        matriz_x,
    )

    # Depois Y
    resultado = multiplicar_matrizes(
        matriz_z,
        temp,
    )

    return resultado


def multiplicar_matrizes(a, b):
    """
    Multiplica duas matrizes 3x3.
    """

    resultado = [
        [0.0, 0.0, 0.0]
        for _ in range(3)
    ]

    for i in range(3):

        for j in range(3):

            resultado[i][j] = sum(
                a[i][k] * b[k][j]
                for k in range(3)
            )

    return resultado


def aplicar_rotacao(vetor, rx, ry, rz):
    """
    Aplica uma rotação 3D aos 21 landmarks.

    A rotação acontece em torno do pulso,
    que já é a origem porque os landmarks
    estão normalizados pelo utils.py.
    """

    matriz = matriz_rotacao(
        rx,
        ry,
        rz,
    )

    resultado = []

    for i in range(
        0,
        len(vetor),
        3,
    ):

        x = vetor[i]
        y = vetor[i + 1]
        z = vetor[i + 2]

        novo_x = (
            matriz[0][0] * x
            + matriz[0][1] * y
            + matriz[0][2] * z
        )

        novo_y = (
            matriz[1][0] * x
            + matriz[1][1] * y
            + matriz[1][2] * z
        )

        novo_z = (
            matriz[2][0] * x
            + matriz[2][1] * y
            + matriz[2][2] * z
        )

        resultado.extend([
            novo_x,
            novo_y,
            novo_z,
        ])

    return resultado


def aumentar_duas_maos(vetor):
    """
    Rotaciona as duas mãos juntas.

    Isso é importante para sinais que utilizam
    as duas mãos: a relação entre elas permanece
    consistente.
    """

    esquerda = vetor[
        :FEATURES_POR_MAO
    ]

    direita = vetor[
        FEATURES_POR_MAO:
    ]

    novas_amostras = []

    for rx, ry, rz in ROTACOES:

        if vetor_zerado(esquerda):
            nova_esquerda = esquerda
        else:
            nova_esquerda = aplicar_rotacao(
                esquerda,
                rx,
                ry,
                rz,
            )

        if vetor_zerado(direita):
            nova_direita = direita
        else:
            nova_direita = aplicar_rotacao(
                direita,
                rx,
                ry,
                rz,
            )

        nova_amostra = (
            nova_esquerda
            + nova_direita
        )

        novas_amostras.append(
            nova_amostra
        )

    return novas_amostras


def main():

    if not os.path.exists(ARQUIVO_CSV):

        print(
            f"Arquivo não encontrado: "
            f"{ARQUIVO_CSV}"
        )

        return

    print(
        f"Lendo dados de: "
        f"{ARQUIVO_CSV}"
    )

    with open(
        ARQUIVO_CSV,
        "r",
        newline="",
        encoding="utf-8",
    ) as arquivo:

        leitor = csv.reader(arquivo)

        cabecalho = next(leitor)

        linhas_originais = list(leitor)

    print(
        f"Amostras originais: "
        f"{len(linhas_originais)}"
    )

    novas_linhas = []

    for linha in linhas_originais:

        rotulo = linha[0]

        valores = [
            float(valor)
            for valor in linha[1:]
        ]

        quantidade_esperada = (
            FEATURES_POR_MAO * 2
        )

        if len(valores) != quantidade_esperada:

            print(
                f"Ignorando '{rotulo}': "
                f"{len(valores)} valores."
            )

            continue

        variantes = aumentar_duas_maos(
            valores
        )

        for variante in variantes:

            novas_linhas.append(
                [rotulo] + variante
            )

    if not novas_linhas:

        print(
            "Nenhuma nova amostra criada."
        )

        return

    # ---------------------------------------------------------
    # ADICIONA AS VARIAÇÕES AO CSV
    # ---------------------------------------------------------

    with open(
        ARQUIVO_CSV,
        "a",
        newline="",
        encoding="utf-8",
    ) as arquivo:

        escritor = csv.writer(arquivo)

        escritor.writerows(
            novas_linhas
        )

    print()
    print(
        "======================================"
    )

    print(
        "AUMENTO DE DADOS CONCLUÍDO"
    )

    print(
        "======================================"
    )

    print(
        f"Originais: "
        f"{len(linhas_originais)}"
    )

    print(
        f"Novas: "
        f"{len(novas_linhas)}"
    )

    print(
        f"Rotações por amostra: "
        f"{len(ROTACOES)}"
    )

    print()

    print(
        "Agora treine novamente:"
    )

    print(
        "python src/treinar_modelo.py"
    )

    print()

    print(
        "IMPORTANTE:"
    )

    print(
        "Não execute este script novamente "
        "sobre o mesmo CSV, pois as rotações "
        "serão adicionadas novamente."
    )


if __name__ == "__main__":
    main()