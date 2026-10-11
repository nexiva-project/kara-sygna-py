"""Desenho 3D esquelético das mãos (usado só na pré-visualização do sinal).

Recebe vetores já normalizados (o mesmo formato do CSV) e devolve uma imagem
OpenCV (``numpy``). Quem exibe a imagem é a interface; este módulo não abre
janela nenhuma.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

from kara_sygna import config
from kara_sygna.visao.caracteristicas import (
    FEATURES_POR_MAO,
    dividir_maos,
    eh_vetor_zerado,
)

LIGACOES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)
PALMA = (0, 5, 9, 13, 17)

COR_FUNDO = (18, 20, 28)
COR_TEXTO = (225, 225, 235)
COR_DIVISORIA = (58, 63, 78)

#: Escala das mãos já normalizadas pela palma (≈1 unidade) no painel.
ESCALA_MAO_NORMALIZADA = 0.12
POSICAO_ESQUERDA = 0.27
POSICAO_DIREITA = 0.73


def rotacionar(pontos: np.ndarray, angulo_x: float, angulo_y: float) -> np.ndarray:
    """Gira os pontos nos eixos X e Y para revelar a profundidade.

    Args:
        pontos: Matriz (n, 3).
        angulo_x: Inclinação, em radianos.
        angulo_y: Giro, em radianos.
    """
    cx, sx = math.cos(angulo_x), math.sin(angulo_x)
    cy, sy = math.cos(angulo_y), math.sin(angulo_y)
    rot_x = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    rot_y = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    return pontos @ (rot_y @ rot_x).T


def projetar_mao(
    vetor_mao: Sequence[float],
    tamanho: tuple[int, int],
    angulo_x: float,
    angulo_y: float,
    posicao_horizontal: float,
    escala_relativa: float = ESCALA_MAO_NORMALIZADA,
) -> list[tuple[int, int, float]]:
    """Projeta os 21 pontos de uma mão (63 valores) na tela, com perspectiva.

    Args:
        vetor_mao: 63 valores (x, y, z de 21 landmarks), centrados no pulso.
        tamanho: ``(largura, altura)`` do painel.
        angulo_x: Inclinação da câmera virtual, em radianos.
        angulo_y: Giro da câmera virtual, em radianos.
        posicao_horizontal: Onde fica o pulso, de 0 (esquerda) a 1 (direita).
        escala_relativa: Fator de tamanho da mão.

    Returns:
        Lista de 21 tuplas ``(x_px, y_px, profundidade)``.
    """
    p = np.asarray(vetor_mao, dtype=np.float32).reshape(-1, 3)
    pontos = np.stack([p[:, 0], -p[:, 1], -p[:, 2] * 1.8], axis=1)
    pontos = rotacionar(pontos, angulo_x, angulo_y)

    largura, altura = tamanho
    centro_x, centro_y = largura * posicao_horizontal, altura * 0.62
    distancia_camera = 2.6
    escala = min(largura, altura) * 1.25 * escala_relativa

    projetados = []
    for x, y, z in pontos:
        fator = distancia_camera / max(0.25, distancia_camera - z)
        projetados.append(
            (
                int(centro_x + x * escala * fator),
                int(centro_y - y * escala * fator),
                float(z),
            )
        )
    return projetados


def cor_por_profundidade(
    z: float, cor_base: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Escurece/clareia a cor conforme a profundidade (mais perto = mais claro)."""
    brilho = int(np.clip(170 + z * 75, 75, 255))
    return tuple(int(c * brilho / 255) for c in cor_base)


def desenhar_mao_3d(
    tela: np.ndarray,
    pontos: Sequence[tuple[int, int, float]],
    cor_base: tuple[int, int, int],
) -> None:
    """Desenha palma, ossos e juntas ordenados por profundidade.

    Args:
        tela: Imagem onde desenhar (modificada no lugar).
        pontos: Retorno de :func:`projetar_mao`.
        cor_base: Cor BGR da mão.
    """
    palma = np.array([(pontos[i][0], pontos[i][1]) for i in PALMA], np.int32)
    sobreposicao = tela.copy()
    cv2.fillConvexPoly(sobreposicao, palma, cor_base)
    cv2.addWeighted(sobreposicao, 0.22, tela, 0.78, 0, tela)

    for inicio, fim in sorted(LIGACOES, key=lambda lig: pontos[lig[0]][2]):
        x1, y1, z1 = pontos[inicio]
        x2, y2, z2 = pontos[fim]
        z_medio = (z1 + z2) / 2
        espessura = max(2, int(5 + z_medio * 2))
        cv2.line(
            tela,
            (x1, y1),
            (x2, y2),
            cor_por_profundidade(z_medio, cor_base),
            espessura,
            cv2.LINE_AA,
        )

    for x, y, z in sorted(pontos, key=lambda ponto: ponto[2]):
        raio = max(3, int(6 + z * 2))
        cv2.circle(
            tela, (x, y), raio, cor_por_profundidade(z, cor_base), -1, cv2.LINE_AA
        )
        cv2.circle(tela, (x, y), raio, (245, 245, 245), 1, cv2.LINE_AA)


def criar_painel(tamanho: tuple[int, int]) -> np.ndarray:
    """Cria o fundo escuro com linhas de grade e o título "MAO 3D"."""
    largura, altura = tamanho
    tela = np.full((altura, largura, 3), COR_FUNDO, dtype=np.uint8)
    for passo in range(1, 5):
        y = int(altura * passo / 5)
        cv2.line(tela, (0, y), (largura, y), (35, 39, 52), 1)
    cv2.putText(
        tela,
        "MAO 3D",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COR_TEXTO,
        2,
        cv2.LINE_AA,
    )
    return tela


def desenhar_quadro_sinal(
    vetor: Sequence[float],
    tamanho: tuple[int, int],
    angulo_y: float,
    legenda: str,
    angulo_x: float = 0.2,
) -> np.ndarray:
    """Renderiza um exemplo gravado (126 valores) com as duas mãos lado a lado.

    Args:
        vetor: Vetor de 126 valores.
        tamanho: ``(largura, altura)`` da imagem.
        angulo_y: Giro atual (a pré-visualização aumenta a cada quadro).
        legenda: Texto no canto inferior, por exemplo ``"oi - exemplo 3/80"``.
        angulo_x: Inclinação da câmera virtual.

    Returns:
        Imagem BGR (``numpy``) pronta para exibir.
    """
    largura, altura = tamanho
    tela = criar_painel(tamanho)
    esquerda, direita = dividir_maos(vetor)
    cores = (config.CORES_MAOS_BGR["esq"], config.CORES_MAOS_BGR["dir"])

    cv2.line(
        tela,
        (largura // 2, 55),
        (largura // 2, altura - 15),
        COR_DIVISORIA,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        tela,
        "ESQUERDA",
        (int(largura * 0.10), 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        cores[0],
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        tela,
        "DIREITA",
        (int(largura * 0.65), 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        cores[1],
        1,
        cv2.LINE_AA,
    )

    for mao, cor, posicao in (
        (esquerda, cores[0], POSICAO_ESQUERDA),
        (direita, cores[1], POSICAO_DIREITA),
    ):
        if eh_vetor_zerado(mao):
            continue
        pontos = projetar_mao(mao, tamanho, angulo_x, angulo_y, posicao)
        desenhar_mao_3d(tela, pontos, cor)

    cv2.putText(
        tela,
        legenda,
        (18, altura - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COR_TEXTO,
        1,
        cv2.LINE_AA,
    )
    return tela


__all__ = [
    "FEATURES_POR_MAO",
    "LIGACOES",
    "PALMA",
    "cor_por_profundidade",
    "criar_painel",
    "desenhar_mao_3d",
    "desenhar_quadro_sinal",
    "projetar_mao",
    "rotacionar",
]
