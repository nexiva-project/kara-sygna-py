"""Mão 3D esquelética controlada pelos landmarks do MediaPipe.

Mostra a webcam à esquerda e uma estrutura de mão em perspectiva à direita.
Controles: q fecha; A/D giram a câmera virtual; W/S inclinam a câmera virtual.
"""

import math

import cv2
import mediapipe as mp
import numpy as np


LIGACOES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)
PALMA = (0, 5, 9, 13, 17)


def rotacionar(pontos, angulo_x, angulo_y):
    """Gira a mão para que a tela revele sua profundidade."""
    cx, sx = math.cos(angulo_x), math.sin(angulo_x)
    cy, sy = math.cos(angulo_y), math.sin(angulo_y)
    rot_x = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    rot_y = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    return pontos @ (rot_y @ rot_x).T


def projetar_landmarks(
    landmarks, tamanho, angulo_x, angulo_y, posicao_horizontal=0.5,
    escala_relativa=1.0,
):
    """Converte landmarks normalizados em pontos de tela com perspectiva.

    ``posicao_horizontal`` reserva uma área do painel para cada mão. Assim,
    duas mãos não ficam uma em cima da outra quando aparecem ao mesmo tempo.
    """
    pontos = np.array(
        [[p.x - 0.5, 0.5 - p.y, -p.z * 1.8] for p in landmarks.landmark],
        dtype=np.float32,
    )
    pontos -= pontos[0]
    pontos = rotacionar(pontos, angulo_x, angulo_y)

    largura, altura = tamanho
    centro = np.array((largura * posicao_horizontal, altura * 0.62))
    distancia_camera = 2.6
    escala = min(largura, altura) * 1.25 * escala_relativa
    projetados = []

    for x, y, z in pontos:
        fator = distancia_camera / max(0.25, distancia_camera - z)
        projetados.append((
            int(centro[0] + x * escala * fator),
            int(centro[1] - y * escala * fator),
            float(z),
        ))
    return projetados


def cor_por_profundidade(z, cor_base):
    brilho = int(np.clip(170 + z * 75, 75, 255))
    return tuple(int(c * brilho / 255) for c in cor_base)


def desenhar_mao_3d(tela, pontos, cor_base):
    """Renderiza palma, ossos e juntas ordenados aproximadamente por z."""
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
            tela, (x1, y1), (x2, y2), cor_por_profundidade(z_medio, cor_base),
            espessura, cv2.LINE_AA,
        )

    for x, y, z in sorted(pontos, key=lambda ponto: ponto[2]):
        raio = max(3, int(6 + z * 2))
        cv2.circle(tela, (x, y), raio, cor_por_profundidade(z, cor_base), -1,
                   cv2.LINE_AA)
        cv2.circle(tela, (x, y), raio, (245, 245, 245), 1, cv2.LINE_AA)


def painel_3d(tamanho):
    largura, altura = tamanho
    tela = np.full((altura, largura, 3), (18, 20, 28), dtype=np.uint8)
    for passo in range(1, 5):
        y = int(altura * passo / 5)
        cv2.line(tela, (0, y), (largura, y), (35, 39, 52), 1)
    cv2.putText(tela, "MAO 3D", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (225, 225, 235), 2, cv2.LINE_AA)
    return tela


def main():
    # Mantém o mesmo comando de antes, mas agora abre a tela de ensino.
    from interface_sinais import main as abrir_interface

    abrir_interface()


if __name__ == "__main__":
    main()
