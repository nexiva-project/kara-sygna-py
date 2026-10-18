"""Conversão de imagens do OpenCV para o Tkinter."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageTk

MARGEM_PX = 10
TAMANHO_MINIMO_PX = 100


def quadro_para_foto(
    quadro_bgr: np.ndarray, largura_max: int, altura_max: int
) -> ImageTk.PhotoImage:
    """Converte um quadro BGR em imagem do Tk, reduzida para caber na área.

    A imagem nunca é ampliada além do tamanho original nem deformada.

    Args:
        quadro_bgr: Imagem do OpenCV.
        largura_max: Largura disponível, em pixels.
        altura_max: Altura disponível, em pixels.

    Returns:
        ``ImageTk.PhotoImage``. Guarde uma referência a ela (por exemplo em um
        atributo), senão o Tk a descarta e a imagem some.
    """
    imagem = Image.fromarray(cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2RGB))
    imagem.thumbnail(
        (max(1, largura_max - MARGEM_PX), max(1, altura_max - MARGEM_PX)),
        Image.Resampling.BILINEAR,
    )
    return ImageTk.PhotoImage(imagem)


def quadro_para_foto_exata(
    quadro_bgr: np.ndarray,
    largura: int,
    altura: int,
    fundo_bgr: tuple[int, int, int] = (18, 9, 5),
) -> ImageTk.PhotoImage:
    """Converte um quadro BGR em imagem do Tk com exatamente ``largura×altura``.

    O quadro é ampliado ou reduzido sem deformar; o espaço que sobra
    (proporções diferentes, como 4:3 em 16:9) é preenchido com ``fundo_bgr``.
    """
    altura_quadro, largura_quadro = quadro_bgr.shape[:2]
    escala = min(largura / largura_quadro, altura / altura_quadro)
    nova_largura = max(1, round(largura_quadro * escala))
    nova_altura = max(1, round(altura_quadro * escala))

    interpolacao = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    redimensionado = cv2.resize(
        quadro_bgr, (nova_largura, nova_altura), interpolation=interpolacao
    )

    tela = np.empty((altura, largura, 3), dtype=np.uint8)
    tela[:] = fundo_bgr
    x = (largura - nova_largura) // 2
    y = (altura - nova_altura) // 2
    tela[y : y + nova_altura, x : x + nova_largura] = redimensionado

    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(tela, cv2.COLOR_BGR2RGB)))
