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
