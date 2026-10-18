"""Imagens de fundo da interface (gradiente holográfico e textura escura)."""

from __future__ import annotations

import tkinter as tk
from functools import cache
from pathlib import Path

from PIL import Image, ImageTk

PASTA_IMAGENS = Path(__file__).parent / "imagens"
HOLOGRAFICO = "fundo_holografico.png"
TEXTURA = "textura_fundo.png"


@cache
def carregar(nome: str) -> Image.Image:
    """Abre uma imagem da pasta ``imagens`` (fica em cache)."""
    with Image.open(PASTA_IMAGENS / nome) as imagem:
        return imagem.convert("RGB")


def cobrir(
    imagem: Image.Image, largura: int, altura: int, alinhar_y: float = 0.5
) -> Image.Image:
    """Amplia/reduz a imagem para cobrir a área inteira e corta o que sobra.

    Args:
        imagem: Imagem original.
        largura: Largura da área, em pixels.
        altura: Altura da área, em pixels.
        alinhar_y: Qual parte da altura manter: 0 = topo, 0.5 = meio, 1 = base.
    """
    largura, altura = max(1, largura), max(1, altura)
    escala = max(largura / imagem.width, altura / imagem.height)
    nova = imagem.resize(
        (
            max(largura, round(imagem.width * escala)),
            max(altura, round(imagem.height * escala)),
        ),
        Image.Resampling.BILINEAR,
    )
    x = (nova.width - largura) // 2
    y = round((nova.height - altura) * alinhar_y)
    return nova.crop((x, y, x + largura, y + altura))


def aplicar_textura(container: tk.Frame) -> None:
    """Coloca a textura escura atrás dos widgets do container.

    Só aparece nos espaços entre os widgets (os painéis continuam opacos).
    A textura é um ruído uniforme, então não precisa alinhar entre containers.
    """
    fundo = tk.Label(container, bd=0, highlightthickness=0)
    fundo.place(x=0, y=0, relwidth=1, relheight=1)
    fundo.lower()

    def atualizar(evento: tk.Event) -> None:
        if evento.width < 2 or evento.height < 2:
            return
        textura = carregar(TEXTURA)
        if evento.width <= textura.width and evento.height <= textura.height:
            recorte = textura.crop((0, 0, evento.width, evento.height))
        else:
            recorte = cobrir(textura, evento.width, evento.height, 0)
        foto = ImageTk.PhotoImage(recorte)
        fundo.configure(image=foto)
        fundo.image = foto  # guarda a referência, senão o Tk descarta

    container.bind("<Configure>", atualizar, add="+")
