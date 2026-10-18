"""Os cursores da interface precisam existir no Windows, no Linux e no macOS.

Um cursor que só existe no Windows (ex.: "size_nw_se") faz o Tk dar erro no
Linux ("bad cursor spec") e a janela principal nem abre.
"""

import re
from pathlib import Path

# Cursores do X11 que o Tk aceita em todos os sistemas (lista parcial).
CURSORES_PORTATEIS = {
    "arrow",
    "bottom_right_corner",
    "crosshair",
    "fleur",
    "hand1",
    "hand2",
    "left_ptr",
    "sb_h_double_arrow",
    "sb_v_double_arrow",
    "watch",
    "xterm",
}
PASTA = Path(__file__).resolve().parents[1] / "kara_sygna" / "interface"


def test_cursores_escritos_no_codigo_sao_portateis():
    for arquivo in PASTA.glob("*.py"):
        for cursor in re.findall(
            r'cursor="([^"]+)"', arquivo.read_text(encoding="utf-8")
        ):
            assert cursor in CURSORES_PORTATEIS, f"{arquivo.name}: cursor {cursor!r}"


def test_cursor_de_redimensionar_depende_do_sistema(monkeypatch):
    import importlib
    import sys

    from kara_sygna.interface import janela_principal

    try:
        monkeypatch.setattr(sys, "platform", "linux")
        assert importlib.reload(janela_principal).CURSOR_REDIMENSIONAR == (
            "bottom_right_corner"
        )
        monkeypatch.setattr(sys, "platform", "win32")
        assert importlib.reload(janela_principal).CURSOR_REDIMENSIONAR == "size_nw_se"
    finally:
        monkeypatch.undo()
        importlib.reload(janela_principal)
