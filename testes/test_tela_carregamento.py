"""A tela de carregamento no Linux não pode abrir em branco.

No Ubuntu (Wayland + XWayland) ``overrideredirect`` mapeia a janela sem
quadro. Ela só aparecia depois de minimizar. No X11 a splash usa o tipo
``splash`` e fica escondida até o desenho existir.
"""

import pytest

pytest.importorskip("tkinter")


def _ligado(valor: object) -> bool:
    # Sem argumento, o Tk devolve None ou 0 quando a barra continua com o gerenciador.
    return valor in {1, "1"}


def _abrir():
    import tkinter as tk

    from kara_sygna.interface import iniciar

    try:
        raiz = tk.Tk()
    except tk.TclError:
        pytest.skip("sem display")
    raiz.withdraw()
    return iniciar, raiz, iniciar._TelaCarregamento(raiz)


def test_splash_no_x11_nao_usa_overrideredirect():
    iniciar, raiz, tela = _abrir()
    try:
        if not iniciar._x11(raiz):
            assert _ligado(raiz.overrideredirect())
            return

        assert not _ligado(raiz.overrideredirect())
        assert "splash" in str(raiz.attributes("-type"))
        # Ainda escondida: mostrar() é quem mapeia, com o canvas já cheio.
        assert raiz.winfo_viewable() == 0
        assert tela._canvas.find_all()
    finally:
        raiz.destroy()


def test_troca_para_a_janela_principal_devolve_a_barra_escondida():
    iniciar, raiz, tela = _abrir()
    try:
        if not iniciar._x11(raiz):
            tela.restaurar_barra()
            assert not _ligado(raiz.overrideredirect())
            return

        tela.mostrar()
        tela.restaurar_barra()
        assert raiz.winfo_viewable() == 0
        assert "normal" in str(raiz.attributes("-type"))
        assert not _ligado(raiz.overrideredirect())

        import tkinter as tk

        tk.Label(raiz, text="conteudo").pack()
        tela.exibir_janela_principal()
        assert raiz.winfo_viewable() == 1
        # O redesenho atrasado não pode reabrir por cima de um cancelamento.
        tela._cancelar_redesenho()
        assert raiz.winfo_viewable() == 1
    finally:
        raiz.destroy()


def test_redesenho_da_janela_principal_mantem_tamanho():
    iniciar, raiz, tela = _abrir()
    try:
        if not iniciar._x11(raiz):
            return

        import tkinter as tk

        tela.restaurar_barra()
        tk.Frame(raiz, bg="#FF00AA", width=800, height=600).pack(
            fill="both", expand=True
        )
        raiz.geometry("800x600+80+80")
        tela.exibir_janela_principal()
        raiz.after(250, raiz.quit)
        raiz.mainloop()
        assert raiz.winfo_viewable() == 1
        assert "normal" in str(raiz.attributes("-type"))
        assert raiz.winfo_width() >= 700
        assert raiz.winfo_height() >= 500
    finally:
        raiz.destroy()


def test_redesenho_deixa_a_splash_visivel():
    iniciar, raiz, tela = _abrir()
    try:
        if not iniciar._x11(raiz):
            return
        tela.mostrar()
        raiz.after(200, raiz.quit)
        raiz.mainloop()
        assert raiz.winfo_viewable() == 1
        assert raiz.winfo_width() >= tela.LARGURA - 2
        assert raiz.winfo_height() >= tela.ALTURA - 2
        assert "splash" in str(raiz.attributes("-type"))
    finally:
        raiz.destroy()
