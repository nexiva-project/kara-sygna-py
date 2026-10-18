"""Exporta as janelas do Kara Sygna como SVG, para abrir e editar no Figma.

Monta as janelas de verdade (com câmera e IA de mentira), lê a posição,
cor e texto de cada widget do Tkinter e desenha tudo em SVG. No Figma,
arraste o arquivo para dentro do canvas: cada widget vira uma camada.

Uso:
    python -m testes.manuais.exportar_svg [pasta_de_saida]
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import scrolledtext, ttk
from unittest import mock
from xml.sax.saxutils import escape

from kara_sygna.aplicacao.controlador import InfoSinal
from kara_sygna.interface import janela_principal as jp

PASTA_PADRAO = Path(__file__).resolve().parents[2] / "documento" / "figma"
TAMANHO_PRINCIPAL = (1440, 900)

SINAIS_EXEMPLO = [
    InfoSinal("olá", 42, True),
    InfoSinal("obrigado", 35, True),
    InfoSinal("tudo bem", 28, True),
    InfoSinal("água", 12, False),
]

LOG_EXEMPLO = [
    ("18:24:55 [APP] Pasta de dados: armazenamento", "APP"),
    ("18:24:56 [VISION] MediaPipe Hands inicializado.", "VISION"),
    ("18:24:56 [VISION] FaceMesh inicializado: 468 pontos na malha.", "VISION"),
    ("18:24:57 [CAMERA] Câmera conectada: câmera 0 via DirectShow (640x480)", "CAMERA"),
    ("18:24:58 [APP] Interface iniciada.", "APP"),
    ("18:25:02 [DATA] 42 exemplos de “olá” carregados.", "DATA"),
    ("18:25:03 [MODEL] Modelo carregado: 4 sinais.", "MODEL"),
    ("18:25:09 [WARNING] Mão saiu do quadro durante o ensino.", "WARNING"),
]


# ------------------------------------------------------------------ utilidades


def _cor(widget: tk.Misc, nome: str) -> str | None:
    """Converte qualquer cor do Tk (inclusive 'SystemButtonFace') em #RRGGBB."""
    if not nome:
        return None
    try:
        r, g, b = widget.winfo_rgb(nome)
    except tk.TclError:
        return None
    return f"#{r >> 8:02X}{g >> 8:02X}{b >> 8:02X}"


def _opcao(widget: tk.Misc, nome: str, padrao=None):
    try:
        return widget.cget(nome)
    except tk.TclError:
        return padrao


class _Fonte:
    """Fonte do Tk com as medidas necessárias para posicionar o texto."""

    def __init__(self, widget: tk.Misc, descricao) -> None:
        self.tk = tkfont.Font(root=widget, font=descricao)
        atual = self.tk.actual()
        self.familia = atual["family"]
        self.negrito = atual["weight"] == "bold"
        tamanho = atual["size"]
        # Tamanho positivo é em pontos; negativo, em pixels.
        escala = float(widget.tk.call("tk", "scaling"))
        self.px = -tamanho if tamanho < 0 else tamanho * escala
        self.ascendente = self.tk.metrics("ascent")
        self.altura_linha = self.tk.metrics("linespace")

    def medir(self, texto: str) -> int:
        return self.tk.measure(texto)

    def atributos(self) -> str:
        peso = ' font-weight="bold"' if self.negrito else ""
        return f'font-family="{escape(self.familia)}" font-size="{self.px:.1f}"{peso}'


def _quebrar(texto: str, fonte: _Fonte, largura: int) -> list[str]:
    """Quebra o texto como o wraplength do Tk (por palavras)."""
    linhas: list[str] = []
    for paragrafo in texto.split("\n"):
        if largura <= 0:
            linhas.append(paragrafo)
            continue
        atual = ""
        for palavra in paragrafo.split(" "):
            tentativa = f"{atual} {palavra}" if atual else palavra
            if atual and fonte.medir(tentativa) > largura:
                linhas.append(atual)
                atual = palavra
            else:
                atual = tentativa
        linhas.append(atual)
    return linhas


def _nome_camada(widget: tk.Misc, texto: str = "") -> str:
    nome = type(widget).__name__
    if texto:
        nome += f" - {texto.splitlines()[0][:40]}"
    return escape(nome, {'"': "&quot;"})


# ------------------------------------------------------------------ desenho


class ExportadorSVG:
    """Percorre os widgets de uma janela e gera o SVG equivalente."""

    def __init__(self, janela: tk.Misc) -> None:
        self.janela = janela
        self.ox = janela.winfo_rootx()
        self.oy = janela.winfo_rooty()
        self.partes: list[str] = []

    def _pos(self, w: tk.Misc) -> tuple[int, int, int, int]:
        return (
            w.winfo_rootx() - self.ox,
            w.winfo_rooty() - self.oy,
            w.winfo_width(),
            w.winfo_height(),
        )

    def _ret(self, x, y, lg, al, cor, borda=None, espessura=0) -> None:
        if not cor and not borda:
            return
        traco = ""
        if borda and espessura:
            meio = espessura / 2
            x, y, lg, al = x + meio, y + meio, lg - espessura, al - espessura
            traco = f' stroke="{borda}" stroke-width="{espessura}"'
        self.partes.append(
            f'<rect x="{x}" y="{y}" width="{max(lg, 0)}" height="{max(al, 0)}" '
            f'fill="{cor or "none"}"{traco}/>'
        )

    def _texto(self, x, y, linhas, fonte: _Fonte, cor, ancora="start") -> None:
        for i, linha in enumerate(linhas):
            if not linha:
                continue
            base = y + fonte.ascendente + i * fonte.altura_linha
            self.partes.append(
                f'<text x="{x:.1f}" y="{base:.1f}" fill="{cor}" '
                f'{fonte.atributos()} text-anchor="{ancora}" '
                f'xml:space="preserve">{escape(linha)}</text>'
            )

    def _texto_no_widget(self, w, x, y, lg, al, texto) -> None:
        """Posiciona o texto como Label/Button fazem (anchor + justify)."""
        fonte = _Fonte(w, _opcao(w, "font"))
        cor = _cor(w, _opcao(w, "fg") or "black")
        bd = int(float(_opcao(w, "bd", 0) or 0))
        hl = int(float(_opcao(w, "highlightthickness", 0) or 0))
        padx = int(float(_opcao(w, "padx", 0) or 0)) + bd + hl
        pady = int(float(_opcao(w, "pady", 0) or 0)) + bd + hl
        linhas = _quebrar(texto, fonte, int(float(_opcao(w, "wraplength", 0) or 0)))

        largura_texto = max((fonte.medir(linha) for linha in linhas), default=0)
        altura_texto = len(linhas) * fonte.altura_linha

        ancora = str(_opcao(w, "anchor", "center"))
        if ancora == "center":
            ancora = ""  # "center" contém as letras "n" e "e"
        if "w" in ancora:
            bx = x + padx
        elif "e" in ancora:
            bx = x + lg - padx - largura_texto
        else:
            bx = x + (lg - largura_texto) / 2
        if "n" in ancora:
            by = y + pady
        elif "s" in ancora:
            by = y + al - pady - altura_texto
        else:
            by = y + (al - altura_texto) / 2

        justificar = str(_opcao(w, "justify", "center"))
        if justificar == "left":
            self._texto(bx, by, linhas, fonte, cor)
        elif justificar == "right":
            self._texto(bx + largura_texto, by, linhas, fonte, cor, "end")
        else:
            self._texto(bx + largura_texto / 2, by, linhas, fonte, cor, "middle")

    def _widget(self, w: tk.Misc) -> None:
        if not w.winfo_ismapped() or isinstance(w, tk.Menu):
            return
        x, y, lg, al = self._pos(w)
        fundo = _cor(w, _opcao(w, "bg") or _opcao(w, "background") or "")
        hl = int(float(_opcao(w, "highlightthickness", 0) or 0))
        borda = _cor(w, _opcao(w, "highlightbackground") or "") if hl else None

        texto = ""
        if isinstance(w, (tk.Label, tk.Button)):
            variavel = _opcao(w, "textvariable")
            texto = w.getvar(str(variavel)) if str(variavel) else _opcao(w, "text")
            texto = str(texto or "")

        self.partes.append(f'<g id="{_nome_camada(w, texto)}">')

        if isinstance(w, ttk.Sizegrip):
            self._ret(x, y, lg, al, "#0D1525")
            for dx, dy in ((10, 14), (14, 10), (14, 14), (6, 14), (10, 10), (14, 6)):
                self.partes.append(
                    f'<rect x="{x + dx}" y="{y + dy}" width="2" height="2" '
                    'fill="#7186A8"/>'
                )
        elif isinstance(w, tk.Scale):
            self._escala(w, x, y, lg, al, fundo)
        elif isinstance(w, tk.Scrollbar):
            self._ret(x, y, lg, al, "#E6E6E6")
            self._ret(x + 3, y + 18, lg - 6, max(al // 3, 20), "#C1C1C1")
        else:
            self._ret(x, y, lg, al, fundo, borda, hl)
            bd = int(float(_opcao(w, "bd", 0) or 0))
            if isinstance(w, tk.Button) and bd and _opcao(w, "relief") != "flat":
                # Botão padrão do sistema, com borda em relevo.
                self._ret(x + hl, y + hl, lg - 2 * hl, al - 2 * hl, fundo, "#ADADAD", 1)

        if texto:
            self._texto_no_widget(w, x, y, lg, al, texto)
        elif isinstance(w, tk.Entry):
            fonte = _Fonte(w, _opcao(w, "font"))
            linhas = [w.get()]
            self._texto(
                x + 6,
                y + (al - fonte.altura_linha) / 2,
                linhas,
                fonte,
                _cor(w, w.cget("fg")),
            )
        elif isinstance(w, tk.Text):
            self._caixa_texto(w, x, y, lg, al)
        elif isinstance(w, tk.Listbox):
            fonte = _Fonte(w, _opcao(w, "font"))
            cor = _cor(w, w.cget("fg"))
            for i, item in enumerate(w.get(0, tk.END)):
                self._texto(x + 4, y + 3 + i * fonte.altura_linha, [item], fonte, cor)

        for filho in self._filhos(w):
            self._widget(filho)

        self.partes.append("</g>")

    @staticmethod
    def _filhos(w: tk.Misc) -> list[tk.Misc]:
        filhos = [f for f in w.winfo_children() if not isinstance(f, tk.Toplevel)]
        # Widgets dentro de um Canvas (create_window) aparecem como filhos
        # do próprio canvas; a ordem do Tk já é a ordem de empilhamento.
        return filhos

    def _caixa_texto(self, w: tk.Text, x, y, lg, al) -> None:
        fonte = _Fonte(w, _opcao(w, "font"))
        cor_padrao = _cor(w, w.cget("fg"))
        padx = int(float(w.cget("padx"))) + int(float(w.cget("bd")))
        pady = int(float(w.cget("pady"))) + int(float(w.cget("bd")))
        largura_util = lg - 2 * padx
        linha_y = y + pady
        ultima = int(w.index("end-1c").split(".")[0])
        for numero in range(1, ultima + 1):
            texto = w.get(f"{numero}.0", f"{numero}.end")
            cor = cor_padrao
            for etiqueta in w.tag_names(f"{numero}.0"):
                cor = _cor(w, w.tag_cget(etiqueta, "foreground")) or cor
            linhas = _quebrar(texto, fonte, largura_util) if texto else [""]
            if linha_y + len(linhas) * fonte.altura_linha > y + al:
                break
            self._texto(x + padx, linha_y, linhas, fonte, cor)
            linha_y += len(linhas) * fonte.altura_linha

    def _escala(self, w: tk.Scale, x, y, lg, al, fundo) -> None:
        self._ret(x, y, lg, al, fundo)
        fonte = _Fonte(w, _opcao(w, "font"))
        minimo, maximo = float(w.cget("from")), float(w.cget("to"))
        valor = float(w.get())
        trilho_y = y + fonte.altura_linha + 4
        trilho_al = max(al - (trilho_y - y) - 2, 10)
        self._ret(x + 2, trilho_y, lg - 4, trilho_al, _cor(w, w.cget("troughcolor")))
        fracao = 0 if maximo == minimo else (valor - minimo) / (maximo - minimo)
        cursor_lg = int(float(w.cget("sliderlength")))
        cursor_x = x + 2 + fracao * (lg - 4 - cursor_lg)
        self._ret(cursor_x, trilho_y + 1, cursor_lg, trilho_al - 2, "#3A4A66")
        rotulo = f"{valor:g}"
        self._texto(
            cursor_x + cursor_lg / 2,
            y,
            [rotulo],
            fonte,
            _cor(w, w.cget("fg")),
            "middle",
        )

    def gerar(self) -> str:
        lg, al = self.janela.winfo_width(), self.janela.winfo_height()
        fundo = _cor(self.janela, _opcao(self.janela, "bg") or "")
        self.partes = []
        self._ret(0, 0, lg, al, fundo)
        for filho in self._filhos(self.janela):
            self._widget(filho)
        corpo = "\n".join(self.partes)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{lg}" height="{al}" '
            f'viewBox="0 0 {lg} {al}">\n{corpo}\n</svg>\n'
        )


def svg_menu(menu: tk.Menu, destaque: int = 0) -> str:
    """Desenha um menu suspenso (Split Button) aberto."""
    fonte = _Fonte(menu, menu.cget("font"))
    fundo = _cor(menu, menu.cget("bg"))
    cor = _cor(menu, menu.cget("fg"))
    fundo_ativo = _cor(menu, menu.cget("activebackground"))
    cor_ativa = _cor(menu, menu.cget("activeforeground"))

    itens = []
    for i in range(menu.index("end") + 1):
        if menu.type(i) == "separator":
            itens.append(None)
        else:
            itens.append(menu.entrycget(i, "label"))

    largura = max(fonte.medir(t) for t in itens if t) + 48
    altura_item = fonte.altura_linha + 10
    altura_sep = 9
    altura = sum(altura_sep if t is None else altura_item for t in itens) + 4

    partes = [
        (
            f'<rect x="0" y="0" width="{largura}" height="{altura}" '
            f'fill="{fundo}" stroke="#243651" stroke-width="1"/>'
        )
    ]
    y = 2
    for i, item in enumerate(itens):
        if item is None:
            partes.append(
                f'<rect x="6" y="{y + 4}" width="{largura - 12}" height="1" fill="#30415C"/>'
            )
            y += altura_sep
            continue
        ativo = i == destaque
        partes.append(f'<g id="Item - {escape(item)}">')
        if ativo:
            partes.append(
                f'<rect x="2" y="{y}" width="{largura - 4}" height="{altura_item}" '
                f'fill="{fundo_ativo}"/>'
            )
        base = y + (altura_item - fonte.altura_linha) / 2 + fonte.ascendente
        partes.append(
            f'<text x="24" y="{base:.1f}" fill="{cor_ativa if ativo else cor}" '
            f'{fonte.atributos()} xml:space="preserve">{escape(item)}</text>'
        )
        partes.append("</g>")
        y += altura_item

    corpo = "\n".join(partes)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{largura}" height="{altura}" '
        f'viewBox="0 0 {largura} {altura}">\n{corpo}\n</svg>\n'
    )


# ------------------------------------------------------------------ montagem


def _janela_configuracoes_inteira(janela: tk.Toplevel) -> None:
    """Aumenta a janela de configurações para caber tudo, sem rolagem."""
    janela.update_idletasks()
    canvas = next(
        w
        for filho in janela.winfo_children()
        for w in filho.winfo_children()
        if isinstance(w, tk.Canvas)
    )
    conteudo = canvas.nametowidget(canvas.itemcget(canvas.find_all()[0], "window"))
    # Por padrão o Tk não deixa a janela passar da altura do monitor.
    janela.maxsize(10000, 10000)
    janela.geometry(f"{janela.winfo_width()}x{conteudo.winfo_reqheight() + 4}")


def exportar(pasta: Path) -> list[Path]:
    pasta.mkdir(parents=True, exist_ok=True)
    gerados: list[Path] = []

    def salvar(nome: str, conteudo: str) -> None:
        caminho = pasta / nome
        caminho.write_text(conteudo, encoding="utf-8")
        gerados.append(caminho)

    raiz = tk.Tk()
    controlador = mock.Mock()
    controlador.listar_sinais.return_value = SINAIS_EXEMPLO
    camera = mock.Mock(descricao="câmera 0 via DirectShow (640x480)")

    with (
        mock.patch.object(
            jp.JanelaPrincipal,
            "_carregar_config_visual",
            return_value=dict(jp.PADRAO_VISUAL),
        ),
        mock.patch.object(
            jp.JanelaPrincipal, "_carregar_config_suavizacao", return_value=(90, 90)
        ),
        mock.patch.object(jp.JanelaPrincipal, "_salvar_config_suavizacao"),
    ):
        janela = jp.JanelaPrincipal(raiz, controlador, camera, mock.Mock(), mock.Mock())

        # Conteúdo de exemplo, para o design não ficar vazio.
        janela._palavra.set("obrigado")
        janela._sinal_atual.set("OLÁ")
        janela._status.set("Reconhecendo... mostre o sinal para a câmera.")
        janela._texto.insert("1.0", "olá tudo bem obrigado ")
        janela._video.configure(text="IMAGEM DA CÂMERA", fg="#30415C")

        largura, altura = TAMANHO_PRINCIPAL
        raiz.geometry(f"{largura}x{altura}+0+0")
        raiz.update()
        salvar("01_janela_principal.svg", ExportadorSVG(raiz).gerar())

        # Menus dos Split Buttons.
        menus: list[tk.Menu] = []
        with mock.patch.object(
            jp.JanelaPrincipal,
            "_mostrar_menu",
            staticmethod(lambda m, _b: menus.append(m)),
        ):
            # Em 1440×900 a área do vídeo é pequena e o menu ficaria quase
            # vazio; para o design, mostra as opções de uma tela grande.
            with (
                mock.patch.object(janela._area_video, "winfo_width", return_value=1920),
                mock.patch.object(
                    janela._area_video, "winfo_height", return_value=1080
                ),
            ):
                janela._abrir_menu_tamanho_camera(janela._area_video)
            janela._abrir_menu_tamanho(janela._area_video)
        raiz.update()
        salvar("02_menu_imagem_da_camera.svg", svg_menu(menus[0]))
        salvar("03_menu_tamanho_da_janela.svg", svg_menu(menus[1], destaque=5))

        # Configurações.
        janela._abrir_configuracoes()
        config = janela._janela_configuracoes
        config.geometry("+0+0")
        _janela_configuracoes_inteira(config)
        config.update()
        salvar("04_configuracoes.svg", ExportadorSVG(config).gerar())
        janela._fechar_configuracoes()

        # Sinais aprendidos.
        janela._abrir_sinais()
        sinais = janela._janela_sinais
        sinais.geometry("+0+0")
        sinais._lista.selection_set(0)
        sinais.update()
        salvar("05_sinais_aprendidos.svg", ExportadorSVG(sinais).gerar())
        sinais.destroy()

        # Log.
        log = janela._log
        log.mostrar()
        log._janela.geometry("+0+0")
        caixa: scrolledtext.ScrolledText = log._texto
        caixa.configure(state=tk.NORMAL)
        for linha, etiqueta in LOG_EXEMPLO:
            caixa.insert(tk.END, linha + "\n", etiqueta)
        caixa.configure(state=tk.DISABLED)
        log._janela.update()
        salvar("06_log.svg", ExportadorSVG(log._janela).gerar())

        janela.fechar()

    return gerados


def main() -> int:
    pasta = Path(sys.argv[1]) if len(sys.argv) > 1 else PASTA_PADRAO
    for caminho in exportar(pasta):
        print(caminho)
    return 0


if __name__ == "__main__":
    sys.exit(main())
