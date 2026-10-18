"""Ponto de entrada da interface gráfica do Kara Sygna."""

from __future__ import annotations

import logging
import queue
import sys
import threading
from typing import Any

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError:  # O Tkinter é um pacote separado em algumas distros.
    tk = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

MENSAGEM_SEM_TK = (
    "O Tkinter não está instalado. No Ubuntu/Debian, instale com:\n"
    "    sudo apt install python3-tk\n"
    "(depois reinicie o programa; se usar uma venv, "
    "ela precisa ter sido criada com o Python do sistema)."
)

FUNDO = "#0B1020"
PAINEL = "#111A2E"
CIANO = "#00D7FF"
TEXTO = "#F0F6FC"
# Cores escolhidas no Penpot para ler bem sobre o fundo holográfico.
TEXTO_DESCRICAO = "#FFFFFF"
TEXTO_DETALHE = "#000000"
VERMELHO = "#DE1616"


def _x11(raiz: tk.Misc) -> bool:
    """Diz se o Tk está no X11, inclusive no XWayland do Ubuntu."""
    return sys.platform == "linux" and raiz.tk.call("tk", "windowingsystem") == "x11"


def _sem_barra(raiz: tk.Tk) -> None:
    """Esconde a barra de título da tela de carregamento.

    No Windows ``overrideredirect`` tira a barra e a janela desenha na hora.
    No XWayland do Ubuntu a mesma chamada deixa o quadro em branco até a
    janela ser minimizada: o compositor não entrega um quadro novo para
    uma janela fora do gerenciador. O tipo ``splash`` pede a barra
    escondida ao gerenciador, que continua desenhando.
    """
    if _x11(raiz):
        try:
            raiz.attributes("-type", "splash")
            return
        except tk.TclError:
            logger.exception("Não foi possível usar a janela do tipo splash")
    raiz.overrideredirect(True)


def _liberar_recursos(
    camera: Any = None,
    detector_maos: Any = None,
    detector_rosto: Any = None,
    detector_corpo: Any = None,
) -> None:
    """Libera os recursos criados durante a inicialização."""
    if camera is not None:
        try:
            camera.liberar()
        except Exception:
            logger.exception("Erro ao liberar a câmera")

    if detector_rosto is not None:
        try:
            detector_rosto.fechar()
        except Exception:
            logger.exception("Erro ao fechar o FaceMesh")

    if detector_maos is not None:
        try:
            detector_maos.fechar()
        except Exception:
            logger.exception("Erro ao fechar o detector de mãos")

    if detector_corpo is not None:
        try:
            detector_corpo.fechar()
        except Exception:
            logger.exception("Erro ao fechar o detector do corpo")


def _carregar_em_segundo_plano(
    eventos: queue.Queue[tuple[str, Any]],
    cancelamento: threading.Event,
) -> None:
    """Prepara o ambiente sem bloquear a interface gráfica."""
    camera = None
    detector_maos = None
    detector_rosto = None
    detector_corpo = None

    try:
        eventos.put(("etapa", "Preparando o ambiente..."))

        from kara_sygna.aplicacao.inicializacao import preparar_ambiente

        preparar_ambiente()

        if cancelamento.is_set():
            return

        eventos.put(("etapa", "Carregando os componentes de visão..."))

        from kara_sygna.camera import (
            CameraIndisponivelError,
            abrir_camera,
        )
        from kara_sygna.visao.detector_corpo import criar_detector_corpo
        from kara_sygna.visao.detector_maos import DetectorMaos
        from kara_sygna.visao.detector_rosto import DetectorRosto

        detector_maos = DetectorMaos()

        if cancelamento.is_set():
            _liberar_recursos(
                detector_maos=detector_maos,
            )
            return

        detector_rosto = DetectorRosto()
        detector_corpo = criar_detector_corpo()

        if cancelamento.is_set():
            _liberar_recursos(
                detector_maos=detector_maos,
                detector_rosto=detector_rosto,
                detector_corpo=detector_corpo,
            )
            return

        eventos.put(("etapa", "Inicializando a câmera..."))
        camera = abrir_camera()

        if cancelamento.is_set():
            _liberar_recursos(
                camera=camera,
                detector_maos=detector_maos,
                detector_rosto=detector_rosto,
                detector_corpo=detector_corpo,
            )
            return

        eventos.put(("etapa", "Preparando a interface..."))

        from kara_sygna.aplicacao.controlador import ControladorSinais
        from kara_sygna.interface.janela_principal import JanelaPrincipal

        controlador = ControladorSinais()

        if cancelamento.is_set():
            _liberar_recursos(
                camera=camera,
                detector_maos=detector_maos,
                detector_rosto=detector_rosto,
                detector_corpo=detector_corpo,
            )
            return

        # A partir daqui, a interface principal assume os recursos.
        eventos.put(
            (
                "pronto",
                (
                    controlador,
                    camera,
                    detector_maos,
                    detector_rosto,
                    detector_corpo,
                    JanelaPrincipal,
                ),
            )
        )

    except Exception as erro:
        logger.exception("Falha ao iniciar o Kara Sygna")

        _liberar_recursos(
            camera=camera,
            detector_maos=detector_maos,
            detector_rosto=detector_rosto,
            detector_corpo=detector_corpo,
        )

        if not cancelamento.is_set():
            try:
                from kara_sygna.camera import CameraIndisponivelError

                titulo = (
                    "Câmera indisponível"
                    if isinstance(erro, CameraIndisponivelError)
                    else "Falha na inicialização"
                )
            except ImportError:
                titulo = "Falha na inicialização"

            eventos.put(("erro", (titulo, str(erro))))


class _TelaCarregamento:
    """Tela de carregamento desenhada num Canvas, sobre a imagem de fundo.

    No Canvas os textos ficam transparentes por cima da imagem; um ``Label``
    teria um retângulo de cor sólida atrás de cada texto.
    """

    LARGURA = 580
    ALTURA = 370
    MARGEM_X = 34
    LARGURA_CONTEUDO = 512
    #: Parte da imagem holográfica que aparece na tela (igual ao Penpot).
    RECORTE_FUNDO = (122, 137, 702, 507)

    def __init__(self, raiz: tk.Tk) -> None:
        from PIL import ImageTk

        from kara_sygna.interface import fundos

        self._raiz = raiz
        self._redesenhos: list[str] = []
        self._geracao_redesenho = 0
        # No XWayland a janela não pode aparecer antes dos widgets: o primeiro
        # quadro em branco fica preso até minimizar. Esconde agora e só mostra
        # em ``mostrar``, com o desenho pronto.
        if _x11(raiz):
            raiz.withdraw()

        raiz.title("Kara Sygna")
        raiz.resizable(False, False)
        raiz.configure(bg=FUNDO)

        # Centraliza a janela na tela. O tamanho da tela vale mesmo escondida;
        # um update aqui mapeava a janela vazia no Linux.
        x = (raiz.winfo_screenwidth() - self.LARGURA) // 2
        y = (raiz.winfo_screenheight() - self.ALTURA) // 2
        raiz.geometry(f"{self.LARGURA}x{self.ALTURA}+{x}+{y}")

        # Sem a barra de título, como uma splash screen.
        # A janela principal devolve a barra em ``restaurar_barra``.
        _sem_barra(raiz)
        # "Sempre na frente": aberta por um terminal dentro de outro programa
        # (VS Code, Claude...), o Windows não deixa a janela vir para a frente
        # e ela, sem barra nem ícone na barra de tarefas, ficava escondida.
        raiz.attributes("-topmost", True)
        raiz.lift()
        # Sem barra o Windows não dá foco sozinho; o foco é preciso para o Esc.
        raiz.after(100, raiz.focus_force)

        estilo = ttk.Style(raiz)
        estilo.theme_use("clam")
        estilo.configure(
            "Kara.Horizontal.TProgressbar",
            troughcolor=PAINEL,
            background=CIANO,
            bordercolor=PAINEL,
            lightcolor=CIANO,
            darkcolor=CIANO,
            thickness=7,
        )

        self._canvas = canvas = tk.Canvas(
            raiz,
            width=self.LARGURA,
            height=self.ALTURA,
            bg=FUNDO,
            highlightthickness=0,
            bd=0,
        )
        canvas.pack(fill="both", expand=True)

        imagem = fundos.carregar(fundos.HOLOGRAFICO).crop(self.RECORTE_FUNDO)
        self._foto_fundo = ImageTk.PhotoImage(imagem)  # guarda a referência
        canvas.create_image(0, 0, image=self._foto_fundo, anchor="nw")

        esquerda = self.MARGEM_X
        direita = self.MARGEM_X + self.LARGURA_CONTEUDO

        # Marca: quadrado "KS", nome e subtítulo.
        canvas.create_rectangle(esquerda, 24, esquerda + 59, 67, fill=CIANO, outline="")
        canvas.create_text(
            esquerda + 29.5,
            45.5,
            text="KS",
            fill=FUNDO,
            font=("Arial", 18, "bold"),
        )
        canvas.create_text(
            109,
            45,
            text="KARA SYGNA",
            fill=TEXTO,
            font=("Arial", 20, "bold"),
            anchor="w",
        )
        canvas.create_text(
            direita - 4,
            45,
            text="VISÃO COMPUTACIONAL",
            fill=CIANO,
            font=("Arial", 8, "bold"),
            anchor="e",
        )
        canvas.create_rectangle(esquerda, 85, direita, 87, fill=CIANO, outline="")

        canvas.create_text(
            esquerda + 2,
            127,
            text="Seu espaço de sinais está sendo preparado.",
            fill=TEXTO,
            font=("Arial", 15, "bold"),
            anchor="w",
            width=500,
        )
        canvas.create_text(
            esquerda + 3,
            162,
            text=(
                "Estamos inicializando os componentes necessários "
                "para reconhecer mãos e rosto."
            ),
            fill=TEXTO_DESCRICAO,
            font=("Arial", 10),
            anchor="w",
            width=490,
        )

        self.status = tk.StringVar(master=raiz, value="Preparando o ambiente...")
        texto_status = canvas.create_text(
            esquerda + 3,
            206,
            text=self.status.get(),
            fill=CIANO,
            font=("Arial", 10, "bold"),
            anchor="w",
        )
        self.status.trace_add(
            "write",
            lambda *_: canvas.itemconfigure(texto_status, text=self.status.get()),
        )

        self.progresso = ttk.Progressbar(
            canvas,
            mode="indeterminate",
            style="Kara.Horizontal.TProgressbar",
        )
        canvas.create_window(
            esquerda,
            227,
            window=self.progresso,
            anchor="nw",
            width=self.LARGURA_CONTEUDO,
            height=18,
        )
        self.progresso.start(12)

        self._detalhe = canvas.create_text(
            esquerda + 3,
            258,
            text="Aguarde enquanto o aplicativo é inicializado.",
            fill=TEXTO_DETALHE,
            font=("Arial", 9),
            anchor="nw",
            width=500,
        )

        self.botao_tentar = tk.Button(
            canvas,
            text="TENTAR NOVAMENTE",
            bg=CIANO,
            fg=FUNDO,
            activebackground="#65E8FF",
            activeforeground=FUNDO,
            relief="flat",
            cursor="hand2",
            font=("Arial", 10, "bold"),
            padx=12,
            pady=7,
        )
        # Sem barra de título não há o "X": no erro, este botão fecha o app.
        self.botao_fechar = tk.Button(
            canvas,
            text="FECHAR",
            bg=PAINEL,
            fg=TEXTO,
            activebackground="#1B2940",
            activeforeground=TEXTO,
            relief="flat",
            cursor="hand2",
            font=("Arial", 10, "bold"),
            padx=12,
            pady=7,
        )
        self._janelas_botoes: list[int] = []

    def mostrar(self) -> None:
        """Mostra a tela de carregamento já desenhada."""
        raiz = self._raiz
        if _x11(raiz):
            raiz.update_idletasks()
            raiz.deiconify()
            raiz.lift()
            # O primeiro map no XWayland ainda pode sair em branco. O redesenho
            # abaixo repete o que o botão minimizar faz, com o conteúdo pronto.
            self._agendar_redesenho()
            return
        raiz.lift()

    def _cancelar_redesenho(self) -> None:
        """Invalida um unmap/map que ainda não rodou."""
        self._geracao_redesenho += 1
        for identificador in self._redesenhos:
            try:
                self._raiz.after_cancel(identificador)
            except tk.TclError:
                pass
        self._redesenhos.clear()

    def _agendar_redesenho(self) -> None:
        """Desmapeia e mapeia de novo para o XWayland soltar o quadro.

        É o mesmo efeito de minimizar e restaurar, que era o único jeito de
        a janela aparecer no Ubuntu. A geometria é reaplicada porque alguns
        gerenciadores movem a janela ao escondê-la.
        """
        if not _x11(self._raiz):
            return

        self._cancelar_redesenho()
        geracao = self._geracao_redesenho
        raiz = self._raiz
        geometria = raiz.geometry()

        def esconder() -> None:
            if geracao != self._geracao_redesenho:
                return
            try:
                if not raiz.winfo_exists():
                    return
                raiz.withdraw()
            except tk.TclError:
                return
            self._redesenhos.append(raiz.after(20, mostrar))

        def mostrar() -> None:
            if geracao != self._geracao_redesenho:
                return
            try:
                if not raiz.winfo_exists():
                    return
                raiz.deiconify()
                if geometria:
                    raiz.geometry(geometria)
                raiz.lift()
            except tk.TclError:
                pass

        self._redesenhos.append(raiz.after(50, esconder))

    def definir_detalhe(self, texto: str, cor: str) -> None:
        """Troca o texto de detalhe (embaixo da barra de progresso)."""
        self._canvas.itemconfigure(self._detalhe, text=texto, fill=cor)

    def mostrar_botoes(self) -> None:
        """Mostra "TENTAR NOVAMENTE" e "FECHAR" logo abaixo do detalhe."""
        self.esconder_botoes()
        _, _, _, base = self._canvas.bbox(self._detalhe)
        x = self.MARGEM_X
        for botao in (self.botao_tentar, self.botao_fechar):
            self._janelas_botoes.append(
                self._canvas.create_window(x, base + 12, window=botao, anchor="nw")
            )
            x += botao.winfo_reqwidth() + 10

    def esconder_botoes(self) -> None:
        for janela in self._janelas_botoes:
            self._canvas.delete(janela)
        self._janelas_botoes.clear()

    def restaurar_barra(self) -> None:
        """Devolve a barra de título, antes de abrir a janela principal.

        No Linux a janela é escondida antes da troca. Mudar o tipo com ela
        visível deixa o miolo em branco até minimizar; ``exibir_janela_principal``
        mostra de novo quando os widgets da janela principal já existem.
        """
        self._cancelar_redesenho()
        raiz = self._raiz
        if _x11(raiz):
            raiz.withdraw()
            try:
                raiz.attributes("-type", "normal")
            except tk.TclError:
                logger.exception("Não foi possível devolver o tipo normal da janela")
            raiz.overrideredirect(False)
            return
        raiz.overrideredirect(False)
        raiz.lift()

    def exibir_janela_principal(self) -> None:
        """Mostra a janela principal e logo deixa de ficar sempre na frente."""
        raiz = self._raiz
        if _x11(raiz):
            raiz.update_idletasks()
            raiz.deiconify()
            self._agendar_redesenho()
        raiz.lift()
        raiz.after(800, lambda: self._soltar_frente())
        raiz.after(850, lambda: self._focar())

    def _soltar_frente(self) -> None:
        try:
            if self._raiz.winfo_exists():
                self._raiz.attributes("-topmost", False)
        except tk.TclError:
            pass

    def _focar(self) -> None:
        try:
            if self._raiz.winfo_exists():
                self._raiz.focus_force()
        except tk.TclError:
            pass


def executar() -> int:
    """Abre a tela de carregamento e prepara o aplicativo."""
    if tk is None:
        print(MENSAGEM_SEM_TK, file=sys.stderr)
        return 1

    raiz = tk.Tk()
    eventos: queue.Queue[tuple[str, Any]] = queue.Queue()

    cancelamento: threading.Event | None = None
    carregador: threading.Thread | None = None
    janela_principal = None
    iniciado = False

    tela = _TelaCarregamento(raiz)
    tela.mostrar()
    status, progresso = tela.status, tela.progresso

    def fechar() -> None:
        """Encerra a tela e sinaliza o cancelamento da inicialização."""
        if cancelamento is not None:
            cancelamento.set()

        # Se a interface ainda não assumiu os recursos, libera
        # qualquer conjunto completo que esteja aguardando na fila.
        while True:
            try:
                tipo, dados = eventos.get_nowait()
            except queue.Empty:
                break

            if tipo == "pronto":
                _, camera, detector_maos, detector_rosto, detector_corpo, _ = dados
                _liberar_recursos(
                    camera=camera,
                    detector_maos=detector_maos,
                    detector_rosto=detector_rosto,
                    detector_corpo=detector_corpo,
                )

        try:
            progresso.stop()
            raiz.destroy()
        except tk.TclError:
            pass

    raiz.protocol("WM_DELETE_WINDOW", fechar)
    # Sem barra de título, o Esc também fecha a tela de carregamento.
    raiz.bind("<Escape>", lambda _e: fechar())
    tela.botao_fechar.configure(command=fechar)

    def iniciar_carregamento() -> None:
        """Inicia uma tentativa de carregamento sem bloquear o Tkinter."""
        nonlocal cancelamento, carregador

        if cancelamento is not None:
            cancelamento.set()

        cancelamento = threading.Event()
        tela.esconder_botoes()
        tela.definir_detalhe(
            "Aguarde enquanto o aplicativo é inicializado.", TEXTO_DETALHE
        )
        status.set("Preparando o ambiente...")
        progresso.start(12)

        carregador = threading.Thread(
            target=_carregar_em_segundo_plano,
            args=(eventos, cancelamento),
            name="kara-sygna-inicializacao",
            daemon=True,
        )
        carregador.start()

    tela.botao_tentar.configure(command=iniciar_carregamento)

    def verificar_eventos() -> None:
        """Recebe os resultados da thread e atualiza a interface."""
        nonlocal iniciado, janela_principal

        if iniciado:
            return

        try:
            while True:
                tipo, dados = eventos.get_nowait()

                if tipo == "etapa":
                    status.set(dados)

                elif tipo == "erro":
                    titulo, mensagem = dados
                    progresso.stop()
                    status.set(titulo)
                    tela.definir_detalhe(
                        f"{mensagem}\n\nConfira a câmera e tente novamente.",
                        VERMELHO,
                    )
                    tela.mostrar_botoes()

                elif tipo == "pronto":
                    (
                        controlador,
                        camera,
                        detector_maos,
                        detector_rosto,
                        detector_corpo,
                        classe_janela,
                    ) = dados

                    progresso.stop()
                    status.set("Tudo pronto! Abrindo o Kara Sygna...")
                    raiz.update_idletasks()

                    # A janela principal reutiliza a mesma raiz Tk.
                    # A thread de carregamento nunca manipula widgets.
                    for widget in raiz.winfo_children():
                        widget.destroy()
                    raiz.unbind("<Escape>")
                    tela.restaurar_barra()

                    try:
                        janela_principal = classe_janela(
                            raiz,
                            controlador,
                            camera,
                            detector_maos,
                            detector_rosto,
                            detector_corpo,
                        )
                        iniciado = True
                        # Mostra só agora: no Linux a janela ficou escondida
                        # durante a montagem, para não mapear um quadro vazio.
                        tela.exibir_janela_principal()

                        # O loop Tkinter já está ativo; a janela principal
                        # agenda os próximos quadros sozinha.
                        janela_principal.iniciar_captura()

                    except Exception as erro:
                        # A tela de carregamento já foi desmontada, então
                        # não há como oferecer "tentar novamente" aqui.
                        logger.exception("Falha ao construir a interface principal")
                        _liberar_recursos(
                            camera=camera,
                            detector_maos=detector_maos,
                            detector_rosto=detector_rosto,
                            detector_corpo=detector_corpo,
                        )
                        iniciado = False
                        try:
                            raiz.deiconify()
                        except tk.TclError:
                            pass
                        messagebox.showerror(
                            "Inicialização",
                            f"Não foi possível abrir o Kara Sygna:\n\n{erro}",
                        )
                        try:
                            raiz.destroy()
                        except tk.TclError:
                            pass

                    return

        except queue.Empty:
            pass

        except Exception as erro:
            logger.exception("Erro ao finalizar a inicialização")
            progresso.stop()
            status.set("Não foi possível abrir a interface.")
            tela.definir_detalhe(str(erro), VERMELHO)
            tela.mostrar_botoes()

        try:
            raiz.after(100, verificar_eventos)
        except tk.TclError:
            pass

    iniciar_carregamento()
    raiz.after(100, verificar_eventos)
    raiz.mainloop()

    # Se a janela foi fechada durante o carregamento, dá à thread a chance
    # de perceber o cancelamento e liberar a câmera antes de sair.
    if carregador is not None and carregador.is_alive():
        carregador.join(timeout=5)

    return 0 if iniciado else 1


if __name__ == "__main__":
    sys.exit(executar())
