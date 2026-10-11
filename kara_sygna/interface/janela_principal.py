"""Janela principal: câmera, painel de controle e texto dos sinais."""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import messagebox, scrolledtext

import cv2

from kara_sygna import config
from kara_sygna.aplicacao.controlador import ControladorSinais, Modo
from kara_sygna.aplicacao.erros import ErroAplicacao
from kara_sygna.camera import Camera
from kara_sygna.dados.repositorio import DadosInvalidosError
from kara_sygna.ia.modelo_io import ModeloInvalidoError
from kara_sygna.ia.treinamento import ResultadoTreino, TreinoImpossivelError
from kara_sygna.interface import tema
from kara_sygna.interface.imagem_tk import TAMANHO_MINIMO_PX, quadro_para_foto
from kara_sygna.interface.janela_log import JanelaLog
from kara_sygna.interface.janela_previsualizacao import JanelaPrevisualizacao
from kara_sygna.interface.janela_sinais import JanelaSinais
from kara_sygna.interface.tarefas import ExecutorTk
from kara_sygna.visao.caracteristicas import montar_vetor_duas_maos
from kara_sygna.visao.detector_maos import DetectorMaos, tem_maos

logger = logging.getLogger(__name__)

ERROS_DE_USUARIO = (
    ErroAplicacao,
    ModeloInvalidoError,
    DadosInvalidosError,
    TreinoImpossivelError,
)


class JanelaPrincipal:
    """Monta a interface e liga a câmera ao :class:`ControladorSinais`.

    Args:
        raiz: Janela raiz do Tk (``tk.Tk()``).
        controlador: Regras do programa.
        camera: Câmera já aberta.
        detector: Detector de mãos já criado.
    """

    def __init__(
        self,
        raiz: tk.Tk,
        controlador: ControladorSinais,
        camera: Camera,
        detector: DetectorMaos,
    ) -> None:
        """Monta a interface e guarda câmera, detector e controlador (veja a classe)."""
        self._raiz = raiz
        self._controlador = controlador
        self._camera = camera
        self._detector = detector

        self._executor = ExecutorTk(raiz)
        self._log = JanelaLog(raiz)
        self._janela_sinais: JanelaSinais | None = None
        self._janela_previa: JanelaPrevisualizacao | None = None
        self._foto = None
        self._encerrado = False
        self._ultimo_erro_quadro = ""

        raiz.title("Kara Sygna")
        raiz.geometry("1250x850")
        raiz.minsize(1050, 720)
        raiz.protocol("WM_DELETE_WINDOW", self.fechar)

        self._palavra = tk.StringVar()
        self._status = tk.StringVar(
            value=f"Escolha um modo para começar. ({camera.descricao})"
        )
        self._construir()
        logger.info("Interface iniciada.")

    # ------------------------------------------------------------------ layout

    def _caixa(
        self, mestre: tk.Misc, titulo: str, fundo: str, **opcoes
    ) -> tk.LabelFrame:
        """Cria uma caixa com título, no estilo do tema escuro."""
        return tk.LabelFrame(mestre, text=titulo, bg=fundo, fg=tema.TEXTO, **opcoes)

    def _construir(self) -> None:
        """Cria todos os widgets da janela."""
        fundo = tk.Frame(self._raiz, bg=tema.FUNDO)
        fundo.pack(fill="both", expand=True)
        tk.Label(
            fundo,
            text="KARA SYGNA",
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            font=tema.FONTE_TITULO,
        ).pack(anchor="w", padx=18, pady=(12, 5))

        superior = tk.Frame(fundo, bg=tema.FUNDO)
        superior.pack(fill="both", expand=True, padx=15, pady=(0, 8))
        self._construir_video(superior)
        self._construir_painel(superior)
        self._construir_texto(fundo)

    def _construir_video(self, mestre: tk.Misc) -> None:
        """Área da câmera. O rótulo é posicionado com ``place`` para que o tamanho
        da imagem nunca altere o tamanho da área (evita que ela cresça sozinha)."""
        caixa = self._caixa(
            mestre, " CÂMERA ", tema.FUNDO, font=tema.FONTE_CAIXA, bd=1, relief="solid"
        )
        caixa.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self._area_video = tk.Frame(caixa, bg=tema.FUNDO_VIDEO)
        self._area_video.pack(fill="both", expand=True, padx=5, pady=5)
        self._video = tk.Label(
            self._area_video,
            bg=tema.FUNDO_VIDEO,
            fg="#aaaaaa",
            text="Iniciando câmera...",
            font=("Arial", 14),
        )
        self._video.place(relx=0.5, rely=0.5, anchor="center")

    def _construir_painel(self, mestre: tk.Misc) -> None:
        """Painel lateral com os modos ensinar/sinal e o status."""
        painel = tk.Frame(mestre, bg=tema.FUNDO_PAINEL, width=300)
        painel.pack(side="right", fill="y")
        painel.pack_propagate(False)

        ensinar = self._caixa(
            painel, " MODO ENSINAR ", tema.FUNDO_PAINEL, padx=10, pady=10
        )
        ensinar.pack(fill="x", padx=10, pady=(10, 8))
        tk.Label(
            ensinar, text="Palavra do sinal:", bg=tema.FUNDO_PAINEL, fg=tema.TEXTO_SUAVE
        ).pack(anchor="w")
        campo = tk.Entry(ensinar, textvariable=self._palavra, font=("Arial", 11))
        campo.pack(fill="x", pady=(5, 8))
        campo.bind("<Return>", lambda _e: self._iniciar_ensino())
        tk.Button(
            ensinar, text="COMEÇAR A ENSINAR", command=self._iniciar_ensino, height=2
        ).pack(fill="x")
        tk.Button(ensinar, text="PARAR", command=self._parar).pack(
            fill="x", pady=(5, 0)
        )
        self._botao_treinar = tk.Button(
            ensinar, text="TREINAR IA", command=self._treinar, height=2
        )
        self._botao_treinar.pack(fill="x", pady=(8, 0))

        sinal = self._caixa(painel, " MODO SINAL ", tema.FUNDO_PAINEL, padx=10, pady=10)
        sinal.pack(fill="x", padx=10, pady=(0, 8))
        tk.Button(
            sinal,
            text="INICIAR RECONHECIMENTO",
            command=self._iniciar_reconhecimento,
            height=2,
        ).pack(fill="x")
        tk.Button(sinal, text="PARAR", command=self._parar).pack(fill="x", pady=(5, 0))

        tk.Button(
            painel, text="ABRIR SINAIS APRENDIDOS", command=self._abrir_sinais, height=2
        ).pack(fill="x", padx=10, pady=(2, 4))
        tk.Button(painel, text="ABRIR LOG", command=self._log.mostrar).pack(
            fill="x", padx=10, pady=(0, 8)
        )

        status = self._caixa(painel, " STATUS ", tema.FUNDO_PAINEL, padx=8, pady=8)
        status.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(
            status,
            textvariable=self._status,
            bg=tema.FUNDO_PAINEL,
            fg=tema.TEXTO_SUAVE,
            justify="left",
            anchor="w",
            wraplength=260,
        ).pack(fill="x")

    def _construir_texto(self, mestre: tk.Misc) -> None:
        """Caixa com o texto formado pelos sinais reconhecidos."""
        caixa = self._caixa(
            mestre,
            " TEXTO DOS SINAIS ",
            tema.FUNDO,
            font=("Arial", 11, "bold"),
            padx=10,
            pady=10,
        )
        caixa.pack(fill="x", padx=15, pady=(0, 15))
        self._texto = scrolledtext.ScrolledText(
            caixa,
            height=4,
            font=tema.FONTE_TEXTO_SINAIS,
            bg=tema.FUNDO_VIDEO,
            fg=tema.TEXTO,
            insertbackground=tema.TEXTO,
            wrap="word",
            relief="flat",
        )
        self._texto.pack(fill="both", expand=True)
        botoes = tk.Frame(caixa, bg=tema.FUNDO)
        botoes.pack(fill="x", pady=(8, 0))
        tk.Button(botoes, text="LIMPAR TEXTO", command=self._limpar_texto).pack(
            side="right"
        )

    # ------------------------------------------------------------- ações (botões)

    def _tratar_erro(self, erro: BaseException, titulo: str) -> None:
        """Mostra o erro: aviso simples para erros esperados, log completo para os demais."""
        if isinstance(erro, ERROS_DE_USUARIO):
            logger.warning("%s: %s", titulo, erro)
            messagebox.showwarning(titulo, str(erro))
        else:
            logger.error("%s", titulo, exc_info=erro)
            messagebox.showerror(titulo, f"{erro}\n\nVeja os detalhes em ABRIR LOG.")
        self._status.set(f"{titulo}: {erro}")

    def _bloqueado_por_treino(self) -> bool:
        """Avisa e devolve ``True`` se o treinamento ainda está rodando."""
        if self._executor.ocupado:
            messagebox.showinfo("Treinando", "Aguarde o treinamento terminar.")
        return self._executor.ocupado

    def _iniciar_ensino(self) -> None:
        """Botão COMEÇAR A ENSINAR."""
        if self._bloqueado_por_treino():
            return
        try:
            self._status.set(self._controlador.iniciar_ensino(self._palavra.get()))
        except Exception as erro:  # noqa: BLE001
            self._tratar_erro(erro, "Não foi possível ensinar")

    def _iniciar_reconhecimento(self) -> None:
        """Botão INICIAR RECONHECIMENTO."""
        if self._bloqueado_por_treino():
            return
        try:
            self._status.set(self._controlador.iniciar_reconhecimento())
        except Exception as erro:  # noqa: BLE001
            self._tratar_erro(erro, "Não foi possível reconhecer")

    def _parar(self) -> None:
        """Botão PARAR (dos dois modos)."""
        estava_ensinando = self._controlador.modo is Modo.ENSINANDO
        self._status.set(self._controlador.parar())
        if estava_ensinando:
            self._atualizar_sinais()

    def _treinar(self) -> None:
        """Botão TREINAR IA: treina em segundo plano para a câmera não congelar."""
        if self._executor.ocupado:
            return
        self._parar()
        self._botao_treinar.config(state="disabled")
        self._status.set("Treinando a IA... pode levar cerca de um minuto.")
        self._executor.executar(
            self._controlador.treinar,
            ao_concluir=self._treino_concluido,
            ao_falhar=self._treino_falhou,
            ao_progresso=self._status.set,
        )

    def _treino_concluido(self, resultado: ResultadoTreino) -> None:
        """Mostra o resultado do treinamento."""
        self._botao_treinar.config(state="normal")
        self._atualizar_sinais()
        if resultado.acuracia_validacao is None:
            detalhe = f"\n\nAviso: {resultado.relatorio}"
        else:
            detalhe = f"\n\nAcurácia na validação: {resultado.acuracia_validacao:.0%}"
        self._status.set(
            f"IA treinada com {resultado.amostras} exemplos e {len(resultado.classes)} sinais."
        )
        messagebox.showinfo(
            "IA treinada", "Pronto! Agora use INICIAR RECONHECIMENTO." + detalhe
        )

    def _treino_falhou(self, erro: BaseException) -> None:
        """Mostra o motivo da falha no treinamento."""
        self._botao_treinar.config(state="normal")
        self._tratar_erro(erro, "Não foi possível treinar")

    def _adicionar_texto(self, palavra: str) -> None:
        """Acrescenta uma palavra reconhecida ao texto."""
        self._texto.insert(tk.END, palavra + " ")
        self._texto.see(tk.END)
        logger.info("Sinal adicionado ao texto: %s", palavra)

    def _limpar_texto(self) -> None:
        """Botão LIMPAR TEXTO (também reinicia a votação)."""
        self._texto.delete("1.0", tk.END)
        self._controlador.limpar_reconhecimento()

    # ------------------------------------------------------------ janelas filhas

    def _atualizar_sinais(self) -> None:
        """Atualiza a lista de sinais, se a janela estiver aberta."""
        if self._janela_sinais is not None and self._janela_sinais.winfo_exists():
            self._janela_sinais.atualizar()

    def _abrir_sinais(self) -> None:
        """Abre (ou traz para a frente) a janela de sinais aprendidos."""
        if self._janela_sinais is not None and self._janela_sinais.winfo_exists():
            self._janela_sinais.deiconify()
            self._janela_sinais.lift()
            self._janela_sinais.atualizar()
            return
        self._janela_sinais = JanelaSinais(
            self._raiz,
            self._controlador.listar_sinais,
            self._ver_sinal,
            self._apagar_sinal,
        )

    def _ver_sinal(self, palavra: str) -> None:
        """Abre a pré-visualização 3D de um sinal."""
        try:
            reproducao = self._controlador.criar_reproducao(palavra)
        except Exception as erro:  # noqa: BLE001
            self._tratar_erro(erro, "Não foi possível mostrar o sinal")
            return
        if self._janela_previa is not None and self._janela_previa.winfo_exists():
            self._janela_previa.destroy()
        self._janela_previa = JanelaPrevisualizacao(self._raiz, reproducao)

    def _apagar_sinal(self, palavra: str) -> None:
        """Apaga um sinal depois da confirmação feita na janela de sinais."""
        try:
            self._status.set(self._controlador.apagar_sinal(palavra))
        except Exception as erro:  # noqa: BLE001
            self._tratar_erro(erro, "Não foi possível apagar")
        self._atualizar_sinais()

    # ------------------------------------------------------------ loop da câmera

    def iniciar(self) -> None:
        """Começa a ler a câmera e entra no laço de eventos do Tk."""
        self._atualizar_quadro()
        self._raiz.mainloop()

    def _atualizar_quadro(self) -> None:
        """Lê um quadro, processa e agenda o próximo (roda a cada ~15 ms)."""
        if self._encerrado:
            return
        try:
            quadro = self._camera.ler()
            if quadro is None:
                self._status.set("Não foi possível ler a câmera.")
                self._agendar(100)
                return
            self._processar_quadro(cv2.flip(quadro, 1))
        except Exception as erro:
            mensagem = f"{type(erro).__name__}: {erro}"
            if (
                mensagem != self._ultimo_erro_quadro
            ):  # evita repetir o mesmo erro 60 vezes por segundo
                logger.error("Erro no laço da câmera", exc_info=erro)
                self._ultimo_erro_quadro = mensagem
            self._status.set(f"Erro na câmera: {erro}")
        self._agendar(config.INTERVALO_QUADRO_MS)

    def _agendar(self, ms: int) -> None:
        """Agenda o próximo quadro, ignorando a janela já destruída."""
        try:
            self._raiz.after(ms, self._atualizar_quadro)
        except tk.TclError:
            pass

    def _processar_quadro(self, quadro) -> None:
        """Detecta as mãos, aplica as regras e mostra o quadro."""
        resultado_maos = self._detector.detectar(quadro)
        self._detector.desenhar(quadro, resultado_maos)
        vetor = (
            montar_vetor_duas_maos(resultado_maos) if tem_maos(resultado_maos) else None
        )

        try:
            resposta = self._controlador.processar_quadro(vetor)
        except Exception as erro:  # noqa: BLE001 - para o modo para não repetir o erro a cada quadro
            self._controlador.parar()
            self._tratar_erro(erro, "Erro no reconhecimento")
            resposta = None

        if resposta is not None:
            if resposta.status:
                self._status.set(resposta.status)
            if resposta.sinal_confirmado:
                self._adicionar_texto(resposta.sinal_confirmado)
            if resposta.texto_video:
                self._escrever_no_video(
                    quadro, resposta.texto_video, tema.COR_VIDEO_BGR[resposta.nivel]
                )
        self._mostrar(quadro)

    @staticmethod
    def _escrever_no_video(quadro, texto: str, cor: tuple[int, int, int]) -> None:
        """Escreve o texto em uma faixa escura no topo do quadro."""
        cv2.rectangle(
            quadro, (10, 10), (min(quadro.shape[1] - 10, 700), 55), (15, 15, 18), -1
        )
        cv2.putText(
            quadro, texto, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 2, cv2.LINE_AA
        )

    def _mostrar(self, quadro) -> None:
        """Exibe o quadro no ``Label`` da câmera, ajustado ao espaço disponível."""
        largura = self._area_video.winfo_width()
        altura = self._area_video.winfo_height()
        if largura < TAMANHO_MINIMO_PX or altura < TAMANHO_MINIMO_PX:
            largura, altura = config.LARGURA_AREA_CAMERA, config.ALTURA_AREA_CAMERA
        self._foto = quadro_para_foto(quadro, largura, altura)  # guarda a referência
        self._video.configure(image=self._foto, text="")

    # -------------------------------------------------------------------- fechar

    def fechar(self) -> None:
        """Encerra tudo: laço da câmera, MediaPipe, câmera e janelas."""
        if self._encerrado:
            return
        self._encerrado = True
        logger.info("Encerrando Kara Sygna.")
        self._detector.fechar()
        self._camera.liberar()
        self._log.fechar()
        try:
            self._raiz.destroy()
        except tk.TclError:
            pass
