"""Janela principal: câmera, painel de controle e configurações visuais."""

from __future__ import annotations

import copy
import json
import logging
import sys
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import colorchooser, messagebox, scrolledtext, simpledialog, ttk

import cv2
import numpy as np
from PIL import ImageTk

from kara_sygna import config
from kara_sygna.aplicacao.controlador import ControladorSinais, Modo
from kara_sygna.aplicacao.erros import (
    ErroAplicacao,
    PalavraInvalidaError,
    SemDadosError,
)
from kara_sygna.aplicacao.etapas import Etapa, montar_etapas
from kara_sygna.aplicacao.temporizador import (
    Evento,
    Fase,
    TemporizadorEnsino,
    ler_duracao,
)
from kara_sygna.camera import Camera
from kara_sygna.dados.repositorio import DadosInvalidosError
from kara_sygna.ia.modelo_io import ModeloInvalidoError
from kara_sygna.ia.treinamento import ResultadoTreino, TreinoImpossivelError
from kara_sygna.interface import fundos, tema
from kara_sygna.interface.imagem_tk import (
    TAMANHO_MINIMO_PX,
    quadro_para_foto,
    quadro_para_foto_exata,
)
from kara_sygna.interface.janela_log import JanelaLog
from kara_sygna.interface.janela_previsualizacao import JanelaPrevisualizacao
from kara_sygna.interface.janela_sinais import JanelaSinais
from kara_sygna.interface.tarefas import ExecutorTk
from kara_sygna.visao.detector_corpo import DetectorCorpo
from kara_sygna.visao.detector_maos import DetectorMaos, tem_maos
from kara_sygna.visao.detector_rosto import DetectorRosto, ResultadoRosto
from kara_sygna.visao.maos_pelo_corpo import conferir_maos
from kara_sygna.visao.memoria_maos import MemoriaMaos
from kara_sygna.visao.quadro import montar_vetor_quadro
from kara_sygna.visao.realce_pele import RealcePele
from kara_sygna.visao.suavizacao import SuavizadorPontos

logger = logging.getLogger(__name__)

ERROS_DE_USUARIO = (
    ErroAplicacao,
    ModeloInvalidoError,
    DadosInvalidosError,
    TreinoImpossivelError,
    SemDadosError,
)

PADRAO_VISUAL = {
    "maos_cor_linha_esquerda": "#00D7FF",
    "maos_cor_ponto_esquerda": "#00D7FF",
    "maos_cor_linha_direita": "#FF9F43",
    "maos_cor_ponto_direita": "#FF9F43",
    "maos_cor_texto": "#FFFFFF",
    "maos_espessura_linha": 2,
    "maos_tamanho_ponto": 3,
    "maos_espessura_texto": 2,
    "face_cor_malha": "#506450",
    "face_cor_contorno": "#FFDC00",
    "face_cor_ponto": "#FF0000",
    "face_cor_regiao_ponto": "#00FFFF",
    "face_cor_regiao_texto": "#00FFFF",
    "face_cor_status": "#00FF00",
    "face_cor_leve": "#00AAFF",
    "face_espessura_linha": 1,
    "face_tamanho_ponto": 1,
    "face_tamanho_regiao": 4,
    "face_espessura_texto": 1,
    "face_espessura_leve": 2,
    "video_cor_texto": "#FFFFFF",
    "texto_cor_sinais": "#FFFFFF",
    # Exibição: esconde o desenho sem afetar a detecção nem a IA.
    "mostrar_maos": True,
    "mostrar_rosto": True,
    "mostrar_linhas_rosto": True,
    "mostrar_pontos_rosto": True,
    "mostrar_nomes_rosto": True,
    # Tamanho da imagem da câmera: "ajustar" ou "LARGURAxALTURA".
    "video_tamanho": "ajustar",
}

# Resoluções oferecidas no Split Button de tamanho da janela.
RESOLUCOES_JANELA = (
    (640, 480),
    (720, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1366, 768),
    (1440, 900),
    (1600, 900),
    (1680, 1050),
    (1920, 1080),
    (2560, 1080),
    (2560, 1440),
    (3840, 2160),
)

# Menor tamanho permitido para a janela (a menor resolução da lista).
LARGURA_MINIMA_JANELA, ALTURA_MINIMA_JANELA = RESOLUCOES_JANELA[0]

# Cursor da alça de redimensionar. "size_nw_se" só existe no Windows; no Linux
# e no macOS o Tk dá erro com ele ("bad cursor spec") e a janela não abre.
CURSOR_REDIMENSIONAR = (
    "size_nw_se" if sys.platform.startswith("win") else "bottom_right_corner"
)


def area_rolavel(mestre: tk.Misc, fundo: str) -> tk.Frame:
    """Cria uma área com barra de rolagem vertical e devolve o quadro de dentro.

    A roda do mouse rola a área enquanto o ponteiro está sobre ela (Windows e
    macOS usam ``<MouseWheel>``; o Linux usa os botões 4 e 5). A barra só
    aparece quando o conteúdo não cabe.
    """
    canvas = tk.Canvas(mestre, bg=fundo, highlightthickness=0, bd=0)
    barra = tk.Scrollbar(mestre, orient="vertical", command=canvas.yview)
    conteudo = tk.Frame(canvas, bg=fundo)
    janela_interna = canvas.create_window((0, 0), window=conteudo, anchor="nw")

    def atualizar(_evento=None) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))
        cabe = conteudo.winfo_reqheight() <= canvas.winfo_height()
        if cabe and barra.winfo_ismapped():
            barra.pack_forget()
            canvas.yview_moveto(0)
        elif not cabe and not barra.winfo_ismapped():
            barra.pack(side="right", fill="y", before=canvas)

    def ajustar_largura(evento) -> None:
        canvas.itemconfigure(janela_interna, width=evento.width)
        atualizar()

    def dentro_da_area(evento) -> bool:
        """O ponteiro está sobre esta área (ou sobre algo dentro dela)?"""
        widget = canvas.winfo_containing(evento.x_root, evento.y_root)
        while widget is not None:
            if widget is canvas:
                return True
            widget = widget.master
        return False

    def rolar(evento) -> None:
        try:
            if not canvas.winfo_exists() or not barra.winfo_ismapped():
                return
            if not dentro_da_area(evento):
                return
        except (tk.TclError, KeyError):
            return
        if getattr(evento, "num", None) in (4, 5):
            passos = -1 if evento.num == 4 else 1
        else:
            passos = -1 if evento.delta > 0 else 1
        canvas.yview_scroll(passos * 3, "units")

    # Cada área confere se o ponteiro está sobre ela; "+" mantém as outras.
    for sequencia in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        canvas.bind_all(sequencia, rolar, add="+")

    canvas.configure(yscrollcommand=barra.set)
    conteudo.bind("<Configure>", atualizar)
    canvas.bind("<Configure>", ajustar_largura)
    canvas.pack(side="left", fill="both", expand=True)
    return conteudo


def _formatar_resolucao(largura: int, altura: int) -> str:
    return f"{largura}×{altura}"


def _ler_resolucao(valor: object) -> tuple[int, int] | None:
    """Converte "1280x720" em (1280, 720); None se não for da lista."""
    if not isinstance(valor, str):
        return None
    try:
        largura, altura = (int(parte) for parte in valor.split("x"))
    except ValueError:
        return None
    if (largura, altura) not in RESOLUCOES_JANELA:
        return None
    return largura, altura


class JanelaPrincipal:
    """Monta a interface e liga a câmera ao ControladorSinais."""

    def __init__(
        self,
        raiz: tk.Tk,
        controlador: ControladorSinais,
        camera: Camera,
        detector: DetectorMaos,
        detector_rosto: DetectorRosto,
        detector_corpo: DetectorCorpo | None = None,
    ) -> None:
        self._raiz = raiz
        self._controlador = controlador
        self._camera = camera
        self._detector = detector
        self._detector_rosto = detector_rosto
        self._detector_corpo = detector_corpo
        self._executor_corpo = ThreadPoolExecutor(1, thread_name_prefix="corpo")
        self._memoria_maos = MemoriaMaos()
        self._realce_pele = RealcePele()

        self._arquivo_config_suavizacao = Path.home() / ".kara_sygna_suavizacao.json"

        self._config_visual = self._carregar_config_visual()
        self._percentual_maos, self._percentual_rosto = (
            self._carregar_config_suavizacao()
        )

        self._variavel_suavizacao_maos = tk.IntVar(
            master=raiz, value=self._percentual_maos
        )
        self._variavel_suavizacao_rosto = tk.IntVar(
            master=raiz, value=self._percentual_rosto
        )

        self._janela_configuracoes: tk.Toplevel | None = None
        self._variaveis_tamanho: dict[str, tk.IntVar] = {}

        self._suavizador_maos = SuavizadorPontos(
            intensidade=0.30,
            zona_morta=0.002,
            limiar_movimento=0.012,
            resposta_movimento=0.85,
        )
        self._suavizador_rosto = SuavizadorPontos(
            intensidade=0.30,
            zona_morta=0.002,
            limiar_movimento=0.012,
            resposta_movimento=0.85,
        )
        self._aplicar_percentuais_suavizacao()

        self._executor = ExecutorTk(raiz)
        self._log = JanelaLog(raiz)
        self._janela_sinais: JanelaSinais | None = None
        self._janela_previa: JanelaPrevisualizacao | None = None
        # Estado do Split Button de tamanho da janela.
        # Último tamanho escolhido: (largura, altura) ou None para tela cheia.
        self._tamanho_janela: tuple[int, int] | None = None
        self._em_tela_cheia = False

        self._texto_tamanho_janela = tk.StringVar(master=raiz)
        self._texto_tamanho_camera = tk.StringVar(
            master=raiz,
            value=self._descrever_tamanho_camera(),
        )
        # Usado ao alternar de "ajustar à área" de volta para um tamanho fixo.
        self._ultimo_tamanho_camera = self._tamanho_camera() or RESOLUCOES_JANELA[0]
        self._foto = None
        self._encerrado = False
        self._ultimo_erro_quadro = ""

        raiz.title("Kara Sygna")

        # Detecta o tamanho do monitor sem alterar a resolução do sistema.
        largura_tela = raiz.winfo_screenwidth()
        altura_tela = raiz.winfo_screenheight()

        # Tamanho padrão (config.LARGURA/ALTURA_JANELA_PADRAO), sem passar da tela.
        largura_janela = min(largura_tela, config.LARGURA_JANELA_PADRAO)
        altura_janela = min(altura_tela, config.ALTURA_JANELA_PADRAO)

        # Centraliza a janela no monitor.
        pos_x = max(0, (largura_tela - largura_janela) // 2)
        pos_y = max(0, (altura_tela - altura_janela) // 2)

        raiz.geometry(f"{largura_janela}x{altura_janela}+{pos_x}+{pos_y}")
        self._tamanho_janela = (largura_janela, altura_janela)
        self._texto_tamanho_janela.set(
            _formatar_resolucao(largura_janela, altura_janela)
        )

        # A tela de carregamento usa a mesma raiz e a deixa travada;
        # aqui a janela volta a poder ser redimensionada e maximizada.
        raiz.resizable(True, True)

        # Permite diminuir a janela até a menor resolução da lista.
        raiz.minsize(
            min(LARGURA_MINIMA_JANELA, largura_tela),
            min(ALTURA_MINIMA_JANELA, altura_tela),
        )

        raiz.protocol("WM_DELETE_WINDOW", self.fechar)
        raiz.bind("<Configure>", self._ao_redimensionar_janela, add="+")
        # Enter começa a ensinar ou, se já está ensinando, para.
        raiz.bind("<Return>", self._alternar_ensino, add="+")
        raiz.bind("<KP_Enter>", self._alternar_ensino, add="+")

        self._palavra = tk.StringVar(master=raiz)
        # Segundos de gravação (campo abaixo da câmera); vazio = até parar.
        self._duracao = tk.StringVar(master=raiz)
        self._temporizador = TemporizadorEnsino()
        # Etapas do ensino que ainda faltam: (tipo, nome, pré-sinal, segundos).
        # Com pré-sinal são duas: primeiro o pré-sinal, depois o sinal.
        self._etapas: list[Etapa] = []
        self._etapa_atual: Etapa | None = None
        self._pre_ligado = tk.BooleanVar(master=raiz, value=False)
        self._pre_nome = tk.StringVar(master=raiz)
        self._sinal_atual = tk.StringVar(
            master=raiz,
            value="AGUARDANDO SINAL",
        )
        self._status = tk.StringVar(
            master=raiz,
            value=f"Escolha um modo para começar. ({camera.descricao})",
        )

        self._construir()
        logger.info("Interface iniciada.")

    # ------------------------------------------------------- salvar configurações

    def _carregar_config_visual(self) -> dict:
        """Carrega as preferências visuais, usando padrões para valores inválidos."""
        dados = dict(PADRAO_VISUAL)

        try:
            with self._arquivo_config_suavizacao.open("r", encoding="utf-8") as arquivo:
                salvo = json.load(arquivo)

            visual = salvo.get("visual", {})
            if isinstance(visual, dict):
                for chave, padrao in PADRAO_VISUAL.items():
                    valor = visual.get(chave, padrao)

                    if isinstance(padrao, bool):
                        if isinstance(valor, bool):
                            dados[chave] = valor
                        continue

                    if chave == "video_tamanho":
                        if _ler_resolucao(valor) is not None:
                            dados[chave] = valor
                        continue

                    if (
                        chave.endswith("cor_texto")
                        or chave.startswith("face_cor")
                        or chave == "video_cor_texto"
                        or chave == "texto_cor_sinais"
                    ):
                        if (
                            isinstance(valor, str)
                            and len(valor) == 7
                            and valor.startswith("#")
                        ):
                            try:
                                int(valor[1:], 16)
                                dados[chave] = valor
                            except ValueError:
                                pass
                    else:
                        try:
                            dados[chave] = max(1, min(8, int(valor)))
                        except (TypeError, ValueError):
                            pass

        except (OSError, ValueError, TypeError, AttributeError):
            pass

        return dados

    def _carregar_config_suavizacao(self) -> tuple[int, int]:
        """Carrega os percentuais de suavização."""
        try:
            with self._arquivo_config_suavizacao.open("r", encoding="utf-8") as arquivo:
                dados = json.load(arquivo)

            maos = int(dados.get("maos", 90))
            rosto = int(dados.get("rosto", 90))
        except (OSError, ValueError, TypeError, AttributeError):
            maos = rosto = 90

        return max(0, min(100, maos)), max(0, min(100, rosto))

    def _salvar_config_suavizacao(self) -> None:
        """Salva suavização e aparência no mesmo arquivo, sem perder opções."""
        dados = {
            "maos": self._percentual_maos,
            "rosto": self._percentual_rosto,
            "visual": self._config_visual,
        }

        try:
            with self._arquivo_config_suavizacao.open("w", encoding="utf-8") as arquivo:
                json.dump(dados, arquivo, indent=4, ensure_ascii=False)
        except OSError:
            logger.exception("Não foi possível salvar as configurações.")

    def _aplicar_percentuais_suavizacao(self) -> None:
        """Aplica suavização somente às cópias usadas no desenho."""
        for suavizador, percentual in (
            (self._suavizador_maos, self._percentual_maos),
            (self._suavizador_rosto, self._percentual_rosto),
        ):
            intensidade = 1.0 - (percentual / 100.0) * 0.70
            zona_morta = (percentual / 100.0) * 0.002
            suavizador.intensidade = intensidade
            suavizador.zona_morta = zona_morta
            suavizador.limpar()

    def _alterar_suavizacao_maos(self, valor: str) -> None:
        self._percentual_maos = max(0, min(100, int(float(valor))))
        self._aplicar_percentuais_suavizacao()
        self._salvar_config_suavizacao()

    def _alterar_suavizacao_rosto(self, valor: str) -> None:
        self._percentual_rosto = max(0, min(100, int(float(valor))))
        self._aplicar_percentuais_suavizacao()
        self._salvar_config_suavizacao()

    def _restaurar_suavizacao_padrao(self) -> None:
        self._percentual_maos = 90
        self._percentual_rosto = 90
        self._variavel_suavizacao_maos.set(90)
        self._variavel_suavizacao_rosto.set(90)
        self._aplicar_percentuais_suavizacao()
        self._salvar_config_suavizacao()

    # ------------------------------------------------------------ janela de ajustes

    def _abrir_configuracoes(self) -> None:
        """Abre a janela independente, com rolagem para todos os controles."""
        if (
            self._janela_configuracoes is not None
            and self._janela_configuracoes.winfo_exists()
        ):
            self._janela_configuracoes.deiconify()
            self._janela_configuracoes.lift()
            self._janela_configuracoes.focus_force()
            return

        janela = tk.Toplevel(self._raiz)
        self._janela_configuracoes = janela
        janela.title("Configurações - Kara Sygna")
        janela.geometry("570x760")
        janela.minsize(480, 500)
        janela.configure(bg=tema.FUNDO)
        janela.protocol("WM_DELETE_WINDOW", self._fechar_configuracoes)

        recipiente = tk.Frame(janela, bg=tema.FUNDO)
        recipiente.pack(fill="both", expand=True)
        conteudo = area_rolavel(recipiente, tema.FUNDO)

        tk.Label(
            conteudo,
            text="CONFIGURAÇÕES VISUAIS",
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            font=tema.FONTE_TITULO,
        ).pack(pady=(15, 5))

        tk.Label(
            conteudo,
            text="As alterações são aplicadas e salvas automaticamente.",
            bg=tema.FUNDO,
            fg=tema.TEXTO_SUAVE,
            wraplength=500,
        ).pack(pady=(0, 12))

        self._titulo_config(conteudo, "EXIBIÇÃO NA CÂMERA (liga/desliga)")
        tk.Label(
            conteudo,
            text=(
                "Só esconde o desenho. A detecção e a IA continuam "
                "funcionando normalmente."
            ),
            bg=tema.FUNDO,
            fg=tema.TEXTO_SUAVE,
            wraplength=500,
            justify="left",
        ).pack(fill="x", padx=20, pady=(0, 6))

        self._controle_interruptor(conteudo, "Mostrar mãos", "mostrar_maos")
        self._controle_interruptor(conteudo, "Mostrar rosto", "mostrar_rosto")
        self._controle_interruptor(conteudo, "Linhas do rosto", "mostrar_linhas_rosto")
        self._controle_interruptor(conteudo, "Pontos do rosto", "mostrar_pontos_rosto")
        self._controle_interruptor(
            conteudo, "Nomes e textos do rosto", "mostrar_nomes_rosto"
        )

        self._titulo_config(conteudo, "JANELA")
        linha_janela = tk.Frame(conteudo, bg=tema.FUNDO)
        linha_janela.pack(fill="x", padx=20, pady=3)

        tk.Label(
            linha_janela,
            text="Tamanho da janela",
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        grupo_janela = tk.Frame(linha_janela, bg=tema.FUNDO)
        grupo_janela.pack(side="right")
        self._criar_split_button(
            grupo_janela,
            self._texto_tamanho_janela,
            self._reaplicar_tamanho_janela,
            self._abrir_menu_tamanho,
        )

        self._titulo_config(conteudo, "MÃOS")
        self._controle_cor(
            conteudo, "Linhas da mão esquerda", "maos_cor_linha_esquerda"
        )
        self._controle_cor(
            conteudo, "Pontos da mão esquerda", "maos_cor_ponto_esquerda"
        )
        self._controle_cor(conteudo, "Linhas da mão direita", "maos_cor_linha_direita")
        self._controle_cor(conteudo, "Pontos da mão direita", "maos_cor_ponto_direita")
        self._controle_cor(conteudo, "Texto das mãos", "maos_cor_texto")
        self._controle_tamanho(
            conteudo, "Espessura das linhas das mãos", "maos_espessura_linha"
        )
        self._controle_tamanho(
            conteudo, "Tamanho dos pontos das mãos", "maos_tamanho_ponto"
        )
        self._controle_tamanho(
            conteudo, "Espessura do texto das mãos", "maos_espessura_texto"
        )

        self._titulo_config(conteudo, "ROSTO / FACEMESH")
        self._controle_cor(conteudo, "Linhas do rosto leve", "face_cor_leve")
        self._controle_tamanho(
            conteudo, "Espessura do rosto leve", "face_espessura_leve"
        )
        self._controle_cor(conteudo, "Malha facial", "face_cor_malha")
        self._controle_cor(conteudo, "Contorno facial", "face_cor_contorno")
        self._controle_cor(conteudo, "Pontos do rosto", "face_cor_ponto")
        self._controle_cor(
            conteudo,
            "Pontos de referência (nariz, testa etc.)",
            "face_cor_regiao_ponto",
        )
        self._controle_cor(
            conteudo, "Nomes das regiões faciais", "face_cor_regiao_texto"
        )
        self._controle_cor(conteudo, "Texto de estado do FaceMesh", "face_cor_status")
        self._controle_tamanho(
            conteudo, "Espessura das linhas do rosto", "face_espessura_linha"
        )
        self._controle_tamanho(
            conteudo, "Tamanho dos pontos do rosto", "face_tamanho_ponto"
        )
        self._controle_tamanho(
            conteudo,
            "Tamanho dos pontos de referência",
            "face_tamanho_regiao",
        )
        self._controle_tamanho(
            conteudo, "Espessura dos nomes faciais", "face_espessura_texto"
        )

        self._titulo_config(conteudo, "TEXTOS")
        self._controle_cor(
            conteudo, "Texto de reconhecimento sobre a câmera", "video_cor_texto"
        )
        self._controle_cor(
            conteudo, "Texto dos sinais reconhecidos", "texto_cor_sinais"
        )

        self._titulo_config(conteudo, "SUAVIZAÇÃO VISUAL")

        tk.Label(
            conteudo,
            text="Suavização do rosto (%)",
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            anchor="w",
        ).pack(fill="x", padx=20)

        tk.Scale(
            conteudo,
            from_=0,
            to=100,
            resolution=5,
            orient="horizontal",
            variable=self._variavel_suavizacao_rosto,
            command=self._alterar_suavizacao_rosto,
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            troughcolor=tema.FUNDO_PAINEL,
            highlightthickness=0,
        ).pack(fill="x", padx=20)

        tk.Label(
            conteudo,
            text="Suavização das mãos (%)",
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            anchor="w",
        ).pack(fill="x", padx=20)

        tk.Scale(
            conteudo,
            from_=0,
            to=100,
            resolution=5,
            orient="horizontal",
            variable=self._variavel_suavizacao_maos,
            command=self._alterar_suavizacao_maos,
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            troughcolor=tema.FUNDO_PAINEL,
            highlightthickness=0,
        ).pack(fill="x", padx=20)

        tk.Button(
            conteudo,
            text="RESTAURAR CORES, TAMANHOS E EXIBIÇÃO",
            command=self._restaurar_visual,
            bg=tema.FUNDO_PAINEL,
            fg=tema.TEXTO,
            relief="flat",
            cursor="hand2",
        ).pack(fill="x", padx=20, pady=(14, 5))

        tk.Button(
            conteudo,
            text="RESTAURAR SUAVIZAÇÃO (90%)",
            command=self._restaurar_suavizacao_padrao,
            bg=tema.FUNDO_PAINEL,
            fg=tema.TEXTO,
            relief="flat",
            cursor="hand2",
        ).pack(fill="x", padx=20, pady=(0, 18))

    def _titulo_config(self, mestre: tk.Misc, titulo: str) -> None:
        tk.Label(
            mestre,
            text=titulo,
            bg=tema.FUNDO_PAINEL,
            fg=tema.TEXTO,
            font=("Arial", 10, "bold"),
            anchor="w",
        ).pack(fill="x", padx=12, pady=(12, 5))

    def _controle_cor(
        self,
        mestre: tk.Misc,
        rotulo: str,
        chave: str,
    ) -> None:
        linha = tk.Frame(mestre, bg=tema.FUNDO)
        linha.pack(fill="x", padx=20, pady=3)

        tk.Label(
            linha,
            text=rotulo,
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        amostra = tk.Label(
            linha,
            text=self._config_visual[chave],
            bg=self._config_visual[chave],
            fg=self._cor_contraste(self._config_visual[chave]),
            width=10,
        )
        amostra.pack(side="right", padx=(5, 0))

        tk.Button(
            linha,
            text="Escolher cor",
            command=lambda: self._escolher_cor(chave, amostra),
            bg=tema.FUNDO_PAINEL,
            fg=tema.TEXTO,
            relief="flat",
            cursor="hand2",
        ).pack(side="right")

    @staticmethod
    def _cor_contraste(cor: str) -> str:
        try:
            r, g, b = (
                int(cor[1:3], 16),
                int(cor[3:5], 16),
                int(cor[5:7], 16),
            )
            return (
                "#000000" if (r * 299 + g * 587 + b * 114) / 1000 > 140 else "#FFFFFF"
            )
        except (ValueError, TypeError, IndexError):
            return "#FFFFFF"

    def _escolher_cor(self, chave: str, amostra: tk.Label) -> None:
        escolhido = colorchooser.askcolor(
            color=self._config_visual[chave],
            title="Escolher cor",
            parent=self._janela_configuracoes,
        )
        if not escolhido or not escolhido[1]:
            return

        self._config_visual[chave] = escolhido[1]
        amostra.configure(
            text=escolhido[1],
            bg=escolhido[1],
            fg=self._cor_contraste(escolhido[1]),
        )
        self._aplicar_cor_texto()
        self._salvar_config_suavizacao()

    def _controle_interruptor(
        self,
        mestre: tk.Misc,
        rotulo: str,
        chave: str,
    ) -> None:
        """Botão liga/desliga: só muda o que aparece na câmera."""
        linha = tk.Frame(mestre, bg=tema.FUNDO)
        linha.pack(fill="x", padx=20, pady=3)

        tk.Label(
            linha,
            text=rotulo,
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

        botao = tk.Button(
            linha,
            width=11,
            relief="flat",
            fg="#FFFFFF",
            activeforeground="#FFFFFF",
            cursor="hand2",
        )
        botao.pack(side="right")

        def atualizar_botao() -> None:
            ligado = bool(self._config_visual.get(chave, True))
            cor = "#2E7D32" if ligado else "#6B6B6B"
            botao.configure(
                text="LIGADO" if ligado else "DESLIGADO",
                bg=cor,
                activebackground=cor,
            )

        def alternar() -> None:
            self._config_visual[chave] = not bool(self._config_visual.get(chave, True))
            atualizar_botao()
            self._salvar_config_suavizacao()

        botao.configure(command=alternar)
        atualizar_botao()

    def _controle_tamanho(
        self,
        mestre: tk.Misc,
        rotulo: str,
        chave: str,
    ) -> None:
        linha = tk.Frame(mestre, bg=tema.FUNDO)
        linha.pack(fill="x", padx=20, pady=3)

        tk.Label(
            linha,
            text=rotulo,
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            anchor="w",
        ).pack(fill="x")

        variavel = tk.IntVar(
            master=self._raiz,
            value=int(self._config_visual[chave]),
        )
        self._variaveis_tamanho[chave] = variavel

        tk.Scale(
            linha,
            from_=1,
            to=8,
            resolution=1,
            orient="horizontal",
            variable=variavel,
            command=lambda valor, k=chave: self._alterar_tamanho(k, valor),
            bg=tema.FUNDO,
            fg=tema.TEXTO,
            troughcolor=tema.FUNDO_PAINEL,
            highlightthickness=0,
        ).pack(fill="x")

    def _alterar_tamanho(self, chave: str, valor: str) -> None:
        self._config_visual[chave] = max(1, min(8, int(float(valor))))
        self._salvar_config_suavizacao()

    def _restaurar_visual(self) -> None:
        self._config_visual = dict(PADRAO_VISUAL)

        self._salvar_config_suavizacao()

        if (
            self._janela_configuracoes is not None
            and self._janela_configuracoes.winfo_exists()
        ):
            self._fechar_configuracoes()
            self._abrir_configuracoes()

        self._aplicar_cor_texto()
        self._texto_tamanho_camera.set(self._descrever_tamanho_camera())

    def _aplicar_cor_texto(self) -> None:
        """Aplica a cor escolhida à caixa de texto dos sinais."""
        texto = getattr(self, "_texto", None)
        if texto is not None:
            try:
                cor = self._config_visual["texto_cor_sinais"]
                texto.configure(fg=cor, insertbackground=cor)
            except tk.TclError:
                pass

    def _fechar_configuracoes(self) -> None:
        janela = self._janela_configuracoes
        self._janela_configuracoes = None
        if janela is not None:
            try:
                janela.destroy()
            except tk.TclError:
                pass

    # ------------------------------------------------------------------ layout

    def _caixa(
        self,
        mestre: tk.Misc,
        titulo: str,
        fundo: str,
        **opcoes,
    ) -> tk.LabelFrame:
        return tk.LabelFrame(
            mestre,
            text=titulo,
            bg=fundo,
            fg=tema.TEXTO,
            **opcoes,
        )

    def _construir(self) -> None:
        """Monta a interface principal com visual escuro e detalhes em ciano."""
        self._raiz.configure(bg="#080D18")

        fundo = tk.Frame(self._raiz, bg="#080D18")
        fundo.pack(fill="both", expand=True)

        self._construir_cabecalho(fundo)

        # Linha decorativa
        tk.Frame(fundo, bg="#00D7FF", height=2).pack(fill="x")

        # Área principal
        conteudo = tk.Frame(fundo, bg="#080D18")
        conteudo.pack(fill="both", expand=True, padx=18, pady=16)

        superior = tk.Frame(conteudo, bg="#080D18")
        superior.pack(fill="both", expand=True)

        self._construir_video(superior)
        self._construir_painel(superior)
        self._construir_texto(conteudo)

        # Textura escura nos espaços entre os painéis.
        for container in (fundo, conteudo, superior):
            fundos.aplicar_textura(container)

        # Rodapé
        rodape = tk.Frame(fundo, bg="#0D1525", height=28)
        rodape.pack(fill="x", side="bottom")
        rodape.pack_propagate(False)

        tk.Label(
            rodape,
            text="KARA SYGNA  •  ACESSIBILIDADE EM LIBRAS",
            bg="#0D1525",
            fg="#7186A8",
            font=("Arial", 8),
        ).pack(side="left", padx=18, pady=5)

        # Alça no canto inferior direito para redimensionar arrastando.
        estilo = ttk.Style(self._raiz)
        estilo.configure("Kara.TSizegrip", background="#0D1525")
        ttk.Sizegrip(rodape, style="Kara.TSizegrip", cursor=CURSOR_REDIMENSIONAR).pack(
            side="right", anchor="se"
        )

        tk.Label(
            rodape,
            text="PROCESSAMENTO LOCAL",
            bg="#0D1525",
            fg="#00D7FF",
            font=("Arial", 8, "bold"),
        ).pack(side="right", padx=18, pady=5)

    def _construir_cabecalho(self, mestre: tk.Misc) -> None:
        """Cabeçalho com a faixa holográfica de fundo.

        É um Canvas para os textos ficarem por cima da imagem sem um
        retângulo de cor atrás de cada um.
        """
        altura = 76
        altura_faixa = 73  # os 3px de baixo mostram a textura escura
        canvas = tk.Canvas(
            mestre, height=altura, bg="#080D18", highlightthickness=0, bd=0
        )
        canvas.pack(fill="x")

        textura = canvas.create_image(0, 0, anchor="nw")
        faixa = canvas.create_image(0, 0, anchor="nw")

        canvas.create_text(
            36.5,
            37.5,
            text="◈",
            fill="#00D7FF",
            font=("Arial", 27, "bold"),
        )
        canvas.create_text(
            61,
            28,
            text="KARA SYGNA",
            fill="#F1F7FF",
            font=("Arial", 19, "bold"),
            anchor="w",
        )
        canvas.create_text(
            61,
            57,
            text="RECONHECIMENTO DE SINAIS",
            fill="#7D93B5",
            font=("Arial", 8, "bold"),
            anchor="w",
        )
        status = canvas.create_text(
            0,
            37.5,
            text="●  SISTEMA INICIALIZADO",
            fill="#5CE1A5",
            font=("Arial", 9, "bold"),
            anchor="e",
        )

        def redimensionar(evento: tk.Event) -> None:
            largura = evento.width
            if largura < 2:
                return
            # A imagem ocupa a largura toda; aparece só a parte de baixo dela.
            holografico = fundos.carregar(fundos.HOLOGRAFICO)
            altura_total = round(holografico.height * largura / holografico.width)
            recorte = fundos.cobrir(
                holografico, largura, max(altura_total, altura_faixa)
            ).crop((0, altura_total - altura_faixa, largura, altura_total))
            fundo_textura = fundos.carregar(fundos.TEXTURA).crop(
                (0, 0, largura, altura)
            )
            # Guarda as referências, senão o Tk descarta as imagens.
            self._fotos_cabecalho = (
                ImageTk.PhotoImage(fundo_textura),
                ImageTk.PhotoImage(recorte),
            )
            canvas.itemconfigure(textura, image=self._fotos_cabecalho[0])
            canvas.itemconfigure(faixa, image=self._fotos_cabecalho[1])
            canvas.coords(status, largura - 24, 37.5)

        canvas.bind("<Configure>", redimensionar)

    # ----------------------------------------------------- tamanho da janela

    def _criar_split_button(
        self,
        mestre: tk.Misc,
        texto: tk.StringVar,
        ao_clicar,
        ao_abrir_menu,
    ) -> tk.Button:
        """Cria um Split Button: ação principal à esquerda e menu na seta.

        ``ao_abrir_menu`` recebe o botão da seta, para posicionar o menu.
        Retorna o botão da seta.
        """
        grupo = tk.Frame(
            mestre,
            bg="#00BFD9",
            highlightbackground="#00D7FF",
            highlightthickness=1,
        )
        grupo.pack(side="left")

        tk.Button(
            grupo,
            textvariable=texto,
            command=ao_clicar,
            bg="#13243A",
            fg="#EAF2FF",
            activebackground="#1B3954",
            activeforeground="#FFFFFF",
            font=("Arial", 9, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=12,
            pady=7,
        ).pack(side="left")

        seta = tk.Button(
            grupo,
            text="▼",
            bg="#00BFD9",
            fg="#07111E",
            activebackground="#48E5F5",
            activeforeground="#07111E",
            font=("Arial", 9, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=10,
            pady=7,
        )
        seta.configure(command=lambda: ao_abrir_menu(seta))
        seta.pack(side="left")
        return seta

    def _novo_menu(self) -> tk.Menu:
        return tk.Menu(
            self._raiz,
            tearoff=0,
            bg="#0D1525",
            fg="#EAF2FF",
            activebackground="#00BFD9",
            activeforeground="#07111E",
            font=("Arial", 10),
            relief="flat",
            bd=1,
        )

    @staticmethod
    def _mostrar_menu(menu: tk.Menu, botao: tk.Widget) -> None:
        """Abre o menu logo abaixo do botão."""
        menu.post(
            botao.winfo_rootx(),
            botao.winfo_rooty() + botao.winfo_height(),
        )

    def _abrir_menu_tamanho(self, botao: tk.Widget) -> None:
        """Abre as opções de tamanho da janela (em Configurações)."""
        menu = self._novo_menu()

        largura_tela = self._raiz.winfo_screenwidth()
        altura_tela = self._raiz.winfo_screenheight()

        for largura, altura in RESOLUCOES_JANELA:
            # Resoluções maiores que o monitor não aparecem.
            if largura > largura_tela or altura > altura_tela:
                continue
            menu.add_command(
                label=_formatar_resolucao(largura, altura),
                command=lambda l=largura, a=altura: self._aplicar_tamanho_janela(l, a),
            )

        menu.add_separator()
        menu.add_command(
            label="Tela cheia",
            command=self._ativar_tela_cheia,
        )
        menu.add_command(
            label="Personalizado...",
            command=self._definir_tamanho_personalizado,
        )

        self._mostrar_menu(menu, botao)

    def _reaplicar_tamanho_janela(self) -> None:
        """Aplica novamente a opção atualmente selecionada."""
        if self._tamanho_janela is None:
            self._ativar_tela_cheia()
        else:
            self._aplicar_tamanho_janela(*self._tamanho_janela)

    def _ao_redimensionar_janela(self, evento: tk.Event) -> None:
        """Mostra no botão o tamanho real quando a janela é arrastada."""
        # A raiz repassa o <Configure> de todos os widgets; só a janela importa.
        if evento.widget is not self._raiz or self._em_tela_cheia:
            return

        tamanho = (evento.width, evento.height)
        if tamanho != self._tamanho_janela:
            self._tamanho_janela = tamanho
            self._texto_tamanho_janela.set(_formatar_resolucao(*tamanho))

    def _ativar_tela_cheia(self) -> None:
        self._em_tela_cheia = True
        self._tamanho_janela = None
        self._texto_tamanho_janela.set("TELA CHEIA")
        self._raiz.attributes("-fullscreen", True)

    def _aplicar_tamanho_janela(self, largura: int, altura: int) -> None:
        """Redimensiona a janela sem alterar a resolução do monitor."""
        largura_tela = self._raiz.winfo_screenwidth()
        altura_tela = self._raiz.winfo_screenheight()

        # Sai da tela cheia antes de aplicar outro tamanho.
        if self._em_tela_cheia:
            self._raiz.attributes("-fullscreen", False)
            self._em_tela_cheia = False

        # Nunca ultrapassa as dimensões do monitor.
        largura = min(largura, largura_tela)
        altura = min(altura, altura_tela)

        # Centraliza a janela no monitor.
        pos_x = max(0, (largura_tela - largura) // 2)
        pos_y = max(0, (altura_tela - altura) // 2)

        self._raiz.geometry(f"{largura}x{altura}+{pos_x}+{pos_y}")

        self._tamanho_janela = (largura, altura)
        self._texto_tamanho_janela.set(_formatar_resolucao(largura, altura))

    def _definir_tamanho_personalizado(self) -> None:
        """Permite informar a largura e a altura manualmente."""
        largura_tela = self._raiz.winfo_screenwidth()
        altura_tela = self._raiz.winfo_screenheight()
        largura_minima = min(LARGURA_MINIMA_JANELA, largura_tela)
        altura_minima = min(ALTURA_MINIMA_JANELA, altura_tela)

        largura_atual = self._raiz.winfo_width()
        altura_atual = self._raiz.winfo_height()

        largura = simpledialog.askinteger(
            "Tamanho personalizado",
            f"Largura da janela em pixels ({largura_minima} a {largura_tela}):",
            parent=self._raiz,
            initialvalue=max(largura_minima, min(largura_atual, largura_tela)),
            minvalue=largura_minima,
            maxvalue=largura_tela,
        )
        if largura is None:
            return

        altura = simpledialog.askinteger(
            "Tamanho personalizado",
            f"Altura da janela em pixels ({altura_minima} a {altura_tela}):",
            parent=self._raiz,
            initialvalue=max(altura_minima, min(altura_atual, altura_tela)),
            minvalue=altura_minima,
            maxvalue=altura_tela,
        )
        if altura is None:
            return

        self._aplicar_tamanho_janela(largura, altura)

    def _construir_video(self, mestre: tk.Misc) -> None:
        """Área ampla para a webcam e o rastreamento das mãos e do rosto."""
        caixa = tk.Frame(
            mestre,
            bg="#111B2D",
            highlightbackground="#243651",
            highlightthickness=1,
        )
        caixa.pack(side="left", fill="both", expand=True, padx=(0, 14))

        cabecalho = tk.Frame(caixa, bg="#111B2D")
        cabecalho.pack(fill="x", padx=16, pady=(14, 10))

        tk.Label(
            cabecalho,
            text="VISÃO AO VIVO",
            bg="#111B2D",
            fg="#F1F7FF",
            font=("Arial", 11, "bold"),
        ).pack(side="left")

        tk.Label(
            cabecalho,
            text="● CÂMERA",
            bg="#111B2D",
            fg="#5CE1A5",
            font=("Arial", 9, "bold"),
        ).pack(side="right")

        tk.Frame(caixa, bg="#243651", height=1).pack(fill="x")

        moldura = tk.Frame(caixa, bg="#050912")
        moldura.pack(fill="both", expand=True, padx=12, pady=12)

        self._area_video = tk.Frame(moldura, bg="#050912")
        self._area_video.pack(fill="both", expand=True, padx=2, pady=2)

        self._video = tk.Label(
            self._area_video,
            bg="#050912",
            fg="#8DA5C7",
            text="Preparando a câmera...",
            font=("Arial", 15),
        )
        self._video.place(relx=0.5, rely=0.5, anchor="center")

        legenda = tk.Frame(caixa, bg="#111B2D")
        legenda.pack(fill="x", padx=16, pady=(0, 12))

        tk.Label(
            legenda,
            text="● MÃO ESQUERDA",
            bg="#111B2D",
            fg="#00D7FF",
            font=("Arial", 8, "bold"),
        ).pack(side="left")

        tk.Label(
            legenda,
            text="● MÃO DIREITA",
            bg="#111B2D",
            fg="#FF9F43",
            font=("Arial", 8, "bold"),
        ).pack(side="left", padx=18)

        # Tempo de gravação do ensino, logo abaixo da câmera.
        linha_tempo = tk.Frame(caixa, bg="#111B2D")
        linha_tempo.pack(fill="x", padx=12, pady=(0, 10))

        tk.Label(
            linha_tempo,
            text="TEMPO DE GRAVAÇÃO (SEGUNDOS)",
            bg="#111B2D",
            fg="#7186A8",
            font=("Arial", 8, "bold"),
        ).pack(side="left", padx=(2, 12))

        tk.Entry(
            linha_tempo,
            textvariable=self._duracao,
            width=6,
            justify="center",
            font=("Arial", 11),
            bg="#080D18",
            fg="#F1F7FF",
            insertbackground="#00D7FF",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#30415C",
            highlightcolor="#00D7FF",
        ).pack(side="left", ipady=4)

        tk.Label(
            linha_tempo,
            text=(
                f"Enter: espera {config.SEGUNDOS_CONTAGEM_ENSINO} s e grava esse "
                "tempo. Vazio: grava até parar."
            ),
            bg="#111B2D",
            fg="#7186A8",
            font=("Arial", 8),
        ).pack(side="left", padx=(12, 0))

        # Split Button: tamanho da imagem da câmera, logo abaixo dela.
        linha_tamanho = tk.Frame(caixa, bg="#111B2D")
        linha_tamanho.pack(
            fill="x",
            padx=12,
            pady=(0, 12),
        )

        tk.Label(
            linha_tamanho,
            text="IMAGEM DA CÂMERA",
            bg="#111B2D",
            fg="#7186A8",
            font=("Arial", 8, "bold"),
        ).pack(side="left", padx=(2, 12))

        self._criar_split_button(
            linha_tamanho,
            self._texto_tamanho_camera,
            self._alternar_tamanho_camera,
            self._abrir_menu_tamanho_camera,
        )

    # ------------------------------------------------ tamanho da imagem da câmera

    def _tamanho_camera(self) -> tuple[int, int] | None:
        """Tamanho fixo escolhido para a imagem, ou None para ajustar à área."""
        return _ler_resolucao(self._config_visual["video_tamanho"])

    def _descrever_tamanho_camera(self) -> str:
        tamanho = self._tamanho_camera()
        if tamanho is None:
            return "AJUSTAR À ÁREA"
        return _formatar_resolucao(*tamanho)

    def _abrir_menu_tamanho_camera(self, botao: tk.Widget) -> None:
        """Abre as resoluções disponíveis para a imagem da câmera."""
        menu = self._novo_menu()
        menu.add_command(
            label="Ajustar à área",
            command=lambda: self._definir_tamanho_camera(None),
        )
        menu.add_separator()

        largura_area = self._area_video.winfo_width()
        altura_area = self._area_video.winfo_height()

        for largura, altura in RESOLUCOES_JANELA:
            # Tamanhos maiores que a área do vídeo não aparecem.
            if largura > largura_area or altura > altura_area:
                continue
            menu.add_command(
                label=_formatar_resolucao(largura, altura),
                command=lambda l=largura, a=altura: self._definir_tamanho_camera(
                    (l, a)
                ),
            )

        self._mostrar_menu(menu, botao)

    def _alternar_tamanho_camera(self) -> None:
        """Clique no botão principal: alterna entre ajustar e o último tamanho."""
        tamanho = self._tamanho_camera()
        if tamanho is not None:
            self._ultimo_tamanho_camera = tamanho
            self._definir_tamanho_camera(None)
        else:
            self._definir_tamanho_camera(self._ultimo_tamanho_camera)

    def _definir_tamanho_camera(self, tamanho: tuple[int, int] | None) -> None:
        if tamanho is not None:
            self._ultimo_tamanho_camera = tamanho
        self._config_visual["video_tamanho"] = (
            "ajustar" if tamanho is None else f"{tamanho[0]}x{tamanho[1]}"
        )
        self._texto_tamanho_camera.set(self._descrever_tamanho_camera())
        self._salvar_config_suavizacao()

    def _construir_painel(self, mestre: tk.Misc) -> None:
        """Painel lateral com sinal reconhecido e controles."""
        largura_tela = self._raiz.winfo_screenwidth()

        largura_painel = max(
            270,
            min(380, int(largura_tela * 0.22)),
        )

        painel = tk.Frame(
            mestre,
            bg="#0D1525",
            width=largura_painel,
            highlightbackground="#243651",
            highlightthickness=1,
        )
        painel.pack(side="right", fill="y")
        painel.pack_propagate(False)
        # Em telas mais baixas (ex.: 1024x768) os controles não cabem na altura:
        # o painel rola, em vez de esconder os botões de baixo.
        painel = area_rolavel(painel, "#0D1525")

        tk.Label(
            painel,
            text="PAINEL DE CONTROLE",
            bg="#0D1525",
            fg="#F1F7FF",
            font=("Arial", 12, "bold"),
        ).pack(anchor="w", padx=16, pady=(16, 12))

        # Resultado em destaque
        resultado = tk.Frame(
            painel,
            bg="#13243A",
            highlightbackground="#00AFCB",
            highlightthickness=1,
        )
        resultado.pack(fill="x", padx=12, pady=(0, 12))

        tk.Label(
            resultado,
            text="SINAL RECONHECIDO",
            bg="#13243A",
            fg="#7D93B5",
            font=("Arial", 8, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 5))

        tk.Label(
            resultado,
            textvariable=self._sinal_atual,
            bg="#13243A",
            fg="#00D7FF",
            font=("Arial", 19, "bold"),
            wraplength=255,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 14))

        # Modo de ensino
        ensinar = tk.Frame(painel, bg="#111B2D")
        ensinar.pack(fill="x", padx=12, pady=(0, 10))

        tk.Label(
            ensinar,
            text="01  /  MODO ENSINAR",
            bg="#111B2D",
            fg="#F1F7FF",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 8))

        tk.Label(
            ensinar,
            text="Nome do sinal",
            bg="#111B2D",
            fg="#8DA5C7",
            font=("Arial", 9),
        ).pack(anchor="w", padx=12)

        campo = tk.Entry(
            ensinar,
            textvariable=self._palavra,
            font=("Arial", 11),
            bg="#080D18",
            fg="#F1F7FF",
            insertbackground="#00D7FF",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#30415C",
            highlightcolor="#00D7FF",
        )
        campo.pack(fill="x", padx=12, pady=(5, 10), ipady=7)

        # Pré-sinal: forma de mão (ex.: "S") gravada antes do sinal.
        linha_pre = tk.Frame(ensinar, bg="#111B2D")
        linha_pre.pack(fill="x", padx=12, pady=(0, 10))
        campo_pre = tk.Entry(
            linha_pre,
            textvariable=self._pre_nome,
            width=8,
            justify="center",
            font=("Arial", 11),
            bg="#080D18",
            fg="#F1F7FF",
            disabledbackground="#0B1220",
            insertbackground="#00D7FF",
            relief="flat",
            highlightthickness=1,
            highlightbackground="#30415C",
            highlightcolor="#00D7FF",
            state="disabled",
        )

        def alternar_pre(*_args) -> None:
            ligado = self._pre_ligado.get()
            campo_pre.configure(state="normal" if ligado else "disabled")
            if ligado:
                campo_pre.focus_set()

        # trace: o campo acompanha o botão mesmo quando o valor muda pelo código.
        self._pre_ligado.trace_add("write", alternar_pre)

        tk.Checkbutton(
            linha_pre,
            text="Pré-sinal",
            variable=self._pre_ligado,
            bg="#111B2D",
            fg="#DCE8FA",
            selectcolor="#080D18",
            activebackground="#111B2D",
            activeforeground="#F1F7FF",
            font=("Arial", 9, "bold"),
            cursor="hand2",
        ).pack(side="left")
        campo_pre.pack(side="left", padx=(8, 0), ipady=4)

        self._botao_estilizado(
            ensinar,
            "COMEÇAR A ENSINAR (ENTER)",
            self._iniciar_ensino,
            principal=True,
        ).pack(fill="x", padx=12, pady=(0, 7))

        self._botao_estilizado(
            ensinar, "PARAR ENSINO / RECONHECIMENTO", self._parar
        ).pack(fill="x", padx=12, pady=(0, 10))

        self._botao_treinar = self._botao_estilizado(
            ensinar,
            "TREINAR IA",
            self._treinar,
            principal=True,
        )
        self._botao_treinar.pack(fill="x", padx=12, pady=(0, 12))

        # Reconhecimento
        sinal = tk.Frame(painel, bg="#111B2D")
        sinal.pack(fill="x", padx=12, pady=(0, 10))

        tk.Label(
            sinal,
            text="02  /  MODO SINAL",
            bg="#111B2D",
            fg="#F1F7FF",
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", padx=12, pady=(12, 8))

        self._botao_estilizado(
            sinal,
            "INICIAR RECONHECIMENTO",
            self._iniciar_reconhecimento,
            principal=True,
        ).pack(fill="x", padx=12, pady=(0, 7))

        self._botao_estilizado(
            painel,
            "SINAIS APRENDIDOS",
            self._abrir_sinais,
        ).pack(fill="x", padx=12, pady=(2, 7))

        self._botao_estilizado(
            painel,
            "ABRIR LOG",
            self._log.mostrar,
        ).pack(fill="x", padx=12, pady=(0, 7))

        self._botao_estilizado(
            painel,
            "CONFIGURAÇÕES VISUAIS",
            self._abrir_configuracoes,
        ).pack(fill="x", padx=12, pady=(0, 12))

        # Status
        status = tk.Frame(
            painel,
            bg="#080D18",
            highlightbackground="#243651",
            highlightthickness=1,
        )
        status.pack(fill="x", padx=12, pady=(0, 12))

        tk.Label(
            status,
            text="STATUS DO SISTEMA",
            bg="#080D18",
            fg="#00D7FF",
            font=("Arial", 9, "bold"),
        ).pack(anchor="w", padx=10, pady=(10, 5))

        tk.Label(
            status,
            textvariable=self._status,
            bg="#080D18",
            fg="#A7B8D2",
            justify="left",
            anchor="w",
            wraplength=260,
            font=("Arial", 9),
        ).pack(fill="x", padx=10, pady=(0, 10))

    def _botao_estilizado(
        self,
        mestre: tk.Misc,
        texto: str,
        comando,
        principal: bool = False,
    ) -> tk.Button:
        """Cria botões consistentes com o tema escuro do Kara Sygna."""
        fundo = "#00BFD9" if principal else "#1B2940"
        texto_cor = "#07111E" if principal else "#DCE8FA"
        hover = "#48E5F5" if principal else "#2A3C58"

        botao = tk.Button(
            mestre,
            text=texto,
            command=comando,
            bg=fundo,
            fg=texto_cor,
            activebackground=hover,
            activeforeground=texto_cor,
            disabledforeground="#7186A8",
            font=("Arial", 9, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            pady=9,
            wraplength=260,
        )

        botao.bind(
            "<Enter>",
            lambda _evento: botao.configure(
                bg=hover if botao["state"] != "disabled" else fundo
            ),
        )
        botao.bind(
            "<Leave>",
            lambda _evento: botao.configure(bg=fundo),
        )

        return botao

    def _construir_texto(self, mestre: tk.Misc) -> None:
        """Área de texto reconhecido, com leitura confortável."""
        caixa = tk.Frame(
            mestre,
            bg="#111B2D",
            highlightbackground="#243651",
            highlightthickness=1,
        )
        caixa.pack(fill="x", pady=(14, 0))

        cabecalho = tk.Frame(caixa, bg="#111B2D")
        cabecalho.pack(fill="x", padx=14, pady=(12, 8))

        tk.Label(
            cabecalho,
            text="TEXTO DOS SINAIS",
            bg="#111B2D",
            fg="#F1F7FF",
            font=("Arial", 11, "bold"),
        ).pack(side="left")

        tk.Label(
            cabecalho,
            text="O texto reconhecido aparece aqui",
            bg="#111B2D",
            fg="#7186A8",
            font=("Arial", 9),
        ).pack(side="left", padx=12)

        cor_texto = self._config_visual["texto_cor_sinais"]

        self._texto = scrolledtext.ScrolledText(
            caixa,
            height=4,
            font=tema.FONTE_TEXTO_SINAIS,
            bg="#080D18",
            fg=cor_texto,
            insertbackground=cor_texto,
            selectbackground="#164A66",
            selectforeground="#FFFFFF",
            wrap="word",
            relief="flat",
            padx=12,
            pady=10,
            borderwidth=0,
        )
        self._texto.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        botoes = tk.Frame(caixa, bg="#111B2D")
        botoes.pack(fill="x", padx=12, pady=(0, 12))

        self._botao_estilizado(
            botoes,
            "LIMPAR TEXTO",
            self._limpar_texto,
        ).pack(side="right")

    # --------------------------------------------------------------- ações

    def _tratar_erro(self, erro: BaseException, titulo: str) -> None:
        if isinstance(erro, ERROS_DE_USUARIO):
            logger.warning("%s: %s", titulo, erro)
            messagebox.showwarning(titulo, str(erro))
        else:
            logger.error("%s", titulo, exc_info=erro)
            messagebox.showerror(titulo, f"{erro}\n\nVeja os detalhes em ABRIR LOG.")
        self._status.set(f"{titulo}: {erro}")

    def _bloqueado_por_treino(self) -> bool:
        if self._executor.ocupado:
            messagebox.showinfo("Treinando", "Aguarde o treinamento terminar.")
        return self._executor.ocupado

    def _iniciar_ensino(self) -> None:
        """Monta as etapas (pré-sinal, se ligado, e sinal) e começa a primeira."""
        if self._bloqueado_por_treino():
            return
        palavra = self._palavra.get().strip()
        pre = self._pre_nome.get().strip() if self._pre_ligado.get() else ""
        try:
            if not palavra:
                raise PalavraInvalidaError(
                    "Escreva a palavra do sinal antes de começar."
                )
            if self._pre_ligado.get() and not pre:
                raise PalavraInvalidaError(
                    "Escreva o pré-sinal (ex.: S) ou desligue o botão Pré-sinal."
                )
            duracao = ler_duracao(self._duracao.get())
        except ErroAplicacao as erro:
            self._tratar_erro(erro, "Não foi possível ensinar")
            return

        self._parar()  # para o reconhecimento ou um ensino em andamento
        # Pré-sinal uma vez e o sinal em cada ritmo (veja aplicacao/etapas.py).
        self._etapas = montar_etapas(palavra, pre, duracao)
        self._proxima_etapa()

    def _proxima_etapa(self) -> None:
        """Começa a contagem da próxima etapa do ensino."""
        self._etapa_atual = etapa = self._etapas.pop(0)
        self._temporizador.iniciar(etapa.duracao)
        tempo = "até parar" if etapa.duracao is None else f"por {etapa.duracao:g} s"
        self._status.set(
            f"Prepare “{etapa.legenda}”: a gravação começa em "
            f"{config.SEGUNDOS_CONTAGEM_ENSINO} s e vai {tempo}."
        )

    def _comecar_gravacao(self) -> None:
        """Fim da contagem: começa a gravar de verdade."""
        etapa = self._etapa_atual
        try:
            self._status.set(
                self._controlador.iniciar_ensino(
                    etapa.nome, etapa.tipo, etapa.pre_sinal
                )
            )
        except ErroAplicacao as erro:
            self._temporizador.cancelar()
            self._etapas = []
            self._tratar_erro(erro, "Não foi possível ensinar")

    def _fim_da_etapa(self) -> None:
        """O tempo da etapa acabou: passa para a próxima ou termina."""
        if not self._etapas:
            self._parar()
            return
        resumo = self._controlador.parar()  # grava a etapa que acabou
        self._proxima_etapa()
        self._status.set(f"{resumo}\n{self._status.get()}")

    def _alternar_ensino(self, _evento: tk.Event | None = None) -> str:
        """Tecla Enter: começa a ensinar ou, na contagem ou gravação, para."""
        if self._temporizador.ativo or self._controlador.modo is Modo.ENSINANDO:
            self._parar()
        else:
            self._iniciar_ensino()
        return "break"

    def _iniciar_reconhecimento(self) -> None:
        if self._bloqueado_por_treino():
            return
        try:
            # O campo de segundos é só da gravação; a ajuda do pré-sinal usa
            # config.SEGUNDOS_ESPERA_SINAL_PADRAO.
            self._status.set(self._controlador.iniciar_reconhecimento())
        except ErroAplicacao as erro:
            self._tratar_erro(erro, "Não foi possível reconhecer")

    def _parar(self) -> None:
        self._temporizador.cancelar()
        self._etapas = []
        estava_ensinando = self._controlador.modo is Modo.ENSINANDO
        self._status.set(self._controlador.parar())
        if estava_ensinando:
            self._atualizar_sinais()

    def _treinar(self) -> None:
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
        self._botao_treinar.config(state="normal")
        self._atualizar_sinais()

        detalhe = (
            f"\n\nAcurácia na validação: {resultado.acuracia_validacao:.0%}"
            if resultado.acuracia_validacao is not None
            else f"\n\nAviso: {resultado.relatorio}"
        )

        self._status.set(
            f"IA treinada com {resultado.amostras} exemplos "
            f"e {len(resultado.classes)} sinais."
        )
        messagebox.showinfo(
            "IA treinada",
            "Pronto! Agora use INICIAR RECONHECIMENTO." + detalhe,
        )

    def _treino_falhou(self, erro: BaseException) -> None:
        self._botao_treinar.config(state="normal")
        self._tratar_erro(erro, "Não foi possível treinar")

    def _adicionar_texto(self, palavra: str) -> None:
        self._texto.insert(tk.END, palavra + " ")
        self._texto.see(tk.END)
        self._sinal_atual.set(palavra.upper())
        logger.info("Sinal adicionado ao texto: %s", palavra)

    def _limpar_texto(self) -> None:
        self._texto.delete("1.0", tk.END)
        self._controlador.limpar_reconhecimento()
        self._sinal_atual.set("AGUARDANDO SINAL")

    # ------------------------------------------------------------ janelas filhas

    def _atualizar_sinais(self) -> None:
        if self._janela_sinais is not None and self._janela_sinais.winfo_exists():
            self._janela_sinais.atualizar()

    def _abrir_sinais(self) -> None:
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
        try:
            reproducao = self._controlador.criar_reproducao(palavra)
        except ErroAplicacao as erro:
            self._tratar_erro(erro, "Não foi possível mostrar o sinal")
            return

        if self._janela_previa is not None and self._janela_previa.winfo_exists():
            self._janela_previa.destroy()

        self._janela_previa = JanelaPrevisualizacao(self._raiz, reproducao)

    def _apagar_sinal(self, palavra: str) -> None:
        try:
            self._status.set(self._controlador.apagar_sinal(palavra))
        except ErroAplicacao as erro:
            self._tratar_erro(erro, "Não foi possível apagar")
        self._atualizar_sinais()

    # ------------------------------------------------------------ loop da câmera

    def iniciar(self) -> None:
        """Começa a captura e entra no loop do Tkinter."""
        self.iniciar_captura()
        self._raiz.mainloop()

    def iniciar_captura(self) -> None:
        """Começa a captura quando o loop do Tkinter já está rodando."""
        self._atualizar_quadro()

    def _atualizar_quadro(self) -> None:
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
            if mensagem != self._ultimo_erro_quadro:
                logger.error("Erro no laço da câmera", exc_info=erro)
                self._ultimo_erro_quadro = mensagem
            self._status.set(f"Erro na câmera: {erro}")

        self._agendar(config.INTERVALO_QUADRO_MS)

    def _agendar(self, ms: int) -> None:
        try:
            self._raiz.after(ms, self._atualizar_quadro)
        except tk.TclError:
            pass

    def _processar_quadro(self, quadro) -> None:
        """Suaviza cópias para desenhar; os dados originais vão para a IA."""
        quadro_original = quadro.copy()

        # O corpo roda em paralelo (outra thread), na imagem original: o realce
        # escurece os braços e a roupa. Em paralelo, quase não custa tempo.
        corpo_futuro = (
            self._executor_corpo.submit(self._detector_corpo.detectar, quadro_original)
            if self._detector_corpo is not None
            else None
        )
        # O rosto vem primeiro: a cor da pele dele é usada no realce das mãos.
        resultado_rosto = self._detector_rosto.detectar(quadro_original)
        self._realce_pele.calibrar(quadro_original, resultado_rosto)
        # O detector de mãos vê a pele clara e o resto escuro (as mãos se
        # destacam da roupa); mãos que "piscaram" continuam valendo um instante.
        maos_detectadas = self._detector.detectar(
            self._realce_pele.aplicar(quadro_original)
        )
        resultado_corpo = corpo_futuro.result() if corpo_futuro is not None else None
        # Só valem mãos na ponta de um braço; o lado de cada uma é o do braço.
        altura_q, largura_q = quadro_original.shape[:2]
        maos_detectadas = conferir_maos(
            maos_detectadas, resultado_corpo, largura_q, altura_q
        )
        resultado_maos = self._memoria_maos.aplicar(maos_detectadas)

        # Os botões de exibição só escondem o desenho.
        # A detecção e a IA continuam usando os resultados originais.
        visual = self._config_visual

        if visual["mostrar_rosto"]:
            if resultado_rosto.presente and resultado_rosto.landmarks is not None:
                rosto_visual = copy.deepcopy(resultado_rosto.landmarks)
                coordenadas = np.asarray(
                    [(p.x, p.y, p.z) for p in rosto_visual.landmark],
                    dtype=np.float32,
                )
                suavizadas = self._suavizador_rosto.suavizar("rosto", coordenadas)

                for ponto, coordenada in zip(rosto_visual.landmark, suavizadas):
                    ponto.x, ponto.y, ponto.z = map(float, coordenada)

                resultado_rosto_visual = ResultadoRosto(
                    landmarks=rosto_visual,
                    vetor=resultado_rosto.vetor,
                    presente=True,
                )
            else:
                self._suavizador_rosto.limpar("rosto")
                resultado_rosto_visual = resultado_rosto

            quadro = self._detector_rosto.desenhar(
                quadro,
                resultado_rosto_visual,
                desenhar_malha=False,
                desenhar_pontos=visual["mostrar_pontos_rosto"],
                desenhar_rotulos=visual["mostrar_nomes_rosto"],
                desenhar_linhas=visual["mostrar_linhas_rosto"],
                estilo=visual,
            )
        else:
            # Rosto escondido: não suaviza e esquece o histórico.
            self._suavizador_rosto.limpar("rosto")

        if visual["mostrar_maos"]:
            if tem_maos(resultado_maos):
                resultado_maos_visual = copy.deepcopy(resultado_maos)

                for indice, mao in enumerate(
                    resultado_maos_visual.multi_hand_landmarks
                ):
                    coordenadas = np.asarray(
                        [(p.x, p.y, p.z) for p in mao.landmark],
                        dtype=np.float32,
                    )
                    suavizadas = self._suavizador_maos.suavizar(
                        f"mao_{indice}", coordenadas
                    )

                    for ponto, coordenada in zip(mao.landmark, suavizadas):
                        ponto.x, ponto.y, ponto.z = map(float, coordenada)
            else:
                self._suavizador_maos.limpar()
                resultado_maos_visual = resultado_maos

            self._detector.desenhar(
                quadro,
                resultado_maos_visual,
                estilo=visual,
            )
        else:
            self._suavizador_maos.limpar()

        self._escrever_contador_maos(quadro, resultado_maos)

        # Contagem antes do ensino e parada automática no fim do tempo.
        evento = self._temporizador.atualizar()
        if evento is Evento.COMECAR:
            self._comecar_gravacao()
        elif evento is Evento.PARAR:
            self._fim_da_etapa()

        # A IA continua recebendo o resultado original, não o visual suavizado.
        altura, largura = quadro_original.shape[:2]
        # No ensino a IA recebe só as mãos vistas de verdade: os buracos são
        # preenchidos depois, entre antes e depois (veja preenchimento.py).
        # Ao vivo vale a memória (a mão congelada por um instante).
        ensinando = self._controlador.modo is Modo.ENSINANDO
        vetor = montar_vetor_quadro(
            maos_detectadas if ensinando else resultado_maos,
            resultado_rosto,
            largura,
            altura,
            resultado_corpo,
        )

        try:
            resposta = self._controlador.processar_quadro(vetor)
        except ErroAplicacao as erro:
            self._controlador.parar()
            self._tratar_erro(erro, "Erro no reconhecimento")
            return

        if resposta is not None:
            if resposta.status:
                self._status.set(resposta.status)

            if resposta.sinal_confirmado:
                self._adicionar_texto(resposta.sinal_confirmado)

            texto_video = self._texto_do_temporizador(resposta.texto_video)
            if texto_video:
                self._escrever_no_video(
                    quadro,
                    texto_video,
                    tema.COR_VIDEO_BGR[resposta.nivel],
                    self._config_visual.get("video_cor_texto", "#FFFFFF"),
                )

        self._mostrar(quadro)

    def _escrever_contador_maos(self, quadro, resultado_maos) -> None:
        """Mostra quantas mãos a IA está recebendo (e quantas vieram da memória)."""
        total = len(getattr(resultado_maos, "multi_hand_landmarks", None) or [])
        lembradas = self._memoria_maos.lembradas
        texto = f"MAOS: {total}" + (f" ({lembradas} lembrada)" if lembradas else "")
        cor = (92, 225, 165) if total == 2 else (0, 200, 255)
        altura = quadro.shape[0]
        cv2.rectangle(quadro, (10, altura - 38), (230, altura - 10), (15, 15, 18), -1)
        cv2.putText(
            quadro,
            texto,
            (18, altura - 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            cor,
            1,
            cv2.LINE_AA,
        )

    def _texto_do_temporizador(self, texto_video: str) -> str:
        """Junta a contagem ou o tempo que falta ao texto sobre a câmera."""
        restantes = self._temporizador.segundos_restantes()
        etapa = self._etapa_atual
        if self._temporizador.fase is Fase.CONTAGEM and etapa:
            return f"PREPARE: {etapa.legenda_video}  {restantes}"
        if self._temporizador.fase is Fase.GRAVANDO and etapa:
            texto = f"{etapa.legenda_video} ({self._controlador.total_gravado})"
            return f"{texto}  FALTAM {restantes} s" if restantes is not None else texto
        if restantes is not None and texto_video:
            return f"{texto_video}  FALTAM {restantes} s"
        return texto_video

    @staticmethod
    def _escrever_no_video(
        quadro,
        texto: str,
        cor: tuple[int, int, int],
        cor_texto: str = "#FFFFFF",
    ) -> None:
        """Desenha o texto de reconhecimento sobre uma faixa escura."""
        cv2.rectangle(
            quadro,
            (10, 10),
            (min(quadro.shape[1] - 10, 700), 55),
            (15, 15, 18),
            -1,
        )

        try:
            cor_bgr = (
                int(cor_texto[5:7], 16),
                int(cor_texto[3:5], 16),
                int(cor_texto[1:3], 16),
            )
        except (ValueError, TypeError, IndexError):
            cor_bgr = cor

        cv2.putText(
            quadro,
            texto,
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            cor_bgr,
            2,
            cv2.LINE_AA,
        )

    def _mostrar(self, quadro) -> None:
        """Adapta a imagem da câmera ao espaço disponível, sem distorção."""
        try:
            largura = self._area_video.winfo_width()
            altura = self._area_video.winfo_height()

            if largura < TAMANHO_MINIMO_PX or altura < TAMANHO_MINIMO_PX:
                # Espera o layout terminar de ser calculado.
                return

            tamanho = self._tamanho_camera()
            if tamanho is not None and tamanho[0] <= largura and tamanho[1] <= altura:
                self._foto = quadro_para_foto_exata(quadro, *tamanho)
            else:
                # "Ajustar à área", ou o tamanho escolhido não cabe mais.
                self._foto = quadro_para_foto(quadro, largura, altura)
            self._video.configure(
                image=self._foto,
                text="",
            )

        except tk.TclError:
            # A janela pode estar sendo fechada durante a atualização.
            return

    # -------------------------------------------------------------------- fechar

    def fechar(self) -> None:
        if self._encerrado:
            return

        self._encerrado = True
        logger.info("Encerrando Kara Sygna.")

        self._fechar_configuracoes()

        try:
            self._detector.fechar()
        except Exception:
            logger.exception("Erro ao fechar o detector de mãos")

        try:
            self._detector_rosto.fechar()
        except Exception:
            logger.exception("Erro ao fechar o FaceMesh")

        self._executor_corpo.shutdown(wait=True)
        if self._detector_corpo is not None:
            try:
                self._detector_corpo.fechar()
            except Exception:
                logger.exception("Erro ao fechar o detector do corpo")

        try:
            self._camera.liberar()
        except Exception:
            logger.exception("Erro ao liberar a câmera")

        try:
            self._log.fechar()
        except Exception:
            logger.exception("Erro ao fechar a janela de log")

        try:
            self._raiz.destroy()
        except tk.TclError:
            pass
