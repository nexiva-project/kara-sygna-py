"""Kara Sygna - Interface de ensino e reconhecimento de sinais."""

import csv
import os
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier

from mao_3d import desenhar_mao_3d, painel_3d, projetar_landmarks
from utils import (
    FEATURES_POR_MAO,
    eh_vetor_zerado,
    espelhar_mao,
    montar_vetor_duas_maos,
)


ARQUIVO_CSV = "dados_sinais.csv"
ARQUIVO_MODELO = "modelo_sinais.pkl"

CONFIANCA_MINIMA = 0.70

# Para evitar que o mesmo sinal seja colocado no texto
# centenas de vezes por segundo.
FRAMES_ESTAVEIS = 6
TEMPO_MINIMO_ENTRE_SINAIS = 0.8

# Tamanho aproximado da câmera dentro da interface.
LARGURA_CAMERA = 900
ALTURA_CAMERA = 510

MP_HANDS = mp.solutions.hands
MP_DESENHO = mp.solutions.drawing_utils


# ============================================================
# AUXILIARES DA MÃO 3D
# ============================================================

class _PontoFalso:
    """Imita um landmark do MediaPipe."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class _MaoFalsa:
    """Objeto compatível com projetar_landmarks()."""

    def __init__(self, pontos):
        self.landmark = pontos


def _vetor_para_mao_falsa(vetor_63):
    pontos = [
        _PontoFalso(
            vetor_63[i],
            vetor_63[i + 1],
            vetor_63[i + 2],
        )
        for i in range(0, len(vetor_63), 3)
    ]

    return _MaoFalsa(pontos)


# ============================================================
# CSV
# ============================================================

def garantir_cabecalho_csv():
    """Cria o CSV caso ainda não exista."""

    if os.path.exists(ARQUIVO_CSV):
        return

    cabecalho = ["rotulo"]

    for lado in ("esq", "dir"):
        for indice in range(21):
            cabecalho.extend(
                [
                    f"{lado}_x{indice}",
                    f"{lado}_y{indice}",
                    f"{lado}_z{indice}",
                ]
            )

    with open(
        ARQUIVO_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:
        csv.writer(arquivo).writerow(cabecalho)


# ============================================================
# APLICAÇÃO
# ============================================================

class KaraSygnaApp:

    def __init__(self):
        self.root = tk.Tk()

        self.root.title("Kara Sygna")
        self.root.geometry("1250x850")
        self.root.minsize(1050, 720)

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.fechar,
        )

        # ----------------------------------------------------
        # ESTADO
        # ----------------------------------------------------

        self.palavra = tk.StringVar()

        self.status = tk.StringVar(
            value="Escolha um modo para começar."
        )

        self.modo = "parado"

        self.rotulo_atual = ""

        self.total_amostras = 0

        self.modelo = None

        # ----------------------------------------------------
        # CONTROLE DO RECONHECIMENTO
        # ----------------------------------------------------

        self.sinal_candidato = None
        self.sinal_candidato_frames = 0

        self.ultimo_sinal_adicionado = None
        self.ultimo_sinal_tempo = 0

        # ----------------------------------------------------
        # CONTROLE DA CÂMERA
        # ----------------------------------------------------

        self.camera_funcionando = True
        self.camera_photo = None

        # ----------------------------------------------------
        # CONTROLE DA MÃO 3D
        # ----------------------------------------------------

        self.angulo_x = -0.25
        self.angulo_y = 0.35

        # ----------------------------------------------------
        # JANELA DE SINAIS
        # ----------------------------------------------------

        self.janela_sinais = None
        self.lista_janela = None
        self.sinais_da_janela = []

        # ----------------------------------------------------
        # REPRODUÇÃO 3D
        # ----------------------------------------------------

        self._reproducao = None
        self._janela_reproducao = None

        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        self.hands = MP_HANDS.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        # ----------------------------------------------------
        # INTERFACE
        # ----------------------------------------------------

        self.criar_interface()

        # ----------------------------------------------------
        # CÂMERA
        # ----------------------------------------------------

        self.captura = None

        # Tenta encontrar uma câmera disponível.
        for indice_camera in range(5):
            captura = cv2.VideoCapture(indice_camera, cv2.CAP_DSHOW)

            if captura.isOpened():
                # Tenta realmente receber uma imagem.
                ret, frame_teste = captura.read()

                if ret and frame_teste is not None:
                    self.captura = captura
                    self.status.set(f"Câmera {indice_camera} conectada.")
                    break

                captura.release()

        if self.captura is None:
            messagebox.showerror(
                "Câmera",
                "Não foi possível acessar nenhuma câmera.\n\n"
                "Verifique se a câmera está conectada e se outro programa "
                "não está usando ela."
            )
            self.root.after(0, self.fechar)
            return

        if not self.captura.isOpened():
            messagebox.showerror(
                "Câmera",
                "Não foi possível acessar a câmera.",
            )

            self.fechar()

            return

        self.atualizar_camera()

    # ========================================================
    # INTERFACE
    # ========================================================

    def criar_interface(self):

        # ----------------------------------------------------
        # FUNDO PRINCIPAL
        # ----------------------------------------------------

        fundo = tk.Frame(
            self.root,
            bg="#181a20",
        )

        fundo.pack(
            fill="both",
            expand=True,
        )

        # ----------------------------------------------------
        # TÍTULO
        # ----------------------------------------------------

        titulo = tk.Label(
            fundo,
            text="KARA SYGNA",
            bg="#181a20",
            fg="#ffffff",
            font=("Arial", 20, "bold"),
        )

        titulo.pack(
            anchor="w",
            padx=18,
            pady=(12, 5),
        )

        # ----------------------------------------------------
        # ÁREA SUPERIOR
        # ----------------------------------------------------

        superior = tk.Frame(
            fundo,
            bg="#181a20",
        )

        superior.pack(
            fill="both",
            expand=True,
            padx=15,
            pady=(0, 8),
        )

        # ----------------------------------------------------
        # CÂMERA
        # ----------------------------------------------------

        quadro_camera = tk.LabelFrame(
            superior,
            text=" CÂMERA ",
            bg="#181a20",
            fg="#ffffff",
            font=("Arial", 10, "bold"),
            bd=1,
            relief="solid",
        )

        quadro_camera.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 8),
        )

        self.area_camera = tk.Label(
            quadro_camera,
            bg="#101116",
            fg="#aaaaaa",
            text="Inicializando câmera...",
            font=("Arial", 14),
        )

        self.area_camera.pack(
            fill="both",
            expand=True,
            padx=5,
            pady=5,
        )

        # ----------------------------------------------------
        # PAINEL DIREITO
        # ----------------------------------------------------

        painel = tk.Frame(
            superior,
            bg="#20232b",
            width=300,
        )

        painel.pack(
            side="right",
            fill="y",
        )

        painel.pack_propagate(False)

        # ----------------------------------------------------
        # MODO ENSINAR
        # ----------------------------------------------------

        ensinar = tk.LabelFrame(
            painel,
            text=" MODO ENSINAR ",
            bg="#20232b",
            fg="#ffffff",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=10,
        )

        ensinar.pack(
            fill="x",
            padx=10,
            pady=(10, 8),
        )

        tk.Label(
            ensinar,
            text="Palavra do sinal:",
            bg="#20232b",
            fg="#dddddd",
        ).pack(
            anchor="w",
        )

        self.campo_palavra = tk.Entry(
            ensinar,
            textvariable=self.palavra,
            font=("Arial", 11),
        )

        self.campo_palavra.pack(
            fill="x",
            pady=(5, 8),
        )

        tk.Button(
            ensinar,
            text="COMEÇAR A ENSINAR",
            command=self.iniciar_ensino,
            height=2,
        ).pack(
            fill="x",
        )

        tk.Button(
            ensinar,
            text="PARAR",
            command=self.parar,
            height=1,
        ).pack(
            fill="x",
            pady=(5, 0),
        )

        tk.Button(
            ensinar,
            text="TREINAR IA",
            command=self.treinar,
            height=2,
        ).pack(
            fill="x",
            pady=(8, 0),
        )

        # ----------------------------------------------------
        # MODO SINAL
        # ----------------------------------------------------

        sinal = tk.LabelFrame(
            painel,
            text=" MODO SINAL ",
            bg="#20232b",
            fg="#ffffff",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=10,
        )

        sinal.pack(
            fill="x",
            padx=10,
            pady=(0, 8),
        )

        tk.Button(
            sinal,
            text="INICIAR RECONHECIMENTO",
            command=self.iniciar_reconhecimento,
            height=2,
        ).pack(
            fill="x",
        )

        tk.Button(
            sinal,
            text="PARAR",
            command=self.parar,
        ).pack(
            fill="x",
            pady=(5, 0),
        )

        # ----------------------------------------------------
        # SINAIS APRENDIDOS
        # ----------------------------------------------------

        tk.Button(
            painel,
            text="ABRIR SINAIS APRENDIDOS",
            command=self.abrir_janela_sinais,
            height=2,
        ).pack(
            fill="x",
            padx=10,
            pady=(2, 8),
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        quadro_status = tk.LabelFrame(
            painel,
            text=" STATUS ",
            bg="#20232b",
            fg="#ffffff",
            font=("Arial", 10, "bold"),
            padx=8,
            pady=8,
        )

        quadro_status.pack(
            fill="x",
            padx=10,
            pady=(0, 10),
        )

        tk.Label(
            quadro_status,
            textvariable=self.status,
            bg="#20232b",
            fg="#dddddd",
            justify="left",
            anchor="w",
            wraplength=260,
        ).pack(
            fill="x",
        )

        # ----------------------------------------------------
        # ÁREA INFERIOR - TEXTO
        # ----------------------------------------------------

        quadro_texto = tk.LabelFrame(
            fundo,
            text=" TEXTO DOS SINAIS ",
            bg="#181a20",
            fg="#ffffff",
            font=("Arial", 11, "bold"),
            padx=10,
            pady=10,
        )

        quadro_texto.pack(
            fill="x",
            padx=15,
            pady=(0, 15),
        )

        self.area_texto = scrolledtext.ScrolledText(
            quadro_texto,
            height=6,
            font=("Arial", 18),
            bg="#101116",
            fg="#ffffff",
            insertbackground="#ffffff",
            wrap="word",
            relief="flat",
        )

        self.area_texto.pack(
            fill="both",
            expand=True,
        )

        # ----------------------------------------------------
        # BOTÕES DO TEXTO
        # ----------------------------------------------------

        botoes_texto = tk.Frame(
            quadro_texto,
            bg="#181a20",
        )

        botoes_texto.pack(
            fill="x",
            pady=(8, 0),
        )

        tk.Button(
            botoes_texto,
            text="LIMPAR TEXTO",
            command=self.limpar_texto,
        ).pack(
            side="right",
        )

        # ----------------------------------------------------
        # ATALHOS
        # ----------------------------------------------------

        tk.Label(
            fundo,
            text="Q = sair   |   A/D = girar mão 3D   |   W/S = inclinar mão 3D",
            bg="#181a20",
            fg="#777777",
            font=("Arial", 8),
        ).pack(
            pady=(0, 8),
        )

    # ========================================================
    # TEXTO DOS SINAIS
    # ========================================================

    def adicionar_texto(self, palavra):

        palavra = str(palavra).strip()

        if not palavra:
            return

        self.area_texto.insert(
            tk.END,
            palavra + " ",
        )

        self.area_texto.see(
            tk.END,
        )

    def limpar_texto(self):

        self.area_texto.delete(
            "1.0",
            tk.END,
        )

        self.ultimo_sinal_adicionado = None

        self.ultimo_sinal_tempo = 0

        self.sinal_candidato = None

        self.sinal_candidato_frames = 0

    # ========================================================
    # RECONHECIMENTO ESTÁVEL
    # ========================================================

    def processar_sinal_reconhecido(
            self,
            sinal,
            confianca,
    ):
        # --------------------------------------------------------
        # Primeiro, verifica se o sinal mudou.
        # Se mudou, começa uma nova contagem de estabilidade.
        # --------------------------------------------------------

        if sinal != self.sinal_candidato:
            self.sinal_candidato = sinal
            self.sinal_candidato_frames = 1
            return

        # Mesmo sinal continua sendo detectado.
        self.sinal_candidato_frames += 1

        # Ainda não ficou estável.
        if self.sinal_candidato_frames < FRAMES_ESTAVEIS:
            return

        # --------------------------------------------------------
        # Se ESTE MESMO sinal já foi colocado no texto,
        # NÃO coloca novamente enquanto o usuário continuar
        # fazendo o mesmo gesto.
        # --------------------------------------------------------

        if self.ultimo_sinal_adicionado == sinal:
            return

        # --------------------------------------------------------
        # Sinal confirmado pela primeira vez.
        # Adiciona somente UMA vez.
        # --------------------------------------------------------

        self.adicionar_texto(sinal)

        self.ultimo_sinal_adicionado = sinal
        self.ultimo_sinal_tempo = time.monotonic()

        # Mantém o contador estável, mas bloqueia novas
        # inserções enquanto o mesmo sinal continuar presente.
        self.sinal_candidato_frames = FRAMES_ESTAVEIS
    # ========================================================
    # SINAIS ARMAZENADOS
    # ========================================================

    def obter_sinais(self):

        if not os.path.exists(ARQUIVO_CSV):
            return []

        try:

            dados = pd.read_csv(
                ARQUIVO_CSV,
            )

            if dados.empty:
                return []

            sinais_no_modelo = set()

            modelo_atual = (
                os.path.exists(ARQUIVO_MODELO)
                and os.path.getmtime(ARQUIVO_MODELO)
                >= os.path.getmtime(ARQUIVO_CSV)
            )

            if modelo_atual:

                try:

                    modelo = joblib.load(
                        ARQUIVO_MODELO,
                    )

                    sinais_no_modelo = set(
                        modelo.classes_,
                    )

                except Exception:
                    pass

            contagem = (
                dados["rotulo"]
                .value_counts()
                .sort_index()
            )

            return [
                (
                    palavra,
                    quantidade,
                    palavra in sinais_no_modelo,
                )
                for palavra, quantidade
                in contagem.items()
            ]

        except Exception as erro:

            messagebox.showerror(
                "Sinais",
                f"Não foi possível ler os sinais:\n{erro}",
            )

            return []

    # ========================================================
    # JANELA DE SINAIS
    # ========================================================

    def abrir_janela_sinais(self):

        if (
            self.janela_sinais is not None
            and self.janela_sinais.winfo_exists()
        ):

            self.janela_sinais.deiconify()
            self.janela_sinais.lift()

            self.atualizar_janela_sinais()

            return

        self.janela_sinais = tk.Toplevel(
            self.root,
        )

        self.janela_sinais.title(
            "Sinais armazenados",
        )

        self.janela_sinais.geometry(
            "520x500",
        )

        self.janela_sinais.protocol(
            "WM_DELETE_WINDOW",
            self.fechar_janela_sinais,
        )

        conteudo = tk.Frame(
            self.janela_sinais,
            padx=15,
            pady=15,
        )

        conteudo.pack(
            fill="both",
            expand=True,
        )

        tk.Label(
            conteudo,
            text="Sinais que a IA conhece",
            font=("Arial", 13, "bold"),
        ).pack(
            anchor="w",
        )

        tk.Label(
            conteudo,
            text="✓ treinado   |   • precisa treinar IA",
            fg="#666666",
        ).pack(
            anchor="w",
            pady=(3, 10),
        )

        lista_area = tk.Frame(
            conteudo,
        )

        lista_area.pack(
            fill="both",
            expand=True,
        )

        self.lista_janela = tk.Listbox(
            lista_area,
            height=15,
            font=("Arial", 11),
        )

        self.lista_janela.pack(
            side="left",
            fill="both",
            expand=True,
        )

        self.lista_janela.bind(
            "<Double-Button-1>",
            lambda evento:
            self.ver_sinal_selecionado(),
        )

        barra = tk.Scrollbar(
            lista_area,
            command=self.lista_janela.yview,
        )

        barra.pack(
            side="right",
            fill="y",
        )

        self.lista_janela.config(
            yscrollcommand=barra.set,
        )

        tk.Button(
            conteudo,
            text="ATUALIZAR LISTA",
            command=self.atualizar_janela_sinais,
        ).pack(
            fill="x",
            pady=(10, 0),
        )

        tk.Button(
            conteudo,
            text="VER SINAL (SÓ AS MÃOS)",
            command=self.ver_sinal_selecionado,
        ).pack(
            fill="x",
            pady=(7, 0),
        )

        tk.Button(
            conteudo,
            text="APAGAR SINAL SELECIONADO",
            command=self.apagar_sinal_selecionado,
            fg="#9b1c1c",
        ).pack(
            fill="x",
            pady=(7, 0),
        )

        self.atualizar_janela_sinais()

    def fechar_janela_sinais(self):

        if (
            self.janela_sinais is not None
            and self.janela_sinais.winfo_exists()
        ):
            self.janela_sinais.destroy()

        self.janela_sinais = None

    def atualizar_janela_sinais(self):

        if (
            self.janela_sinais is None
            or not self.janela_sinais.winfo_exists()
        ):
            return

        self.lista_janela.delete(
            0,
            tk.END,
        )

        self.sinais_da_janela = []

        sinais = self.obter_sinais()

        if not sinais:

            self.lista_janela.insert(
                tk.END,
                "Nenhum sinal ensinado ainda.",
            )

            return

        for palavra, quantidade, treinado in sinais:

            estado = (
                "✓ treinado"
                if treinado
                else "• precisa treinar"
            )

            self.lista_janela.insert(
                tk.END,
                f"{palavra} — "
                f"{quantidade} exemplos — "
                f"{estado}",
            )

            self.sinais_da_janela.append(
                palavra,
            )

    # ========================================================
    # VISUALIZAR SINAL 3D
    # ========================================================

    def ver_sinal_selecionado(self):
        """Reproduz os exemplos gravados do sinal selecionado.
        As duas linhas criadas para uma mão são combinadas em um único frame,
        para que esquerda e direita apareçam juntas na pré-visualização.
        """

        selecao = self.lista_janela.curselection()

        if not selecao or not self.sinais_da_janela:
            messagebox.showwarning(
                "Selecione um sinal",
                "Selecione na lista o sinal que deseja ver.",
            )
            return

        palavra = self.sinais_da_janela[selecao[0]]

        try:
            dados = pd.read_csv(ARQUIVO_CSV)
        except Exception as erro:
            messagebox.showerror(
                "Não foi possível ler os dados",
                str(erro),
            )
            return

        amostras = dados[
            dados["rotulo"] == palavra
            ].drop(columns=["rotulo"])

        if amostras.empty:
            messagebox.showinfo(
                "Sem exemplos",
                f"Não há exemplos gravados para “{palavra}”.",
            )
            return

        vetores_originais = amostras.values.tolist()
        vetores_reproducao = []

        i = 0

        while i < len(vetores_originais):

            atual = vetores_originais[i]

            esquerda_atual = atual[:FEATURES_POR_MAO]
            direita_atual = atual[FEATURES_POR_MAO:]

            tem_esquerda = not eh_vetor_zerado(esquerda_atual)
            tem_direita = not eh_vetor_zerado(direita_atual)

            # ============================================================
            # JÁ TEM AS DUAS MÃOS NO MESMO FRAME
            # ============================================================

            if tem_esquerda and tem_direita:
                vetores_reproducao.append(atual)
                i += 1
                continue

            # ============================================================
            # UMA MÃO
            #
            # salvar_exemplo() cria duas linhas:
            #
            #   linha 1 = mão original
            #   linha 2 = mão espelhada
            #
            # Aqui juntamos as duas em UM ÚNICO FRAME.
            # ============================================================

            if (tem_esquerda or tem_direita) and i + 1 < len(vetores_originais):

                proximo = vetores_originais[i + 1]

                esquerda_proximo = proximo[:FEATURES_POR_MAO]
                direita_proximo = proximo[FEATURES_POR_MAO:]

                tem_esquerda_proximo = not eh_vetor_zerado(
                    esquerda_proximo
                )

                tem_direita_proximo = not eh_vetor_zerado(
                    direita_proximo
                )

                # Atual = direita
                # Próximo = esquerda
                if (
                        tem_direita
                        and not tem_esquerda
                        and tem_esquerda_proximo
                        and not tem_direita_proximo
                ):
                    combinado = (
                            esquerda_proximo
                            + direita_atual
                    )

                    vetores_reproducao.append(combinado)

                    i += 2
                    continue

                # Atual = esquerda
                # Próximo = direita
                if (
                        tem_esquerda
                        and not tem_direita
                        and tem_direita_proximo
                        and not tem_esquerda_proximo
                ):
                    combinado = (
                            esquerda_atual
                            + direita_proximo
                    )

                    vetores_reproducao.append(combinado)

                    i += 2
                    continue

            # Se não encontrou um par correspondente,
            # mantém o frame original.
            vetores_reproducao.append(atual)
            i += 1

        # ============================================================
        # INICIA A PRÉ-VISUALIZAÇÃO
        # ============================================================

        self._fechar_reproducao()

        self._reproducao = {
            "nome": palavra,
            "vetores": vetores_reproducao,
            "indice": 0,
            "angulo": 0.0,
        }

        self._janela_reproducao = (
            f"Kara Sygna - Pre-visualizacao: {palavra}"
        )

        self._passo_reproducao()

    def _passo_reproducao(self):

        estado = self._reproducao

        if estado is None:
            return

        tamanho = (
            600,
            500,
        )

        largura, altura = tamanho

        tela = painel_3d(
            tamanho,
        )

        vetor = estado["vetores"][
            estado["indice"]
        ]

        esquerda = vetor[
            :FEATURES_POR_MAO
        ]

        direita = vetor[
            FEATURES_POR_MAO:
        ]

        cores = (
            (70, 190, 255),
            (150, 105, 255),
        )

        cv2.line(
            tela,
            (largura // 2, 55),
            (largura // 2, altura - 15),
            (58, 63, 78),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            tela,
            "ESQUERDA",
            (60, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            cores[0],
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            tela,
            "DIREITA",
            (390, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            cores[1],
            1,
            cv2.LINE_AA,
        )

        # MÃO ESQUERDA
        if not eh_vetor_zerado(esquerda):
            mao_esquerda = _vetor_para_mao_falsa(
                esquerda,
            )

            pontos_esquerda = projetar_landmarks(
                mao_esquerda,
                tamanho,
                0.2,
                estado["angulo"],
                0.27,
                0.09,
            )

            desenhar_mao_3d(
                tela,
                pontos_esquerda,
                cores[0],
            )

        # MÃO DIREITA
        if not eh_vetor_zerado(direita):
            mao_direita = _vetor_para_mao_falsa(
                direita,
            )

            pontos_direita = projetar_landmarks(
                mao_direita,
                tamanho,
                0.2,
                estado["angulo"],
                0.73,
                0.09,
            )

            desenhar_mao_3d(
                tela,
                pontos_direita,
                cores[1],
            )

        legenda = (
            f"{estado['nome']} — "
            f"exemplo "
            f"{estado['indice'] + 1}/"
            f"{len(estado['vetores'])}"
        )

        cv2.putText(
            tela,
            legenda,
            (18, altura - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (225, 225, 235),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(
            self._janela_reproducao,
            tela,
        )

        tecla = cv2.waitKey(1) & 0xFF

        try:
            fechada = (
                    cv2.getWindowProperty(
                        self._janela_reproducao,
                        cv2.WND_PROP_VISIBLE,
                    )
                    < 1
            )

        except cv2.error:
            fechada = True

        if (
                tecla == ord("q")
                or fechada
        ):
            self._fechar_reproducao()
            return

        # Próximo exemplo.
        # As duas mãos acima pertencem ao mesmo exemplo.
        estado["indice"] = (
                                   estado["indice"] + 1
                           ) % len(estado["vetores"])

        # Rotação proporcional a 30 FPS.
        estado["angulo"] += 0.009

        # 33 ms ≈ 30 FPS.
        self.root.after(
            33,
            self._passo_reproducao,
        )
    def _fechar_reproducao(self):

        if self._janela_reproducao is not None:

            try:

                cv2.destroyWindow(
                    self._janela_reproducao,
                )

            except cv2.error:
                pass

        self._reproducao = None

        self._janela_reproducao = None

    # ========================================================
    # APAGAR SINAL
    # ========================================================

    def apagar_sinal_selecionado(self):

        selecao = self.lista_janela.curselection()

        if (
            not selecao
            or not self.sinais_da_janela
        ):

            messagebox.showwarning(
                "Selecione um sinal",
                "Selecione na lista o sinal que deseja apagar.",
            )

            return

        palavra = self.sinais_da_janela[
            selecao[0]
        ]

        confirmar = messagebox.askyesno(
            "Apagar sinal",
            f"Apagar todos os exemplos de "
            f"“{palavra}”?\n\n"
            "Esta ação não pode ser desfeita.",
        )

        if not confirmar:
            return

        try:

            dados = pd.read_csv(
                ARQUIVO_CSV,
            )

            restantes = dados[
                dados["rotulo"] != palavra
            ]

            restantes.to_csv(
                ARQUIVO_CSV,
                index=False,
            )

            self.modelo = None

            self.modo = "parado"

            self.status.set(
                f"Sinal “{palavra}” apagado. "
                "Clique em TREINAR IA para atualizar o modelo."
            )

            self.atualizar_janela_sinais()

        except Exception as erro:

            messagebox.showerror(
                "Não foi possível apagar",
                str(erro),
            )

    # ========================================================
    # ENSINAR
    # ========================================================

    def iniciar_ensino(self):

        palavra = (
            self.palavra
            .get()
            .strip()
            .lower()
        )

        if not palavra:

            messagebox.showwarning(
                "Palavra necessária",
                "Escreva a palavra do sinal antes de começar.",
            )

            return

        garantir_cabecalho_csv()

        self.rotulo_atual = palavra

        self.total_amostras = 0

        self.modo = "ensinar"

        self.status.set(
            f"Ensinando “{palavra}”. "
            "Faça o sinal diante da câmera."
        )

    # ========================================================
    # PARAR
    # ========================================================

    def parar(self):

        if self.modo == "ensinar":

            self.status.set(
                f"Ensino de “{self.rotulo_atual}” parado: "
                f"{self.total_amostras} exemplos gravados."
            )

            self.atualizar_janela_sinais()

        elif self.modo == "sinal":

            self.status.set(
                "Reconhecimento parado."
            )

        self.modo = "parado"

    # ========================================================
    # TREINAR IA
    # ========================================================

    def treinar(self):

        if not os.path.exists(
            ARQUIVO_CSV
        ):

            messagebox.showwarning(
                "Sem exemplos",
                "Ensine pelo menos um sinal antes de treinar a IA.",
            )

            return

        try:

            dados = pd.read_csv(
                ARQUIVO_CSV,
            )

            if (
                dados.empty
                or dados["rotulo"].nunique() < 1
            ):

                raise ValueError(
                    "Ensine pelo menos uma palavra antes de treinar."
                )

            X = dados.drop(
                columns=["rotulo"],
            )

            y = dados["rotulo"]

            self.status.set(
                "Treinando a IA... aguarde."
            )

            self.root.update_idletasks()

            self.modelo = RandomForestClassifier(
                n_estimators=150,
                random_state=42,
            )

            self.modelo.fit(
                X,
                y,
            )

            joblib.dump(
                self.modelo,
                ARQUIVO_MODELO,
            )

            self.atualizar_janela_sinais()

            self.status.set(
                f"IA treinada com "
                f"{len(dados)} exemplos e "
                f"{y.nunique()} sinais."
            )

            messagebox.showinfo(
                "IA treinada",
                "Pronto!\n\n"
                "Agora use INICIAR RECONHECIMENTO.",
            )

        except Exception as erro:

            messagebox.showerror(
                "Não foi possível treinar",
                str(erro),
            )

    # ========================================================
    # INICIAR RECONHECIMENTO
    # ========================================================

    def iniciar_reconhecimento(self):

        if not os.path.exists(
            ARQUIVO_CSV
        ):

            messagebox.showwarning(
                "Sem dados",
                "Ensine pelo menos um sinal primeiro.",
            )

            return

        if (
            not os.path.exists(
                ARQUIVO_MODELO
            )
            or os.path.getmtime(
                ARQUIVO_MODELO
            )
            < os.path.getmtime(
                ARQUIVO_CSV
            )
        ):

            messagebox.showwarning(
                "Modelo desatualizado",
                "Os sinais mudaram.\n\n"
                "Clique em TREINAR IA antes de reconhecer.",
            )

            return

        try:

            self.modelo = joblib.load(
                ARQUIVO_MODELO,
            )

        except FileNotFoundError:

            messagebox.showwarning(
                "Modelo não encontrado",
                "Ensine sinais e clique em TREINAR IA primeiro.",
            )

            return

        except Exception as erro:

            messagebox.showerror(
                "Modelo inválido",
                str(erro),
            )

            return

        self.modo = "sinal"

        self.sinal_candidato = None

        self.sinal_candidato_frames = 0

        self.ultimo_sinal_adicionado = None

        self.ultimo_sinal_tempo = 0

        self.status.set(
            "Reconhecendo sinais ao vivo..."
        )

    # ========================================================
    # SALVAR EXEMPLO
    # ========================================================

    def salvar_exemplo(
        self,
        vetor,
    ):

        linhas = [
            [self.rotulo_atual] + vetor
        ]

        esquerda = vetor[
            :FEATURES_POR_MAO
        ]

        direita = vetor[
            FEATURES_POR_MAO:
        ]

        so_esquerda = (
            not eh_vetor_zerado(
                esquerda
            )
            and eh_vetor_zerado(
                direita
            )
        )

        so_direita = (
            not eh_vetor_zerado(
                direita
            )
            and eh_vetor_zerado(
                esquerda
            )
        )

        # Se foi feito com a mão esquerda,
        # também cria versão espelhada para direita.
        if so_esquerda:

            linhas.append(
                [
                    self.rotulo_atual
                ]
                + [0.0] * FEATURES_POR_MAO
                + espelhar_mao(
                    esquerda
                )
            )

        # Se foi feito com a mão direita,
        # também cria versão espelhada para esquerda.
        elif so_direita:

            linhas.append(
                [
                    self.rotulo_atual
                ]
                + espelhar_mao(
                    direita
                )
                + [0.0] * FEATURES_POR_MAO
            )

        with open(
            ARQUIVO_CSV,
            "a",
            newline="",
            encoding="utf-8",
        ) as arquivo:

            csv.writer(
                arquivo
            ).writerows(
                linhas
            )

        self.total_amostras += len(
            linhas
        )

    # ========================================================
    # DESENHAR MÃOS NA CÂMERA
    # ========================================================

    def desenhar_maos_no_frame(
        self,
        frame,
        resultado,
    ):

        if not (
            resultado.multi_hand_landmarks
            and resultado.multi_handedness
        ):
            return

        cores = {
            "Esquerda": (70, 190, 255),
            "Direita": (150, 105, 255),
        }

        for landmarks, info in zip(
            resultado.multi_hand_landmarks,
            resultado.multi_handedness,
        ):

            # Como a imagem foi espelhada,
            # o MediaPipe retorna os lados invertidos.
            lado = (
                "Direita"
                if info.classification[0].label
                == "Left"
                else "Esquerda"
            )

            cor = cores[lado]

            MP_DESENHO.draw_landmarks(
                frame,
                landmarks,
                MP_HANDS.HAND_CONNECTIONS,
                MP_DESENHO.DrawingSpec(
                    color=cor,
                    thickness=2,
                    circle_radius=3,
                ),
                MP_DESENHO.DrawingSpec(
                    color=cor,
                    thickness=2,
                ),
            )

            pulso = landmarks.landmark[0]

            x = int(
                pulso.x
                * frame.shape[1]
            )

            y = max(
                25,
                int(
                    pulso.y
                    * frame.shape[0]
                )
                - 20,
            )

            cv2.putText(
                frame,
                lado,
                (x - 25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                cor,
                2,
                cv2.LINE_AA,
            )

    # ========================================================
    # MÃO 3D
    # ========================================================

    def criar_painel_3d(
        self,
        resultado,
        tamanho,
    ):

        largura, altura = tamanho

        tela = painel_3d(
            tamanho,
        )

        cores = (
            (70, 190, 255),
            (150, 105, 255),
        )

        cv2.line(
            tela,
            (largura // 2, 55),
            (largura // 2, altura - 15),
            (58, 63, 78),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            tela,
            "ESQUERDA",
            (int(largura * 0.08), 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            cores[0],
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            tela,
            "DIREITA",
            (int(largura * 0.62), 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            cores[1],
            1,
            cv2.LINE_AA,
        )

        if (
            resultado.multi_hand_landmarks
            and resultado.multi_handedness
        ):

            maos = {}

            for landmarks, info in zip(
                resultado.multi_hand_landmarks,
                resultado.multi_handedness,
            ):

                numero = (
                    2
                    if info.classification[0].label
                    == "Left"
                    else 1
                )

                maos[numero] = landmarks

            for numero in (1, 2):

                landmarks = maos.get(
                    numero
                )

                if landmarks is None:
                    continue

                posicao = (
                    0.27
                    if numero == 1
                    else 0.73
                )

                pontos = projetar_landmarks(
                    landmarks,
                    tamanho,
                    self.angulo_x,
                    self.angulo_y,
                    posicao,
                    0.55,
                )

                desenhar_mao_3d(
                    tela,
                    pontos,
                    cores[numero - 1],
                )

        return tela

    # ========================================================
    # CONVERTER FRAME PARA TKINTER
    # ========================================================

    def mostrar_camera_tk(
        self,
        frame,
    ):

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        imagem = Image.fromarray(
            frame,
        )

        largura = self.area_camera.winfo_width()
        altura = self.area_camera.winfo_height()

        if largura < 100:
            largura = LARGURA_CAMERA

        if altura < 100:
            altura = ALTURA_CAMERA

        # Mantém proporção.
        imagem.thumbnail(
            (largura - 10, altura - 10),
            Image.Resampling.LANCZOS,
        )

        foto = ImageTk.PhotoImage(
            imagem,
        )

        self.camera_photo = foto

        self.area_camera.configure(
            image=foto,
            text="",
        )

    # ========================================================
    # LOOP DA CÂMERA
    # ========================================================

    def atualizar_camera(self):

        if not self.camera_funcionando:
            return

        if not self.captura.isOpened():
            return

        ret, frame = self.captura.read()

        if not ret:

            self.status.set(
                "Não foi possível ler a câmera."
            )

            self.root.after(
                50,
                self.atualizar_camera,
            )

            return

        # Espelho.
        frame = cv2.flip(
            frame,
            1,
        )

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        resultado = self.hands.process(
            frame_rgb,
        )

        # ----------------------------------------------------
        # DESENHAR LANDMARKS
        # ----------------------------------------------------

        self.desenhar_maos_no_frame(
            frame,
            resultado,
        )

        # ----------------------------------------------------
        # TEXTO NA CÂMERA
        # ----------------------------------------------------

        texto_camera = ""

        cor_camera = (
            220,
            220,
            220,
        )

        # ----------------------------------------------------
        # EXISTE MÃO?
        # ----------------------------------------------------

        if resultado.multi_hand_landmarks:

            vetor = montar_vetor_duas_maos(
                resultado,
            )

            # ------------------------------------------------
            # MODO ENSINAR
            # ------------------------------------------------

            if self.modo == "ensinar":

                self.salvar_exemplo(
                    vetor,
                )

                texto_camera = (
                    f"ENSINANDO: "
                    f"{self.rotulo_atual} "
                    f"({self.total_amostras})"
                )

                cor_camera = (
                    0,
                    220,
                    255,
                )

                self.status.set(
                    f"Ensinando "
                    f"“{self.rotulo_atual}”: "
                    f"{self.total_amostras} "
                    f"exemplos."
                )

            # ------------------------------------------------
            # MODO RECONHECIMENTO
            # ------------------------------------------------

            elif (
                self.modo == "sinal"
                and self.modelo is not None
            ):

                try:

                    entrada = pd.DataFrame(
                        [vetor],
                        columns=(
                            self.modelo
                            .feature_names_in_
                        ),
                    )

                    probabilidades = (
                        self.modelo
                        .predict_proba(
                            entrada
                        )[0]
                    )

                    indice = (
                        probabilidades.argmax()
                    )

                    sinal = (
                        self.modelo
                        .classes_[indice]
                    )

                    confianca = (
                        probabilidades[
                            indice
                        ]
                    )

                    if (
                        confianca
                        >= CONFIANCA_MINIMA
                    ):

                        texto_camera = (
                            f"SINAL: "
                            f"{sinal} "
                            f"({confianca:.0%})"
                        )

                        cor_camera = (
                            0,
                            255,
                            0,
                        )

                        self.status.set(
                            f"Sinal reconhecido: "
                            f"{sinal} "
                            f"({confianca:.0%})"
                        )

                        # Coloca no texto somente
                        # depois de ficar estável.
                        self.processar_sinal_reconhecido(
                            sinal,
                            confianca,
                        )

                    else:

                        texto_camera = (
                            f"INCERTO "
                            f"({confianca:.0%})"
                        )

                        cor_camera = (
                            0,
                            165,
                            255,
                        )

                except Exception as erro:

                    texto_camera = (
                        "Erro no reconhecimento"
                    )

                    self.status.set(
                        f"Erro: {erro}"
                    )

        else:

            # Quando não há mão, permite que
            # o próximo sinal seja reconhecido.
            self.sinal_candidato = None
            self.sinal_candidato_frames = 0

            if self.modo == "sinal":

                texto_camera = (
                    "MOSTRE UM SINAL"
                )

                cor_camera = (
                    200,
                    200,
                    200,
                )

        # ----------------------------------------------------
        # TEXTO SOBRE A CÂMERA
        # ----------------------------------------------------

        if texto_camera:

            cv2.rectangle(
                frame,
                (10, 10),
                (
                    min(
                        frame.shape[1] - 10,
                        500,
                    ),
                    55,
                ),
                (15, 15, 18),
                -1,
            )

            cv2.putText(
                frame,
                texto_camera,
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                cor_camera,
                2,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # MOSTRAR CÂMERA NO TKINTER
        # ----------------------------------------------------

        self.mostrar_camera_tk(
            frame,
        )

        # ----------------------------------------------------
        # TECLAS
        # ----------------------------------------------------

        # Tkinter precisa receber foco para os atalhos.
        # A/D/W/S podem ser usados também pelos botões
        # da janela 3D separada.
        self.root.after(
            15,
            self.atualizar_camera,
        )

    # ========================================================
    # TECLAS
    # ========================================================

    def tratar_tecla(
        self,
        evento,
    ):

        tecla = evento.keysym.lower()

        if tecla == "q":
            self.fechar()

    # ========================================================
    # FECHAR
    # ========================================================

    def fechar(self):

        self.camera_funcionando = False

        self._fechar_reproducao()

        try:

            if hasattr(
                self,
                "hands",
            ):
                self.hands.close()

        except Exception:
            pass

        try:

            if hasattr(
                self,
                "captura",
            ):
                self.captura.release()

        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            self.root.destroy()
        except Exception:
            pass

    # ========================================================
    # EXECUTAR
    # ========================================================

    def executar(self):

        self.root.bind(
            "<Key>",
            self.tratar_tecla,
        )

        self.root.mainloop()


# ============================================================
# MAIN
# ============================================================

def main():

    app = KaraSygnaApp()

    app.executar()


if __name__ == "__main__":
    main()