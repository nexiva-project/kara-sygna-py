
"""Kara Sygna - Interface de ensino e reconhecimento de sinais."""

import csv
import os
import time
import tkinter as tk
import traceback
from tkinter import messagebox, scrolledtext

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier

from janela_log import iniciar_log, log, log_erro
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
FRAMES_ESTAVEIS = 6

LARGURA_CAMERA = 900
ALTURA_CAMERA = 510

MP_HANDS = mp.solutions.hands
MP_DESENHO = mp.solutions.drawing_utils

TOTAL_CARACTERISTICAS = FEATURES_POR_MAO * 2


# ============================================================
# AUXILIARES DA MÃO 3D
# ============================================================

class _PontoFalso:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class _MaoFalsa:
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
    """Cria o CSV com 126 características se ele não existir."""

    if os.path.exists(ARQUIVO_CSV):
        return

    cabecalho = ["rotulo"]

    for lado in ("esq", "dir"):
        for indice in range(21):
            cabecalho.extend([
                f"{lado}_x{indice}",
                f"{lado}_y{indice}",
                f"{lado}_z{indice}",
            ])

    with open(
        ARQUIVO_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as arquivo:
        csv.writer(arquivo).writerow(cabecalho)

    log(f"CSV criado: {os.path.abspath(ARQUIVO_CSV)}", "DATA")


# ============================================================
# APLICAÇÃO
# ============================================================

class KaraSygnaApp:

    def __init__(self):
        self.root = tk.Tk()
        self.janela_log = iniciar_log(self.root)

        log("Inicializando Kara Sygna.", "APP")
        log(f"Diretório de trabalho: {os.getcwd()}", "APP")
        log(f"Arquivo CSV: {os.path.abspath(ARQUIVO_CSV)}", "DATA")
        log(f"Arquivo do modelo: {os.path.abspath(ARQUIVO_MODELO)}", "MODEL")
        log(f"Características por mão: {FEATURES_POR_MAO}", "MODEL")
        log(f"Características esperadas no total: {TOTAL_CARACTERISTICAS}", "MODEL")

        self.root.title("Kara Sygna")
        self.root.geometry("1250x850")
        self.root.minsize(1050, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.fechar)

        self.palavra = tk.StringVar()
        self.status = tk.StringVar(value="Escolha um modo para começar.")

        self.modo = "parado"
        self.rotulo_atual = ""
        self.total_amostras = 0
        self.modelo = None

        self.sinal_candidato = None
        self.sinal_candidato_frames = 0
        self.ultimo_sinal_adicionado = None
        self.ultimo_sinal_tempo = 0

        self.camera_funcionando = True
        self.camera_photo = None

        self.angulo_x = -0.25
        self.angulo_y = 0.35

        self.janela_sinais = None
        self.lista_janela = None
        self.sinais_da_janela = []

        self._reproducao = None
        self._janela_reproducao = None

        self.captura = None
        self.hands = None

        self.criar_interface()

        try:
            self.hands = MP_HANDS.Hands(
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5,
            )
            log("MediaPipe inicializado.", "CAMERA")
        except Exception as erro:
            log_erro("Falha ao inicializar o MediaPipe.", erro)
            messagebox.showerror(
                "MediaPipe",
                f"Não foi possível inicializar o MediaPipe:\n{erro}",
            )
            self.root.after(0, self.fechar)
            return

        for indice_camera in range(5):
            log(f"Testando câmera {indice_camera}.", "CAMERA")

            try:
                captura = cv2.VideoCapture(
                    indice_camera,
                    cv2.CAP_DSHOW,
                )

                if not captura.isOpened():
                    captura.release()
                    log(f"Câmera {indice_camera} indisponível.", "CAMERA")
                    continue

                ret, frame_teste = captura.read()

                if ret and frame_teste is not None:
                    self.captura = captura
                    self.status.set(f"Câmera {indice_camera} conectada.")
                    log(f"Câmera {indice_camera} conectada.", "CAMERA")
                    break

                captura.release()
                log(
                    f"Câmera {indice_camera} abriu, mas não entregou imagem.",
                    "WARNING",
                )

            except Exception as erro:
                log_erro(f"Erro ao testar câmera {indice_camera}.", erro)

        if self.captura is None:
            log_erro("Nenhuma câmera disponível.")
            messagebox.showerror(
                "Câmera",
                "Não foi possível acessar nenhuma câmera.\n\n"
                "Verifique se a câmera está conectada e se outro "
                "programa não está usando ela.",
            )
            self.root.after(0, self.fechar)
            return

        self.atualizar_camera()

    # ========================================================
    # INTERFACE
    # ========================================================

    def criar_interface(self):
        fundo = tk.Frame(self.root, bg="#181a20")
        fundo.pack(fill="both", expand=True)

        tk.Label(
            fundo,
            text="KARA SYGNA",
            bg="#181a20",
            fg="#ffffff",
            font=("Arial", 20, "bold"),
        ).pack(anchor="w", padx=18, pady=(12, 5))

        superior = tk.Frame(fundo, bg="#181a20")
        superior.pack(fill="both", expand=True, padx=15, pady=(0, 8))

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
        self.area_camera.pack(fill="both", expand=True, padx=5, pady=5)

        painel = tk.Frame(superior, bg="#20232b", width=300)
        painel.pack(side="right", fill="y")
        painel.pack_propagate(False)

        ensinar = tk.LabelFrame(
            painel,
            text=" MODO ENSINAR ",
            bg="#20232b",
            fg="#ffffff",
            padx=10,
            pady=10,
        )
        ensinar.pack(fill="x", padx=10, pady=(10, 8))

        tk.Label(
            ensinar,
            text="Palavra do sinal:",
            bg="#20232b",
            fg="#dddddd",
        ).pack(anchor="w")

        self.campo_palavra = tk.Entry(
            ensinar,
            textvariable=self.palavra,
            font=("Arial", 11),
        )
        self.campo_palavra.pack(fill="x", pady=(5, 8))

        tk.Button(
            ensinar,
            text="COMEÇAR A ENSINAR",
            command=self.iniciar_ensino,
            height=2,
        ).pack(fill="x")

        tk.Button(
            ensinar,
            text="PARAR",
            command=self.parar,
        ).pack(fill="x", pady=(5, 0))

        tk.Button(
            ensinar,
            text="TREINAR IA",
            command=self.treinar,
            height=2,
        ).pack(fill="x", pady=(8, 0))

        sinal = tk.LabelFrame(
            painel,
            text=" MODO SINAL ",
            bg="#20232b",
            fg="#ffffff",
            padx=10,
            pady=10,
        )
        sinal.pack(fill="x", padx=10, pady=(0, 8))

        tk.Button(
            sinal,
            text="INICIAR RECONHECIMENTO",
            command=self.iniciar_reconhecimento,
            height=2,
        ).pack(fill="x")

        tk.Button(
            sinal,
            text="PARAR",
            command=self.parar,
        ).pack(fill="x", pady=(5, 0))

        tk.Button(
            painel,
            text="ABRIR SINAIS APRENDIDOS",
            command=self.abrir_janela_sinais,
            height=2,
        ).pack(fill="x", padx=10, pady=(2, 8))

        quadro_status = tk.LabelFrame(
            painel,
            text=" STATUS ",
            bg="#20232b",
            fg="#ffffff",
            padx=8,
            pady=8,
        )
        quadro_status.pack(fill="x", padx=10, pady=(0, 10))

        tk.Label(
            quadro_status,
            textvariable=self.status,
            bg="#20232b",
            fg="#dddddd",
            justify="left",
            anchor="w",
            wraplength=260,
        ).pack(fill="x")

        quadro_texto = tk.LabelFrame(
            fundo,
            text=" TEXTO DOS SINAIS ",
            bg="#181a20",
            fg="#ffffff",
            font=("Arial", 11, "bold"),
            padx=10,
            pady=10,
        )
        quadro_texto.pack(fill="x", padx=15, pady=(0, 15))

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
        self.area_texto.pack(fill="both", expand=True)

        botoes_texto = tk.Frame(quadro_texto, bg="#181a20")
        botoes_texto.pack(fill="x", pady=(8, 0))

        tk.Button(
            botoes_texto,
            text="LIMPAR TEXTO",
            command=self.limpar_texto,
        ).pack(side="right")

        tk.Label(
            fundo,
            text="Q = sair   |   A/D = girar mão 3D   |   W/S = inclinar mão 3D",
            bg="#181a20",
            fg="#777777",
            font=("Arial", 8),
        ).pack(pady=(0, 8))

    # ========================================================
    # TEXTO DOS SINAIS
    # ========================================================

    def adicionar_texto(self, palavra):
        palavra = str(palavra).strip()

        if not palavra:
            return

        self.area_texto.insert(tk.END, palavra + " ")
        self.area_texto.see(tk.END)
        log(f"Sinal adicionado ao texto: {palavra}", "RECOGNITION")

    def limpar_texto(self):
        self.area_texto.delete("1.0", tk.END)
        self.ultimo_sinal_adicionado = None
        self.ultimo_sinal_tempo = 0
        self.sinal_candidato = None
        self.sinal_candidato_frames = 0
        log("Texto e estado do reconhecimento limpos.", "APP")

    def processar_sinal_reconhecido(self, sinal, confianca):
        if sinal != self.sinal_candidato:
            self.sinal_candidato = sinal
            self.sinal_candidato_frames = 1
            return

        self.sinal_candidato_frames += 1

        if self.sinal_candidato_frames < FRAMES_ESTAVEIS:
            return

        # Não repete o mesmo sinal enquanto ele continuar na frente da câmera.
        if self.ultimo_sinal_adicionado == sinal:
            return

        self.adicionar_texto(sinal)
        self.ultimo_sinal_adicionado = sinal
        self.ultimo_sinal_tempo = time.monotonic()
        self.sinal_candidato_frames = FRAMES_ESTAVEIS

    # ========================================================
    # SINAIS ARMAZENADOS
    # ========================================================

    def obter_sinais(self):
        if not os.path.exists(ARQUIVO_CSV):
            log("CSV ainda não existe.", "DATA")
            return []

        try:
            dados = pd.read_csv(ARQUIVO_CSV)

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
                    modelo = joblib.load(ARQUIVO_MODELO)
                    sinais_no_modelo = set(modelo.classes_)
                except Exception as erro:
                    log_erro("Não foi possível consultar as classes do modelo.", erro)

            contagem = dados["rotulo"].value_counts().sort_index()

            return [
                (palavra, quantidade, palavra in sinais_no_modelo)
                for palavra, quantidade in contagem.items()
            ]

        except Exception as erro:
            log_erro("Falha ao ler a lista de sinais.", erro)
            messagebox.showerror(
                "Sinais",
                f"Não foi possível ler os sinais:\n{erro}",
            )
            return []

    def abrir_janela_sinais(self):
        if (
            self.janela_sinais is not None
            and self.janela_sinais.winfo_exists()
        ):
            self.janela_sinais.deiconify()
            self.janela_sinais.lift()
            self.atualizar_janela_sinais()
            return

        self.janela_sinais = tk.Toplevel(self.root)
        self.janela_sinais.title("Sinais armazenados")
        self.janela_sinais.geometry("520x500")
        self.janela_sinais.protocol(
            "WM_DELETE_WINDOW",
            self.fechar_janela_sinais,
        )

        conteudo = tk.Frame(self.janela_sinais, padx=15, pady=15)
        conteudo.pack(fill="both", expand=True)

        tk.Label(
            conteudo,
            text="Sinais que a IA conhece",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w")

        tk.Label(
            conteudo,
            text="✓ treinado   |   • precisa treinar IA",
            fg="#666666",
        ).pack(anchor="w", pady=(3, 10))

        lista_area = tk.Frame(conteudo)
        lista_area.pack(fill="both", expand=True)

        self.lista_janela = tk.Listbox(
            lista_area,
            height=15,
            font=("Arial", 11),
        )
        self.lista_janela.pack(side="left", fill="both", expand=True)

        self.lista_janela.bind(
            "<Double-Button-1>",
            lambda evento: self.ver_sinal_selecionado(),
        )

        barra = tk.Scrollbar(
            lista_area,
            command=self.lista_janela.yview,
        )
        barra.pack(side="right", fill="y")
        self.lista_janela.config(yscrollcommand=barra.set)

        tk.Button(
            conteudo,
            text="ATUALIZAR LISTA",
            command=self.atualizar_janela_sinais,
        ).pack(fill="x", pady=(10, 0))

        tk.Button(
            conteudo,
            text="VER SINAL (SÓ AS MÃOS)",
            command=self.ver_sinal_selecionado,
        ).pack(fill="x", pady=(7, 0))

        tk.Button(
            conteudo,
            text="APAGAR SINAL SELECIONADO",
            command=self.apagar_sinal_selecionado,
            fg="#9b1c1c",
        ).pack(fill="x", pady=(7, 0))

        self.atualizar_janela_sinais()
        log("Janela de sinais aprendidos aberta.", "APP")

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

        self.lista_janela.delete(0, tk.END)
        self.sinais_da_janela = []

        sinais = self.obter_sinais()

        if not sinais:
            self.lista_janela.insert(
                tk.END,
                "Nenhum sinal ensinado ainda.",
            )
            return

        for palavra, quantidade, treinado in sinais:
            estado = "✓ treinado" if treinado else "• precisa treinar"

            self.lista_janela.insert(
                tk.END,
                f"{palavra} — {quantidade} exemplos — {estado}",
            )
            self.sinais_da_janela.append(palavra)

    # ========================================================
    # VISUALIZAR SINAL 3D
    # ========================================================

    def ver_sinal_selecionado(self):
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
            amostras = dados[dados["rotulo"] == palavra].drop(
                columns=["rotulo"]
            )
        except Exception as erro:
            log_erro("Erro ao carregar amostras para a visualização 3D.", erro)
            messagebox.showerror(
                "Não foi possível ler os dados",
                str(erro),
            )
            return

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

            if tem_esquerda and tem_direita:
                vetores_reproducao.append(atual)
                i += 1
                continue

            if (
                (tem_esquerda or tem_direita)
                and i + 1 < len(vetores_originais)
            ):
                proximo = vetores_originais[i + 1]
                esquerda_proximo = proximo[:FEATURES_POR_MAO]
                direita_proximo = proximo[FEATURES_POR_MAO:]

                tem_esquerda_proximo = not eh_vetor_zerado(esquerda_proximo)
                tem_direita_proximo = not eh_vetor_zerado(direita_proximo)

                if (
                    tem_direita
                    and not tem_esquerda
                    and tem_esquerda_proximo
                    and not tem_direita_proximo
                ):
                    vetores_reproducao.append(
                        esquerda_proximo + direita_atual
                    )
                    i += 2
                    continue

                if (
                    tem_esquerda
                    and not tem_direita
                    and tem_direita_proximo
                    and not tem_esquerda_proximo
                ):
                    vetores_reproducao.append(
                        esquerda_atual + direita_proximo
                    )
                    i += 2
                    continue

            vetores_reproducao.append(atual)
            i += 1

        self._fechar_reproducao()
        self._reproducao = {
            "nome": palavra,
            "vetores": vetores_reproducao,
            "indice": 0,
            "angulo": 0.0,
        }
        self._janela_reproducao = f"Kara Sygna - Pre-visualizacao: {palavra}"

        log(
            f"Reprodução 3D iniciada: {palavra} "
            f"({len(vetores_reproducao)} exemplos).",
            "APP",
        )
        self._passo_reproducao()

    def _passo_reproducao(self):
        estado = self._reproducao

        if estado is None:
            return

        tamanho = (600, 500)
        largura, altura = tamanho
        tela = painel_3d(tamanho)

        vetor = estado["vetores"][estado["indice"]]
        esquerda = vetor[:FEATURES_POR_MAO]
        direita = vetor[FEATURES_POR_MAO:]

        cores = ((70, 190, 255), (150, 105, 255))

        cv2.line(
            tela,
            (largura // 2, 55),
            (largura // 2, altura - 15),
            (58, 63, 78),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            tela, "ESQUERDA", (60, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            cores[0], 1, cv2.LINE_AA,
        )
        cv2.putText(
            tela, "DIREITA", (390, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            cores[1], 1, cv2.LINE_AA,
        )

        if not eh_vetor_zerado(esquerda):
            pontos = projetar_landmarks(
                _vetor_para_mao_falsa(esquerda),
                tamanho, 0.2, estado["angulo"], 0.27, 0.09,
            )
            desenhar_mao_3d(tela, pontos, cores[0])

        if not eh_vetor_zerado(direita):
            pontos = projetar_landmarks(
                _vetor_para_mao_falsa(direita),
                tamanho, 0.2, estado["angulo"], 0.73, 0.09,
            )
            desenhar_mao_3d(tela, pontos, cores[1])

        legenda = (
            f"{estado['nome']} — exemplo "
            f"{estado['indice'] + 1}/{len(estado['vetores'])}"
        )
        cv2.putText(
            tela, legenda, (18, altura - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (225, 225, 235), 1, cv2.LINE_AA,
        )

        try:
            cv2.imshow(self._janela_reproducao, tela)
            tecla = cv2.waitKey(1) & 0xFF
            fechada = (
                cv2.getWindowProperty(
                    self._janela_reproducao,
                    cv2.WND_PROP_VISIBLE,
                ) < 1
            )
        except cv2.error as erro:
            log_erro("Erro na janela de reprodução 3D.", erro)
            self._fechar_reproducao()
            return

        if tecla == ord("q") or fechada:
            self._fechar_reproducao()
            return

        estado["indice"] = (
            estado["indice"] + 1
        ) % len(estado["vetores"])
        estado["angulo"] += 0.009

        self.root.after(33, self._passo_reproducao)

    def _fechar_reproducao(self):
        if self._janela_reproducao is not None:
            try:
                cv2.destroyWindow(self._janela_reproducao)
            except cv2.error:
                pass

        self._reproducao = None
        self._janela_reproducao = None

    # ========================================================
    # APAGAR SINAL
    # ========================================================

    def apagar_sinal_selecionado(self):
        selecao = self.lista_janela.curselection()

        if not selecao or not self.sinais_da_janela:
            messagebox.showwarning(
                "Selecione um sinal",
                "Selecione na lista o sinal que deseja apagar.",
            )
            return

        palavra = self.sinais_da_janela[selecao[0]]

        confirmar = messagebox.askyesno(
            "Apagar sinal",
            f"Apagar todos os exemplos de “{palavra}”?\n\n"
            "Esta ação não pode ser desfeita.",
        )
        if not confirmar:
            return

        try:
            dados = pd.read_csv(ARQUIVO_CSV)
            restantes = dados[dados["rotulo"] != palavra]
            restantes.to_csv(
                ARQUIVO_CSV,
                index=False,
                encoding="utf-8",
            )

            self.modelo = None
            self.modo = "parado"

            # Um modelo antigo não deve ser usado depois de apagar dados.
            if os.path.exists(ARQUIVO_MODELO):
                os.remove(ARQUIVO_MODELO)

            self.status.set(
                f"Sinal “{palavra}” apagado. "
                "Treine a IA novamente."
            )
            log(f"Sinal apagado: {palavra}", "DATA")
            self.atualizar_janela_sinais()

        except Exception as erro:
            log_erro(f"Falha ao apagar o sinal {palavra}.", erro)
            messagebox.showerror(
                "Não foi possível apagar",
                str(erro),
            )

    # ========================================================
    # ENSINAR
    # ========================================================

    def iniciar_ensino(self):
        palavra = self.palavra.get().strip().lower()

        if not palavra:
            messagebox.showwarning(
                "Palavra necessária",
                "Escreva a palavra do sinal antes de começar.",
            )
            return

        try:
            garantir_cabecalho_csv()
        except Exception as erro:
            log_erro("Não foi possível preparar o CSV.", erro)
            messagebox.showerror("CSV", str(erro))
            return

        self.rotulo_atual = palavra
        self.total_amostras = 0
        self.modo = "ensinar"

        self.status.set(
            f"Ensinando “{palavra}”. Faça o sinal diante da câmera."
        )
        log(f"Ensino iniciado: {palavra}", "DATA")

    # ========================================================
    # PARAR
    # ========================================================

    def parar(self):
        modo_anterior = self.modo
        self.modo = "parado"

        if modo_anterior == "ensinar":
            self.status.set(
                f"Ensino de “{self.rotulo_atual}” parado: "
                f"{self.total_amostras} exemplos gravados."
            )
            log(
                f"Ensino parado. Linhas gravadas: {self.total_amostras}.",
                "DATA",
            )
            self.atualizar_janela_sinais()

        elif modo_anterior == "sinal":
            self.status.set("Reconhecimento parado.")
            log("Reconhecimento parado pelo usuário.", "MODEL")

        else:
            self.status.set("Programa parado.")
            log("Modo parado.", "APP")

    # ========================================================
    # TREINAR IA
    # ========================================================

    def treinar(self):
        if not os.path.exists(ARQUIVO_CSV):
            messagebox.showwarning(
                "Sem exemplos",
                "Ensine pelo menos um sinal antes de treinar a IA.",
            )
            return

        try:
            log("Iniciando treinamento.", "MODEL")
            dados = pd.read_csv(ARQUIVO_CSV)

            if "rotulo" not in dados.columns:
                raise ValueError(
                    "O CSV não possui a coluna 'rotulo'."
                )

            if dados.empty or dados["rotulo"].nunique() < 1:
                raise ValueError(
                    "Ensine pelo menos uma palavra antes de treinar."
                )

            X = dados.drop(columns=["rotulo"])
            y = dados["rotulo"]

            if X.shape[1] != TOTAL_CARACTERISTICAS:
                raise ValueError(
                    f"O CSV tem {X.shape[1]} características por exemplo, "
                    f"mas o programa espera {TOTAL_CARACTERISTICAS}. "
                    "Verifique o CSV e utils.py."
                )

            if X.isnull().any().any():
                raise ValueError(
                    "O CSV contém valores vazios. Corrija os dados antes "
                    "de treinar."
                )

            self.status.set("Treinando a IA... aguarde.")
            self.root.update_idletasks()

            modelo_novo = RandomForestClassifier(
                n_estimators=150,
                random_state=42,
            )
            modelo_novo.fit(X, y)

            joblib.dump(modelo_novo, ARQUIVO_MODELO)
            self.modelo = modelo_novo

            log(
                f"Treinamento concluído: {len(dados)} linhas, "
                f"{X.shape[1]} características, "
                f"{y.nunique()} classes.",
                "MODEL",
            )
            log(
                f"Classes: {list(modelo_novo.classes_)}",
                "MODEL",
            )

            self.atualizar_janela_sinais()
            self.status.set(
                f"IA treinada com {len(dados)} exemplos e "
                f"{y.nunique()} sinais."
            )

            messagebox.showinfo(
                "IA treinada",
                "Pronto!\n\nAgora use INICIAR RECONHECIMENTO.",
            )

        except Exception as erro:
            log_erro("Falha durante o treinamento da IA.", erro)
            self.status.set(f"Erro no treinamento: {erro}")
            messagebox.showerror(
                "Não foi possível treinar",
                str(erro),
            )

    # ========================================================
    # INICIAR RECONHECIMENTO
    # ========================================================

    def iniciar_reconhecimento(self):
        if not os.path.exists(ARQUIVO_CSV):
            messagebox.showwarning(
                "Sem dados",
                "Ensine pelo menos um sinal primeiro.",
            )
            return

        if not os.path.exists(ARQUIVO_MODELO):
            messagebox.showwarning(
                "Modelo não encontrado",
                "Ensine sinais e clique em TREINAR IA primeiro.",
            )
            log("Reconhecimento cancelado: modelo não encontrado.", "WARNING")
            return

        try:
            data_csv = os.path.getmtime(ARQUIVO_CSV)
            data_modelo = os.path.getmtime(ARQUIVO_MODELO)

            if data_modelo < data_csv:
                messagebox.showwarning(
                    "Modelo desatualizado",
                    "Os sinais mudaram.\n\n"
                    "Clique em TREINAR IA antes de reconhecer.",
                )
                log("Reconhecimento cancelado: modelo desatualizado.", "WARNING")
                return

            modelo = joblib.load(ARQUIVO_MODELO)

            esperado = getattr(modelo, "n_features_in_", None)
            if esperado is not None and esperado != TOTAL_CARACTERISTICAS:
                raise ValueError(
                    f"O modelo espera {esperado} características, "
                    f"mas o programa gera {TOTAL_CARACTERISTICAS}. "
                    "Treine novamente a IA com o CSV atual."
                )

            if not hasattr(modelo, "predict_proba"):
                raise ValueError(
                    "O modelo carregado não oferece predict_proba()."
                )

            self.modelo = modelo
            self.modo = "sinal"

            self.sinal_candidato = None
            self.sinal_candidato_frames = 0
            self.ultimo_sinal_adicionado = None
            self.ultimo_sinal_tempo = 0

            self.status.set("Reconhecendo sinais ao vivo...")

            log("Reconhecimento iniciado.", "MODEL")
            log(f"Modelo: {type(self.modelo).__name__}", "MODEL")
            log(
                f"Características esperadas: "
                f"{getattr(self.modelo, 'n_features_in_', 'desconhecidas')}",
                "MODEL",
            )
            log(f"Classes do modelo: {list(self.modelo.classes_)}", "MODEL")

        except Exception as erro:
            log_erro("Não foi possível iniciar o reconhecimento.", erro)
            self.status.set(f"Erro ao carregar modelo: {erro}")
            messagebox.showerror(
                "Modelo inválido",
                f"Não foi possível iniciar o reconhecimento:\n\n{erro}",
            )

    # ========================================================
    # SALVAR EXEMPLO
    # ========================================================

    def salvar_exemplo(self, vetor):
        if len(vetor) != TOTAL_CARACTERISTICAS:
            raise ValueError(
                f"O vetor tem {len(vetor)} valores; "
                f"esperava {TOTAL_CARACTERISTICAS}."
            )

        linhas = [[self.rotulo_atual] + list(vetor)]

        esquerda = vetor[:FEATURES_POR_MAO]
        direita = vetor[FEATURES_POR_MAO:]

        so_esquerda = (
            not eh_vetor_zerado(esquerda)
            and eh_vetor_zerado(direita)
        )
        so_direita = (
            not eh_vetor_zerado(direita)
            and eh_vetor_zerado(esquerda)
        )

        if so_esquerda:
            linhas.append(
                [self.rotulo_atual]
                + [0.0] * FEATURES_POR_MAO
                + espelhar_mao(esquerda)
            )
        elif so_direita:
            linhas.append(
                [self.rotulo_atual]
                + espelhar_mao(direita)
                + [0.0] * FEATURES_POR_MAO
            )

        with open(
            ARQUIVO_CSV,
            "a",
            newline="",
            encoding="utf-8",
        ) as arquivo:
            csv.writer(arquivo).writerows(linhas)

        self.total_amostras += len(linhas)

        # Evita encher o log com uma mensagem a cada frame.
        if self.total_amostras == len(linhas) or self.total_amostras % 100 == 0:
            log(
                f"Ensino de '{self.rotulo_atual}': "
                f"{self.total_amostras} linhas gravadas.",
                "DATA",
            )

    # ========================================================
    # DESENHAR MÃOS NA CÂMERA
    # ========================================================

    def desenhar_maos_no_frame(self, frame, resultado):
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
            lado = (
                "Direita"
                if info.classification[0].label == "Left"
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
            x = int(pulso.x * frame.shape[1])
            y = max(25, int(pulso.y * frame.shape[0]) - 20)

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

    def criar_painel_3d(self, resultado, tamanho):
        largura, altura = tamanho
        tela = painel_3d(tamanho)
        cores = ((70, 190, 255), (150, 105, 255))

        cv2.line(
            tela,
            (largura // 2, 55),
            (largura // 2, altura - 15),
            (58, 63, 78),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            tela, "ESQUERDA", (int(largura * 0.08), 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, cores[0], 1, cv2.LINE_AA,
        )
        cv2.putText(
            tela, "DIREITA", (int(largura * 0.62), 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, cores[1], 1, cv2.LINE_AA,
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
                    if info.classification[0].label == "Left"
                    else 1
                )
                maos[numero] = landmarks

            for numero in (1, 2):
                landmarks = maos.get(numero)
                if landmarks is None:
                    continue

                posicao = 0.27 if numero == 1 else 0.73

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

    def mostrar_camera_tk(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagem = Image.fromarray(frame)

        largura = self.area_camera.winfo_width()
        altura = self.area_camera.winfo_height()

        if largura < 100:
            largura = LARGURA_CAMERA
        if altura < 100:
            altura = ALTURA_CAMERA

        imagem.thumbnail(
            (largura - 10, altura - 10),
            Image.Resampling.LANCZOS,
        )

        foto = ImageTk.PhotoImage(imagem)
        self.camera_photo = foto
        self.area_camera.configure(image=foto, text="")

    # ========================================================
    # ROTAÇÃO PARA RECONHECIMENTO
    # ========================================================

    @staticmethod
    def _matriz_rotacao_3d(rx_graus, ry_graus, rz_graus):
        rx, ry, rz = np.radians([rx_graus, ry_graus, rz_graus])

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        matriz_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ], dtype=np.float32)

        matriz_y = np.array([
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ], dtype=np.float32)

        matriz_z = np.array([
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        return matriz_z @ matriz_y @ matriz_x

    @classmethod
    def _rotacionar_vetor_mao(cls, vetor, rx, ry, rz):
        if eh_vetor_zerado(vetor):
            return list(vetor)

        matriz = cls._matriz_rotacao_3d(rx, ry, rz)
        pontos = np.asarray(vetor, dtype=np.float32).reshape(-1, 3)
        return (pontos @ matriz.T).reshape(-1).tolist()

    @classmethod
    def _vetor_com_rotacao(cls, vetor, rx, ry, rz):
        esquerda = vetor[:FEATURES_POR_MAO]
        direita = vetor[FEATURES_POR_MAO:]

        return (
            cls._rotacionar_vetor_mao(esquerda, rx, ry, rz)
            + cls._rotacionar_vetor_mao(direita, rx, ry, rz)
        )

    def prever_com_correcao_rotacao(self, vetor):
        if self.modelo is None:
            raise RuntimeError("Modelo não carregado.")

        if len(vetor) != TOTAL_CARACTERISTICAS:
            raise ValueError(
                f"O vetor atual tem {len(vetor)} características; "
                f"o programa espera {TOTAL_CARACTERISTICAS}."
            )

        esperado = getattr(self.modelo, "n_features_in_", None)
        if esperado is not None and esperado != len(vetor):
            raise ValueError(
                f"O modelo espera {esperado} características, "
                f"mas o vetor atual tem {len(vetor)}. "
                "Treine novamente a IA."
            )

        rotacoes = [
            (0, 0, 0),
            (15, 0, 0), (-15, 0, 0),
            (30, 0, 0), (-30, 0, 0),
            (45, 0, 0), (-45, 0, 0),
            (0, 15, 0), (0, -15, 0),
            (0, 30, 0), (0, -30, 0),
            (0, 45, 0), (0, -45, 0),
            (0, 0, 15), (0, 0, -15),
            (0, 0, 30), (0, 0, -30),
            (0, 0, 45), (0, 0, -45),
            (20, 20, 0), (20, -20, 0),
            (-20, 20, 0), (-20, -20, 0),
            (20, 0, 20), (20, 0, -20),
            (-20, 0, 20), (-20, 0, -20),
            (0, 20, 20), (0, 20, -20),
            (0, -20, 20), (0, -20, -20),
        ]

        vetores = [
            self._vetor_com_rotacao(vetor, rx, ry, rz)
            for rx, ry, rz in rotacoes
        ]

        nomes_caracteristicas = getattr(
            self.modelo,
            "feature_names_in_",
            None,
        )

        if nomes_caracteristicas is not None:
            entradas = pd.DataFrame(
                vetores,
                columns=nomes_caracteristicas,
            )
            probabilidades = self.modelo.predict_proba(entradas)
        else:
            # Compatibilidade com modelos treinados sem nomes de colunas.
            probabilidades = self.modelo.predict_proba(
                np.asarray(vetores, dtype=np.float32)
            )

        melhores_indices = probabilidades.argmax(axis=1)
        melhores_confiancas = probabilidades[
            np.arange(len(probabilidades)),
            melhores_indices,
        ]

        indice_rotacao = int(melhores_confiancas.argmax())
        indice_classe = int(melhores_indices[indice_rotacao])

        sinal = self.modelo.classes_[indice_classe]
        confianca = float(melhores_confiancas[indice_rotacao])
        rotacao = rotacoes[indice_rotacao]

        return sinal, confianca, rotacao

    # ========================================================
    # LOOP DA CÂMERA
    # ========================================================

    def atualizar_camera(self):
        if not self.camera_funcionando:
            return

        try:
            if self.captura is None or not self.captura.isOpened():
                log_erro("A câmera deixou de estar disponível.")
                self.status.set("Câmera desconectada ou indisponível.")
                return

            ret, frame = self.captura.read()

            if not ret or frame is None:
                log("Não foi possível receber um frame da câmera.", "WARNING")
                self.status.set("Não foi possível ler a câmera.")
                self.root.after(100, self.atualizar_camera)
                return

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = self.hands.process(frame_rgb)

            self.desenhar_maos_no_frame(frame, resultado)

            texto_camera = ""
            cor_camera = (220, 220, 220)

            if resultado.multi_hand_landmarks:
                vetor = montar_vetor_duas_maos(resultado)

                if self.modo == "ensinar":
                    self.salvar_exemplo(vetor)

                    texto_camera = (
                        f"ENSINANDO: {self.rotulo_atual} "
                        f"({self.total_amostras})"
                    )
                    cor_camera = (0, 220, 255)

                    self.status.set(
                        f"Ensinando “{self.rotulo_atual}”: "
                        f"{self.total_amostras} exemplos."
                    )

                elif self.modo == "sinal" and self.modelo is not None:
                    try:
                        sinal, confianca, rotacao = (
                            self.prever_com_correcao_rotacao(vetor)
                        )

                        if confianca >= CONFIANCA_MINIMA:
                            rx, ry, rz = rotacao

                            texto_camera = (
                                f"SINAL: {sinal} ({confianca:.0%})"
                            )

                            if rx or ry or rz:
                                texto_camera += (
                                    f" [ajuste {rx:+d},{ry:+d},{rz:+d}°]"
                                )

                            cor_camera = (0, 255, 0)
                            self.status.set(
                                f"Sinal reconhecido: {sinal} "
                                f"({confianca:.0%})"
                            )
                            self.processar_sinal_reconhecido(
                                sinal,
                                confianca,
                            )
                        else:
                            texto_camera = f"INCERTO ({confianca:.0%})"
                            cor_camera = (0, 165, 255)

                    except Exception as erro:
                        texto_camera = "Erro no reconhecimento"
                        self.status.set(f"Erro: {erro}")
                        log_erro(
                            "Falha durante o reconhecimento do sinal.",
                            erro,
                        )

            else:
                self.sinal_candidato = None
                self.sinal_candidato_frames = 0

                if self.modo == "sinal":
                    texto_camera = "MOSTRE UM SINAL"
                    cor_camera = (200, 200, 200)

            if texto_camera:
                cv2.rectangle(
                    frame,
                    (10, 10),
                    (min(frame.shape[1] - 10, 700), 55),
                    (15, 15, 18),
                    -1,
                )
                cv2.putText(
                    frame,
                    texto_camera,
                    (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    cor_camera,
                    2,
                    cv2.LINE_AA,
                )

            self.mostrar_camera_tk(frame)

        except Exception as erro:
            # Registra erros fora da previsão também, incluindo MediaPipe,
            # conversão de imagem e atualização da interface.
            self.status.set(f"Erro na câmera: {erro}")
            log_erro("Erro inesperado no loop da câmera.", erro)

        finally:
            if self.camera_funcionando:
                try:
                    self.root.after(15, self.atualizar_camera)
                except tk.TclError:
                    pass

    # ========================================================
    # TECLAS
    # ========================================================

    def tratar_tecla(self, evento):
        tecla = evento.keysym.lower()

        if tecla == "q":
            self.fechar()

    # ========================================================
    # FECHAR
    # ========================================================

    def fechar(self):
        if not self.camera_funcionando:
            return

        self.camera_funcionando = False
        log("Encerrando Kara Sygna.", "APP")

        try:
            self._fechar_reproducao()
        except Exception as erro:
            log_erro("Erro ao fechar reprodução 3D.", erro)

        try:
            if self.hands is not None:
                self.hands.close()
                log("MediaPipe encerrado.", "CAMERA")
        except Exception as erro:
            log_erro("Erro ao encerrar MediaPipe.", erro)

        try:
            if self.captura is not None:
                self.captura.release()
                log("Câmera liberada.", "CAMERA")
        except Exception as erro:
            log_erro("Erro ao liberar câmera.", erro)

        try:
            cv2.destroyAllWindows()
        except Exception as erro:
            log_erro("Erro ao fechar janelas do OpenCV.", erro)

        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ========================================================
    # EXECUTAR
    # ========================================================

    def executar(self):
        self.root.bind("<Key>", self.tratar_tecla)
        log("Interface iniciada; aguardando eventos.", "APP")
        self.root.mainloop()


# ============================================================
# MAIN
# ============================================================

def main():
    try:
        app = KaraSygnaApp()
        app.executar()
    except Exception as erro:
        log_erro("Erro fatal ao iniciar Kara Sygna.", erro)
        raise


if __name__ == "__main__":
    main()
