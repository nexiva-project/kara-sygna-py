"""Interface de ensino e reconhecimento de sinais com mão 3D."""

import csv
import os
import tkinter as tk
from tkinter import messagebox

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
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


def garantir_cabecalho_csv():
    """Cria o arquivo de exemplos e suas 126 colunas, se necessário."""
    if os.path.exists(ARQUIVO_CSV):
        return

    cabecalho = ["rotulo"]
    for lado in ("esq", "dir"):
        for indice in range(21):
            cabecalho.extend([f"{lado}_x{indice}", f"{lado}_y{indice}", f"{lado}_z{indice}"])
    with open(ARQUIVO_CSV, "w", newline="", encoding="utf-8") as arquivo:
        csv.writer(arquivo).writerow(cabecalho)


class KaraSygnaApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Kara Sygna — Ensino de Sinais")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.fechar)

        self.palavra = tk.StringVar()
        self.status = tk.StringVar(value="Escolha um modo para começar.")
        self.modo = "parado"
        self.rotulo_atual = ""
        self.total_amostras = 0
        self.modelo = None
        self.angulo_x, self.angulo_y = -0.25, 0.35

        self.criar_controles()
        self.atualizar_lista_sinais()
        self.captura = cv2.VideoCapture(0)
        if not self.captura.isOpened():
            messagebox.showerror("Câmera", "Não foi possível acessar a câmera.")
            self.root.after(0, self.fechar)
            return

        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.atualizar_camera()

    def criar_controles(self):
        conteudo = tk.Frame(self.root, padx=16, pady=14)
        conteudo.pack()

        ensinar = tk.LabelFrame(conteudo, text=" MODO ENSINAR ", padx=12, pady=10)
        ensinar.pack(fill="x", pady=(0, 10))
        tk.Label(ensinar, text="Palavra do sinal:").grid(row=0, column=0, sticky="w")
        tk.Entry(ensinar, textvariable=self.palavra, width=24).grid(
            row=0, column=1, padx=(8, 0), sticky="ew"
        )
        tk.Button(ensinar, text="COMEÇAR A ENSINAR", command=self.iniciar_ensino).grid(
            row=1, column=0, pady=(10, 0), sticky="ew"
        )
        tk.Button(ensinar, text="PARAR", command=self.parar).grid(
            row=1, column=1, padx=(8, 0), pady=(10, 0), sticky="ew"
        )
        tk.Button(ensinar, text="TREINAR IA", command=self.treinar).grid(
            row=2, column=0, columnspan=2, pady=(8, 0), sticky="ew"
        )

        sinal = tk.LabelFrame(conteudo, text=" MODO SINAL ", padx=12, pady=10)
        sinal.pack(fill="x")
        tk.Button(sinal, text="INICIAR RECONHECIMENTO", command=self.iniciar_reconhecimento).pack(
            fill="x"
        )
        tk.Button(sinal, text="PARAR", command=self.parar).pack(fill="x", pady=(8, 0))

        aprendidos = tk.LabelFrame(conteudo, text=" SINAIS QUE A IA APRENDEU ", padx=12, pady=10)
        aprendidos.pack(fill="x", pady=(10, 0))
        lista_area = tk.Frame(aprendidos)
        lista_area.pack(fill="x")
        self.lista_sinais = tk.Listbox(lista_area, height=6, width=38)
        self.lista_sinais.pack(side="left", fill="both", expand=True)
        barra = tk.Scrollbar(lista_area, command=self.lista_sinais.yview)
        barra.pack(side="right", fill="y")
        self.lista_sinais.config(yscrollcommand=barra.set)
        tk.Button(aprendidos, text="ATUALIZAR LISTA", command=self.atualizar_lista_sinais).pack(
            fill="x", pady=(8, 0)
        )

        tk.Label(conteudo, textvariable=self.status, justify="left", anchor="w", wraplength=300).pack(
            fill="x", pady=(12, 0)
        )
        tk.Label(
            conteudo,
            text="A janela da câmera mostra a mão 3D.\nA/D e W/S giram a visão; Q fecha.",
            fg="#555555",
            justify="left",
        ).pack(fill="x", pady=(8, 0))

    def atualizar_lista_sinais(self):
        """Mostra cada palavra gravada, sua quantidade de exemplos e treino."""
        self.lista_sinais.delete(0, tk.END)
        if not os.path.exists(ARQUIVO_CSV):
            self.lista_sinais.insert(tk.END, "Nenhum sinal ensinado ainda.")
            return

        try:
            dados = pd.read_csv(ARQUIVO_CSV)
            if dados.empty:
                self.lista_sinais.insert(tk.END, "Nenhum sinal ensinado ainda.")
                return
            sinais_no_modelo = set()
            if os.path.exists(ARQUIVO_MODELO):
                try:
                    sinais_no_modelo = set(joblib.load(ARQUIVO_MODELO).classes_)
                except Exception:
                    pass
            contagem = dados["rotulo"].value_counts().sort_index()
            for palavra, quantidade in contagem.items():
                estado = "✓ treinado" if palavra in sinais_no_modelo else "• falta treinar"
                self.lista_sinais.insert(
                    tk.END, f"{palavra} — {quantidade} exemplos — {estado}"
                )
        except Exception as erro:
            self.lista_sinais.insert(tk.END, f"Não foi possível ler os sinais: {erro}")

    def iniciar_ensino(self):
        palavra = self.palavra.get().strip().lower()
        if not palavra:
            messagebox.showwarning("Palavra necessária", "Escreva a palavra do sinal antes de começar.")
            return
        garantir_cabecalho_csv()
        self.rotulo_atual = palavra
        self.total_amostras = 0
        self.modo = "ensinar"
        self.status.set(f"Ensinando “{palavra}”. Faça o sinal e aperte PARAR ao terminar.")

    def parar(self):
        if self.modo == "ensinar":
            self.status.set(
                f"Ensino de “{self.rotulo_atual}” parado: {self.total_amostras} exemplos gravados."
            )
            self.atualizar_lista_sinais()
        elif self.modo == "sinal":
            self.status.set("Reconhecimento parado.")
        self.modo = "parado"

    def treinar(self):
        if not os.path.exists(ARQUIVO_CSV):
            messagebox.showwarning("Sem exemplos", "Ensine pelo menos um sinal antes de treinar a IA.")
            return
        try:
            dados = pd.read_csv(ARQUIVO_CSV)
            if dados.empty or dados["rotulo"].nunique() < 1:
                raise ValueError("Ensine pelo menos uma palavra antes de treinar.")
            X = dados.drop(columns=["rotulo"])
            y = dados["rotulo"]
            self.status.set("Treinando a IA... aguarde.")
            self.root.update_idletasks()
            self.modelo = RandomForestClassifier(n_estimators=150, random_state=42)
            self.modelo.fit(X, y)
            joblib.dump(self.modelo, ARQUIVO_MODELO)
            self.atualizar_lista_sinais()
            self.status.set(f"IA treinada com {len(dados)} exemplos e {y.nunique()} sinais.")
            messagebox.showinfo("IA treinada", "Pronto! Agora use INICIAR RECONHECIMENTO.")
        except Exception as erro:
            messagebox.showerror("Não foi possível treinar", str(erro))

    def iniciar_reconhecimento(self):
        try:
            self.modelo = joblib.load(ARQUIVO_MODELO)
        except FileNotFoundError:
            messagebox.showwarning("Modelo não encontrado", "Ensine sinais e clique em TREINAR IA primeiro.")
            return
        except Exception as erro:
            messagebox.showerror("Modelo inválido", str(erro))
            return
        self.modo = "sinal"
        self.status.set("Reconhecendo sinal ao vivo...")

    def salvar_exemplo(self, vetor):
        linhas = [[self.rotulo_atual] + vetor]
        esquerda = vetor[:FEATURES_POR_MAO]
        direita = vetor[FEATURES_POR_MAO:]
        so_esquerda = not eh_vetor_zerado(esquerda) and eh_vetor_zerado(direita)
        so_direita = not eh_vetor_zerado(direita) and eh_vetor_zerado(esquerda)
        if so_esquerda:
            linhas.append([self.rotulo_atual] + [0.0] * FEATURES_POR_MAO + espelhar_mao(esquerda))
        elif so_direita:
            linhas.append([self.rotulo_atual] + espelhar_mao(direita) + [0.0] * FEATURES_POR_MAO)
        with open(ARQUIVO_CSV, "a", newline="", encoding="utf-8") as arquivo:
            csv.writer(arquivo).writerows(linhas)
        self.total_amostras += len(linhas)

    def mostrar_maos_3d(self, frame, resultado):
        altura, largura = frame.shape[:2]
        tela = painel_3d((largura, altura))
        cores = ((70, 190, 255), (150, 105, 255))
        cv2.line(tela, (largura // 2, 55), (largura // 2, altura - 15), (58, 63, 78), 1, cv2.LINE_AA)
        cv2.putText(tela, "MAO 1 - ESQUERDA", (int(largura * 0.06), 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, cores[0], 1, cv2.LINE_AA)
        cv2.putText(tela, "MAO 2 - DIREITA", (int(largura * 0.55), 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, cores[1], 1, cv2.LINE_AA)

        if resultado.multi_hand_landmarks and resultado.multi_handedness:
            maos = {}
            for landmarks, info in zip(resultado.multi_hand_landmarks, resultado.multi_handedness):
                # O frame é espelhado: “Right” do MediaPipe representa a mão 1.
                numero = 2 if info.classification[0].label == "Left" else 1
                maos[numero] = landmarks
            for numero in (1, 2):
                landmarks = maos.get(numero)
                if landmarks is None:
                    continue
                pontos = projetar_landmarks(
                    landmarks, (largura, altura), self.angulo_x, self.angulo_y,
                    0.27 if numero == 1 else 0.73, 0.70,
                )
                desenhar_mao_3d(tela, pontos, cores[numero - 1])
        return np.hstack((frame, tela))

    def atualizar_camera(self):
        if not self.captura.isOpened():
            return
        ret, frame = self.captura.read()
        if not ret:
            self.status.set("Não foi possível ler a imagem da câmera.")
            self.root.after(30, self.atualizar_camera)
            return

        frame = cv2.flip(frame, 1)
        resultado = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        texto, cor = "Escolha um modo", (220, 220, 220)

        if resultado.multi_hand_landmarks:
            vetor = montar_vetor_duas_maos(resultado)
            if self.modo == "ensinar":
                self.salvar_exemplo(vetor)
                texto, cor = f"ENSINANDO: {self.rotulo_atual} ({self.total_amostras})", (0, 220, 255)
                self.status.set(f"Ensinando “{self.rotulo_atual}”: {self.total_amostras} exemplos gravados.")
            elif self.modo == "sinal" and self.modelo is not None:
                entrada = pd.DataFrame([vetor], columns=self.modelo.feature_names_in_)
                probabilidades = self.modelo.predict_proba(entrada)[0]
                indice = probabilidades.argmax()
                confianca = probabilidades[indice]
                if confianca >= CONFIANCA_MINIMA:
                    texto, cor = f"SINAL: {self.modelo.classes_[indice]} ({confianca:.0%})", (0, 255, 0)
                else:
                    texto, cor = f"INCERTO ({confianca:.0%})", (0, 165, 255)

        combinado = self.mostrar_maos_3d(frame, resultado)
        cv2.putText(combinado, texto, (18, combinado.shape[0] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, cor, 2, cv2.LINE_AA)
        cv2.imshow("Kara Sygna - Camera e Mao 3D", combinado)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            self.fechar()
            return
        if tecla == ord("a"):
            self.angulo_y -= 0.08
        elif tecla == ord("d"):
            self.angulo_y += 0.08
        elif tecla == ord("w"):
            self.angulo_x -= 0.08
        elif tecla == ord("s"):
            self.angulo_x += 0.08
        self.root.after(15, self.atualizar_camera)

    def fechar(self):
        if hasattr(self, "hands"):
            self.hands.close()
        if hasattr(self, "captura"):
            self.captura.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def executar(self):
        self.root.mainloop()


def main():
    KaraSygnaApp().executar()


if __name__ == "__main__":
    main()
