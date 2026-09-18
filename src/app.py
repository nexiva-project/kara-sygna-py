"""
kara-sygna-py — app.py: painel de controle único (interface gráfica).

Baseado na estrutura de painel sugerida por um colega (caixas "MODO
ENSINAR" / "MODO SINAL" + lista de sinais aprendidos), mas sem a
visualização 3D — aqui a câmera mostra o vídeo normal, com o
"esqueleto" da mão desenhado por cima (como nos passos anteriores do
projeto).

Como funciona:
    - Digite o nome do sinal e clique em "COMEÇAR A ENSINAR": a partir
      daí, todo frame com mão detectada é salvo automaticamente (com
      espelhamento automático para sinais de uma mão só). Clique em
      "PARAR" quando terminar de gravar aquele sinal.
    - Clique em "TREINAR IA" para retreinar o classificador com tudo
      que já foi ensinado até agora.
    - Clique em "INICIAR RECONHECIMENTO" para ver o sinal reconhecido
      ao vivo (com suavização: só mostra um sinal quando ele aparece
      com consistência nos últimos frames, para reduzir confusão
      entre sinais parecidos).
    - "PARAR" interrompe tanto o ensino quanto o reconhecimento.

A janela da câmera abre separada do painel (é assim que o OpenCV
funciona) — mas ambas são atualizadas pelo mesmo loop do tkinter, sem
precisar de threads.

Como rodar:
    python src/app.py
"""

import csv
import os
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import (
    FEATURES_POR_MAO,
    eh_vetor_zerado,
    espelhar_mao,
    montar_vetor_duas_maos,
)

ARQUIVO_CSV = "dados_sinais.csv"
ARQUIVO_MODELO = "modelo_sinais.pkl"
CONFIANCA_MINIMA = 0.70
TAMANHO_JANELA = 10   # frames considerados na suavização temporal
MINIMO_DE_VOTOS = 6   # quantos desses frames precisam concordar


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
        self.root.title("Kara SYGNA IA - Treinamento")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.fechar)

        self.palavra = tk.StringVar()
        self.status = tk.StringVar(value="Escolha um modo para começar.")
        self.modo = "parado"
        self.rotulo_atual = ""
        self.total_amostras = 0
        self.modelo = None
        self.historico_previsoes = deque(maxlen=TAMANHO_JANELA)

        self.criar_controles()
        self.atualizar_lista_sinais()

        self.captura = cv2.VideoCapture(0)
        if not self.captura.isOpened():
            messagebox.showerror("Câmera", "Não foi possível acessar a câmera.")
            self.root.after(0, self.fechar)
            return

        self.mp_hands = mp.solutions.hands
        self.mp_desenho = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.atualizar_camera()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def criar_controles(self):
        conteudo = tk.Frame(self.root, padx=16, pady=14)
        conteudo.pack()

        ensinar = tk.LabelFrame(conteudo, text=" MODO ENSINAR ", padx=12, pady=10)
        ensinar.pack(fill="x", pady=(0, 10))
        tk.Label(ensinar, text="Nome do sinal:").grid(row=0, column=0, sticky="w")
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
            text="A janela separada mostra a câmera com o esqueleto da mão.\nPressione Q nela para fechar.",
            fg="#555555",
            justify="left",
        ).pack(fill="x", pady=(8, 0))

    def atualizar_lista_sinais(self):
        """Mostra cada palavra gravada, sua quantidade de exemplos e se já foi treinada."""
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

    # ------------------------------------------------------------------
    # Ações dos botões
    # ------------------------------------------------------------------
    def iniciar_ensino(self):
        palavra = self.palavra.get().strip().lower()
        if not palavra:
            messagebox.showwarning("Palavra necessária", "Escreva o nome do sinal antes de começar.")
            return
        garantir_cabecalho_csv()
        self.rotulo_atual = palavra
        self.total_amostras = 0
        self.modo = "ensinar"
        self.status.set(f"Ensinando '{palavra}'. Faça o sinal e aperte PARAR ao terminar.")

    def parar(self):
        if self.modo == "ensinar":
            self.status.set(
                f"Ensino de '{self.rotulo_atual}' parado: {self.total_amostras} exemplos gravados."
            )
            self.atualizar_lista_sinais()
        elif self.modo == "sinal":
            self.status.set("Reconhecimento parado.")
        self.modo = "parado"
        self.historico_previsoes.clear()

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

            # Separamos uma parte para medir a acurácia, mas o modelo
            # final é treinado com TODOS os dados disponíveis.
            if y.nunique() > 1 and len(dados) >= 10:
                X_treino, X_teste, y_treino, y_teste = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                modelo_avaliacao = RandomForestClassifier(n_estimators=150, random_state=42)
                modelo_avaliacao.fit(X_treino, y_treino)
                acuracia = accuracy_score(y_teste, modelo_avaliacao.predict(X_teste))
            else:
                acuracia = None

            self.modelo = RandomForestClassifier(n_estimators=150, random_state=42)
            self.modelo.fit(X, y)
            joblib.dump(self.modelo, ARQUIVO_MODELO)

            self.atualizar_lista_sinais()
            texto_acuracia = f" (acurácia estimada: {acuracia:.0%})" if acuracia is not None else ""
            self.status.set(f"IA treinada com {len(dados)} exemplos e {y.nunique()} sinais{texto_acuracia}.")
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
        self.historico_previsoes.clear()
        self.modo = "sinal"
        self.status.set("Reconhecendo sinal ao vivo...")

    # ------------------------------------------------------------------
    # Gravação de exemplos (com espelhamento automático)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Loop principal: câmera + tkinter no mesmo laço (sem threads)
    # ------------------------------------------------------------------
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
        texto, cor = "Escolha um modo", (9, 255, 0)

        if resultado.multi_hand_landmarks:
            for landmarks_da_mao in resultado.multi_hand_landmarks:
                self.mp_desenho.draw_landmarks(
                    frame, landmarks_da_mao, self.mp_hands.HAND_CONNECTIONS
                )

            vetor = montar_vetor_duas_maos(resultado)

            if self.modo == "ensinar":
                self.salvar_exemplo(vetor)
                texto, cor = f"ENSINANDO: {self.rotulo_atual} ({self.total_amostras})", (0, 220, 255)
                self.status.set(f"Ensinando '{self.rotulo_atual}': {self.total_amostras} exemplos gravados.")

            elif self.modo == "sinal" and self.modelo is not None:
                entrada = pd.DataFrame([vetor], columns=self.modelo.feature_names_in_)
                probabilidades = self.modelo.predict_proba(entrada)[0]
                indice = probabilidades.argmax()
                sinal_previsto = self.modelo.classes_[indice]
                confianca = probabilidades[indice]

                # Suavização temporal: só "vota" quando a confiança é alta.
                self.historico_previsoes.append(
                    sinal_previsto if confianca >= CONFIANCA_MINIMA else None
                )
                votos = Counter(v for v in self.historico_previsoes if v is not None)
                if votos:
                    sinal_mais_votado, quantidade_votos = votos.most_common(1)[0]
                else:
                    sinal_mais_votado, quantidade_votos = None, 0

                if sinal_mais_votado is not None and quantidade_votos >= MINIMO_DE_VOTOS:
                    texto = f"SINAL: {sinal_mais_votado} ({quantidade_votos}/{TAMANHO_JANELA})"
                    cor = (0, 255, 0)
                else:
                    texto = f"INCERTO (frame atual: {sinal_previsto}, {confianca:.0%})"
                    cor = (0, 165, 255)
        else:
            self.historico_previsoes.clear()

        cv2.putText(frame, texto, (18, frame.shape[0] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, cor, 2, cv2.LINE_AA)
        cv2.imshow("Kara SYGNA IA - Câmera", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            self.fechar()
            return

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