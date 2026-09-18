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

MP_HANDS = mp.solutions.hands
MP_DESENHO = mp.solutions.drawing_utils

# Escala usada só na pré-visualização (Passo "VER SINAL"). Os vetores salvos
# no CSV já vêm normalizados pelo tamanho da mão (ver utils.py), então a
# grandeza dos números é bem diferente da dos landmarks brutos do MediaPipe
# — por isso precisam de uma escala própria, bem menor, pra caber no painel.
ESCALA_REPRODUCAO = 0.09
INTERVALO_REPRODUCAO_MS = 110


class _PontoFalso:
    """Imita um landmark do MediaPipe (só precisa de x, y, z)."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _MaoFalsa:
    """Imita o objeto de landmarks do MediaPipe, pra reaproveitar
    'projetar_landmarks' de mao_3d.py sem precisar duplicar a lógica."""

    def __init__(self, pontos):
        self.landmark = pontos


def _vetor_para_mao_falsa(vetor_63):
    pontos = [
        _PontoFalso(vetor_63[i], vetor_63[i + 1], vetor_63[i + 2])
        for i in range(0, len(vetor_63), 3)
    ]
    return _MaoFalsa(pontos)


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
        self.janela_sinais = None
        self.sinais_da_janela = []
        self._reproducao = None
        self._janela_reproducao = None

        self.criar_controles()
        self.captura = cv2.VideoCapture(0)
        if not self.captura.isOpened():
            messagebox.showerror("Câmera", "Não foi possível acessar a câmera.")
            self.root.after(0, self.fechar)
            return

        self.hands = MP_HANDS.Hands(
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

        tk.Button(
            conteudo,
            text="ABRIR SINAIS APRENDIDOS",
            command=self.abrir_janela_sinais,
        ).pack(fill="x", pady=(10, 0))

        tk.Label(conteudo, textvariable=self.status, justify="left", anchor="w", wraplength=300).pack(
            fill="x", pady=(12, 0)
        )
        tk.Label(
            conteudo,
            text="A janela da câmera mostra a mão 3D.\nA/D e W/S giram a visão; Q fecha.",
            fg="#555555",
            justify="left",
        ).pack(fill="x", pady=(8, 0))

    def obter_sinais(self):
        """Retorna os sinais gravados, a quantidade de exemplos e seu estado."""
        if not os.path.exists(ARQUIVO_CSV):
            return []

        try:
            dados = pd.read_csv(ARQUIVO_CSV)
            if dados.empty:
                return []
            sinais_no_modelo = set()
            modelo_atual = (
                os.path.exists(ARQUIVO_MODELO)
                and os.path.getmtime(ARQUIVO_MODELO) >= os.path.getmtime(ARQUIVO_CSV)
            )
            if modelo_atual:
                try:
                    sinais_no_modelo = set(joblib.load(ARQUIVO_MODELO).classes_)
                except Exception:
                    pass
            contagem = dados["rotulo"].value_counts().sort_index()
            return [
                (palavra, quantidade, palavra in sinais_no_modelo)
                for palavra, quantidade in contagem.items()
            ]
        except Exception as erro:
            messagebox.showerror("Sinais", f"Não foi possível ler os sinais: {erro}")
            return []

    def abrir_janela_sinais(self):
        """Abre uma janela exclusiva para consultar e apagar sinais."""
        if self.janela_sinais is not None and self.janela_sinais.winfo_exists():
            self.janela_sinais.deiconify()
            self.janela_sinais.lift()
            self.atualizar_janela_sinais()
            return

        self.janela_sinais = tk.Toplevel(self.root)
        self.janela_sinais.title("Sinais armazenados")
        self.janela_sinais.resizable(False, False)
        self.janela_sinais.protocol("WM_DELETE_WINDOW", self.fechar_janela_sinais)
        conteudo = tk.Frame(self.janela_sinais, padx=16, pady=14)
        conteudo.pack()
        tk.Label(conteudo, text="Sinais que a IA conhece", font=("Arial", 11, "bold")).pack(
            anchor="w"
        )
        tk.Label(
            conteudo,
            text="✓ treinado  |  • precisa clicar em TREINAR IA",
            fg="#555555",
        ).pack(anchor="w", pady=(2, 8))
        lista_area = tk.Frame(conteudo)
        lista_area.pack(fill="both", expand=True)
        self.lista_janela = tk.Listbox(lista_area, height=12, width=46)
        self.lista_janela.pack(side="left", fill="both", expand=True)
        self.lista_janela.bind("<Double-Button-1>", lambda evento: self.ver_sinal_selecionado())
        barra = tk.Scrollbar(lista_area, command=self.lista_janela.yview)
        barra.pack(side="right", fill="y")
        self.lista_janela.config(yscrollcommand=barra.set)
        tk.Button(conteudo, text="ATUALIZAR LISTA", command=self.atualizar_janela_sinais).pack(
            fill="x", pady=(10, 0)
        )
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

    def fechar_janela_sinais(self):
        self.janela_sinais.destroy()
        self.janela_sinais = None

    def atualizar_janela_sinais(self):
        if self.janela_sinais is None or not self.janela_sinais.winfo_exists():
            return
        self.lista_janela.delete(0, tk.END)
        self.sinais_da_janela = []
        sinais = self.obter_sinais()
        if not sinais:
            self.lista_janela.insert(tk.END, "Nenhum sinal ensinado ainda.")
            return
        for palavra, quantidade, treinado in sinais:
            estado = "✓ treinado" if treinado else "• precisa treinar"
            self.lista_janela.insert(tk.END, f"{palavra} — {quantidade} exemplos — {estado}")
            self.sinais_da_janela.append(palavra)

    def ver_sinal_selecionado(self):
        """Reproduz, num painel 3D em loop, os exemplos gravados do sinal
        selecionado — só as mãos, sem vídeo da câmera (que nunca é salvo)."""
        selecao = self.lista_janela.curselection()
        if not selecao or not self.sinais_da_janela:
            messagebox.showwarning("Selecione um sinal", "Selecione na lista o sinal que deseja ver.")
            return
        palavra = self.sinais_da_janela[selecao[0]]

        try:
            dados = pd.read_csv(ARQUIVO_CSV)
        except Exception as erro:
            messagebox.showerror("Não foi possível ler os dados", str(erro))
            return

        amostras = dados[dados["rotulo"] == palavra].drop(columns=["rotulo"])
        if amostras.empty:
            messagebox.showinfo("Sem exemplos", f"Não há exemplos gravados para “{palavra}”.")
            return

        # Encerra uma pré-visualização anterior, se houver, antes de abrir outra.
        self._fechar_reproducao()
        self._reproducao = {
            "nome": palavra,
            "vetores": amostras.values.tolist(),
            "indice": 0,
            "angulo": 0.0,
        }
        self._janela_reproducao = f"Kara Sygna - Pre-visualizacao: {palavra}"
        self._passo_reproducao()

    def _passo_reproducao(self):
        estado = self._reproducao
        if estado is None:
            return

        tamanho = (480, 420)
        largura, altura = tamanho
        tela = painel_3d(tamanho)
        vetor = estado["vetores"][estado["indice"]]
        esquerda = vetor[:FEATURES_POR_MAO]
        direita = vetor[FEATURES_POR_MAO:]
        cores = ((70, 190, 255), (150, 105, 255))

        # Zonas fixas — cada mão sempre aparece do seu lado, mesmo quando só
        # uma está presente naquele exemplo (o outro lado fica vazio, com o
        # rótulo). Isso evita a mão "pular" pro centro e trocar de lado.
        cv2.line(tela, (largura // 2, 55), (largura // 2, altura - 15),
                 (58, 63, 78), 1, cv2.LINE_AA)
        cv2.putText(tela, "ESQUERDA", (int(largura * 0.10), 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cores[0], 1, cv2.LINE_AA)
        cv2.putText(tela, "DIREITA", (int(largura * 0.62), 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cores[1], 1, cv2.LINE_AA)

        if not eh_vetor_zerado(esquerda):
            mao = _vetor_para_mao_falsa(esquerda)
            pontos = projetar_landmarks(
                mao, tamanho, 0.2, estado["angulo"], 0.27, ESCALA_REPRODUCAO,
            )
            desenhar_mao_3d(tela, pontos, cores[0])

        if not eh_vetor_zerado(direita):
            mao = _vetor_para_mao_falsa(direita)
            pontos = projetar_landmarks(
                mao, tamanho, 0.2, estado["angulo"], 0.73, ESCALA_REPRODUCAO,
            )
            desenhar_mao_3d(tela, pontos, cores[1])

        legenda = f"{estado['nome']}  —  exemplo {estado['indice'] + 1}/{len(estado['vetores'])}"
        cv2.putText(tela, legenda, (18, tamanho[1] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (225, 225, 235), 1, cv2.LINE_AA)
        cv2.putText(tela, "Feche a janela ou aperte 'q' para sair", (18, tamanho[1] - 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 165), 1, cv2.LINE_AA)

        cv2.imshow(self._janela_reproducao, tela)
        tecla = cv2.waitKey(1) & 0xFF

        fechada_pelo_x = cv2.getWindowProperty(
            self._janela_reproducao, cv2.WND_PROP_VISIBLE
        ) < 1
        if tecla == ord("q") or fechada_pelo_x:
            self._fechar_reproducao()
            return

        estado["indice"] = (estado["indice"] + 1) % len(estado["vetores"])
        estado["angulo"] += 0.03  # gira devagar, tipo um "carrossel" do sinal
        self.root.after(INTERVALO_REPRODUCAO_MS, self._passo_reproducao)

    def _fechar_reproducao(self):
        if self._janela_reproducao is not None:
            try:
                cv2.destroyWindow(self._janela_reproducao)
            except cv2.error:
                pass
        self._reproducao = None
        self._janela_reproducao = None

    def apagar_sinal_selecionado(self):
        selecao = self.lista_janela.curselection()
        if not selecao or not self.sinais_da_janela:
            messagebox.showwarning("Selecione um sinal", "Selecione na lista o sinal que deseja apagar.")
            return
        palavra = self.sinais_da_janela[selecao[0]]
        confirmar = messagebox.askyesno(
            "Apagar sinal",
            f"Apagar todos os exemplos de “{palavra}”? Esta ação não pode ser desfeita.",
        )
        if not confirmar:
            return
        try:
            dados = pd.read_csv(ARQUIVO_CSV)
            restantes = dados[dados["rotulo"] != palavra]
            restantes.to_csv(ARQUIVO_CSV, index=False)
            self.modelo = None
            self.modo = "parado"
            self.status.set(f"Sinal “{palavra}” apagado. Clique em TREINAR IA para atualizar o modelo.")
            self.atualizar_janela_sinais()
        except Exception as erro:
            messagebox.showerror("Não foi possível apagar", str(erro))

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
            self.atualizar_janela_sinais()
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
            self.atualizar_janela_sinais()
            self.status.set(f"IA treinada com {len(dados)} exemplos e {y.nunique()} sinais.")
            messagebox.showinfo("IA treinada", "Pronto! Agora use INICIAR RECONHECIMENTO.")
        except Exception as erro:
            messagebox.showerror("Não foi possível treinar", str(erro))

    def iniciar_reconhecimento(self):
        if (
            not os.path.exists(ARQUIVO_MODELO)
            or os.path.getmtime(ARQUIVO_MODELO) < os.path.getmtime(ARQUIVO_CSV)
        ):
            messagebox.showwarning(
                "Modelo desatualizado",
                "Os sinais mudaram. Clique em TREINAR IA antes de reconhecer.",
            )
            return
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

    def desenhar_maos_no_frame(self, frame, resultado):
        """Desenha o esqueleto de CADA mão detectada por cima do vídeo da
        câmera, com um rótulo indicando se é a mão esquerda ou direita.

        Como o frame já foi espelhado (cv2.flip), o rótulo "Right" que o
        MediaPipe devolve corresponde à mão esquerda de quem está na
        frente da câmera, e vice-versa — por isso a inversão abaixo.
        """
        if not (resultado.multi_hand_landmarks and resultado.multi_handedness):
            return

        cores = {"Esquerda": (70, 190, 255), "Direita": (150, 105, 255)}
        zipped = zip(resultado.multi_hand_landmarks, resultado.multi_handedness)
        for landmarks, info in zipped:
            lado = "Direita" if info.classification[0].label == "Left" else "Esquerda"
            cor = cores[lado]

            MP_DESENHO.draw_landmarks(
                frame, landmarks, MP_HANDS.HAND_CONNECTIONS,
                MP_DESENHO.DrawingSpec(color=cor, thickness=2, circle_radius=3),
                MP_DESENHO.DrawingSpec(color=cor, thickness=2),
            )

            pulso = landmarks.landmark[0]
            x = int(pulso.x * frame.shape[1])
            y = max(20, int(pulso.y * frame.shape[0]) - 20)
            cv2.putText(frame, lado, (x - 20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, cor, 2, cv2.LINE_AA)

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

        self.desenhar_maos_no_frame(frame, resultado)

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
        self._fechar_reproducao()
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