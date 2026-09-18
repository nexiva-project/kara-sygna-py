"""
kara-sygna-py — app.py: painel de controle (interface gráfica).

Objetivo:
    Uma janela única com:
        - Campo para digitar o nome do sinal
        - Botão "Gravar" (grava por 60s, depois aumenta os dados e
          retreina o modelo automaticamente)
        - Botão "Reconhecer sinais" (abre o reconhecimento ao vivo)

    A câmera continua abrindo em uma janela separada (é assim que o
    OpenCV funciona) — o painel fica livre para você continuar a usar
    enquanto a gravação ou o reconhecimento acontecem.

Como rodar:
    python src/app.py

Dependência extra:
    Nenhuma! O tkinter já vem embutido no Python (na maioria das
    instalações Windows/Mac). Se der erro de "No module named tkinter"
    no Linux, instale com: sudo apt install python3-tk
"""

import threading
import tkinter as tk
from tkinter import messagebox, ttk

import pipeline

DURACAO_PADRAO = pipeline.DURACAO_PADRAO_SEGUNDOS


class Aplicativo:
    def __init__(self, raiz):
        self.raiz = raiz
        raiz.title("kara-sygna")
        raiz.geometry("420x260")
        raiz.resizable(False, False)

        # --- Campo de nome do sinal ---
        tk.Label(raiz, text="Nome do sinal:", font=("Segoe UI", 11)).pack(pady=(20, 5))
        self.campo_rotulo = tk.Entry(raiz, font=("Segoe UI", 11), justify="center")
        self.campo_rotulo.pack(pady=(0, 15))

        # --- Duração da gravação ---
        frame_duracao = tk.Frame(raiz)
        frame_duracao.pack(pady=(0, 15))
        tk.Label(frame_duracao, text="Duração (segundos):").pack(side="left", padx=5)
        self.campo_duracao = tk.Entry(frame_duracao, width=6, justify="center")
        self.campo_duracao.insert(0, str(DURACAO_PADRAO))
        self.campo_duracao.pack(side="left")

        # --- Botões ---
        frame_botoes = tk.Frame(raiz)
        frame_botoes.pack(pady=5)

        self.botao_gravar = tk.Button(
            frame_botoes, text="Gravar", width=18, command=self.ao_clicar_gravar
        )
        self.botao_gravar.grid(row=0, column=0, padx=5)

        self.botao_reconhecer = tk.Button(
            frame_botoes, text="Reconhecer sinais", width=18,
            command=self.ao_clicar_reconhecer,
        )
        self.botao_reconhecer.grid(row=0, column=1, padx=5)

        # --- Barra de progresso (indeterminada, só pra indicar "ocupado") ---
        self.barra_progresso = ttk.Progressbar(raiz, mode="indeterminate", length=360)
        self.barra_progresso.pack(pady=15)

        # --- Barra de progresso (horizontal e determinate) ---
        # self.barra_progresso_ = ttk.Progressbar(raiz, orient="horizontal", mode="determinate", length=360)
        # self.barra_progresso_.pack(pady=15)

        # --- Status ---
        self.rotulo_status = tk.Label(raiz, text="Pronto.", fg="gray")
        self.rotulo_status.pack()

    # ------------------------------------------------------------------
    # Utilitário para atualizar a interface a partir de outra thread.
    # O tkinter não é thread-safe: qualquer atualização de widget feita
    # a partir de uma thread de fundo deve passar por `raiz.after(...)`.
    # ------------------------------------------------------------------
    def _atualizar_status(self, texto):
        self.raiz.after(0, lambda: self.rotulo_status.config(text=texto))

    def _travar_botoes(self, travado):
        estado = "disabled" if travado else "normal"
        self.raiz.after(0, lambda: self.botao_gravar.config(state=estado))
        self.raiz.after(0, lambda: self.botao_reconhecer.config(state=estado))
        if travado:
            self.raiz.after(0, self.barra_progresso.start)
        else:
            self.raiz.after(0, self.barra_progresso.stop)

    # ------------------------------------------------------------------
    # Botão "Gravar"
    # ------------------------------------------------------------------
    def ao_clicar_gravar(self):
        rotulo = self.campo_rotulo.get().strip()
        if not rotulo:
            messagebox.showwarning("Atenção", "Digite o nome do sinal antes de gravar.")
            return

        try:
            duracao = int(self.campo_duracao.get())
        except ValueError:
            messagebox.showwarning("Atenção", "Duração inválida — use um número de segundos.")
            return

        self._travar_botoes(True)
        thread = threading.Thread(
            target=self._executar_gravacao_e_treino, args=(rotulo, duracao), daemon=True
        )
        thread.start()

    def _executar_gravacao_e_treino(self, rotulo, duracao):
        try:
            self._atualizar_status(f"Gravando '{rotulo}'... mostre o sinal para a câmera.")
            total = pipeline.coletar_amostras(
                rotulo, duracao_segundos=duracao,
                atualizar_status=self._atualizar_status,
            )

            self._atualizar_status("Aguarde... aumentando dados (espelhamento).")
            adicionadas = pipeline.aumentar_dados()

            self._atualizar_status("Aguarde... treinando modelo.")
            resultado = pipeline.treinar_modelo()

            self._atualizar_status("Pronto.")
            self.raiz.after(0, lambda: messagebox.showinfo(
                "Sucesso",
                f"Sinal '{rotulo}' gravado com {total} amostras "
                f"(+{adicionadas} espelhadas).\n\n"
                f"Modelo retreinado com {resultado['total_amostras']} amostras "
                f"no total, cobrindo {len(resultado['sinais'])} sinais.\n"
                f"Acurácia no teste: {resultado['acuracia']:.0%}",
            ))
        except Exception as erro:
            self._atualizar_status("Erro.")
            self.raiz.after(0, lambda: messagebox.showerror("Erro", str(erro)))
        finally:
            self._travar_botoes(False)

    # ------------------------------------------------------------------
    # Botão "Reconhecer sinais"
    # ------------------------------------------------------------------
    def ao_clicar_reconhecer(self):
        self._travar_botoes(True)
        thread = threading.Thread(target=self._executar_reconhecimento, daemon=True)
        thread.start()

    def _executar_reconhecimento(self):
        try:
            self._atualizar_status("Reconhecendo... feche a janela da câmera para voltar.")
            pipeline.reconhecer_sinais()
            self._atualizar_status("Pronto.")
        except Exception as erro:
            self._atualizar_status("Erro.")
            self.raiz.after(0, lambda: messagebox.showerror("Erro", str(erro)))
        finally:
            self._travar_botoes(False)


if __name__ == "__main__":
    raiz = tk.Tk()
    Aplicativo(raiz)
    raiz.mainloop()