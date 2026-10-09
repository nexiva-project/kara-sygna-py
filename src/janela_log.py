
"""Janela de log em tempo real do Kara Sygna."""

import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime
import queue
import traceback


class JanelaLog:
    def __init__(self, root=None):
        self.root = root
        self.janela = tk.Toplevel(root) if root else tk.Tk()
        self.fila = queue.Queue()
        self.encerrando = False

        self.janela.title("Kara Sygna - Log em tempo real")
        self.janela.geometry("850x500")
        self.janela.minsize(550, 300)

        self.texto = scrolledtext.ScrolledText(
            self.janela,
            wrap=tk.WORD,
            font=("Consolas", 10),
            state=tk.DISABLED,
        )
        self.texto.pack(
            fill=tk.BOTH,
            expand=True,
            padx=8,
            pady=8,
        )

        self.texto.tag_configure("INFO", foreground="#222222")
        self.texto.tag_configure("APP", foreground="#222222")
        self.texto.tag_configure("CAMERA", foreground="#1769aa")
        self.texto.tag_configure("MODEL", foreground="#7b1fa2")
        self.texto.tag_configure("WARNING", foreground="#b36b00")
        self.texto.tag_configure("ERROR", foreground="#cc2222")
        self.texto.tag_configure("TRACEBACK", foreground="#cc2222")

        self.janela.protocol(
            "WM_DELETE_WINDOW",
            self.ocultar,
        )

        self.escrever(
            "Janela de log iniciada.",
            "INFO",
        )
        self.escrever(
            "Aguardando mensagens do programa...",
            "INFO",
        )

        self.janela.after(50, self._processar_fila)

    def escrever(self, mensagem, categoria="APP"):
        """Enfileira uma mensagem para aparecer na janela."""
        if self.encerrando:
            return

        horario = datetime.now().strftime("%H:%M:%S")
        categoria = str(categoria).upper()
        linha = f"[{horario}] [{categoria}] {mensagem}"

        self.fila.put((linha, categoria))

    def erro(self, mensagem, excecao=None):
        """Registra uma mensagem de erro e o traceback completo."""
        self.escrever(mensagem, "ERROR")

        if excecao is not None:
            detalhes = "".join(
                traceback.format_exception(
                    type(excecao),
                    excecao,
                    excecao.__traceback__,
                )
            )

            for linha in detalhes.rstrip().splitlines():
                self.escrever(linha, "TRACEBACK")

    def _processar_fila(self):
        """Atualiza a interface sem bloquear a câmera."""
        if self.encerrando:
            return

        try:
            while True:
                linha, categoria = self.fila.get_nowait()

                self.texto.configure(state=tk.NORMAL)
                self.texto.insert(
                    tk.END,
                    linha + "\n",
                    categoria,
                )
                self.texto.see(tk.END)
                self.texto.configure(state=tk.DISABLED)

        except queue.Empty:
            pass
        except tk.TclError:
            return

        try:
            if self.janela.winfo_exists():
                self.janela.after(50, self._processar_fila)
        except tk.TclError:
            pass

    def mostrar(self):
        try:
            self.janela.deiconify()
            self.janela.lift()
        except tk.TclError:
            pass

    def ocultar(self):
        try:
            self.janela.withdraw()
        except tk.TclError:
            pass

    def fechar(self):
        self.encerrando = True

        try:
            self.janela.destroy()
        except tk.TclError:
            pass


# Instância global utilizada pelos outros módulos.
janela_log = None


def iniciar_log(root=None):
    global janela_log

    if janela_log is not None:
        try:
            if janela_log.janela.winfo_exists():
                janela_log.mostrar()
                return janela_log
        except tk.TclError:
            pass

    janela_log = JanelaLog(root)
    return janela_log


def log(mensagem, categoria="APP"):
    """Registra uma mensagem normal."""
    if janela_log is not None:
        janela_log.escrever(mensagem, categoria)


def log_erro(mensagem, excecao=None):
    """Registra um erro e seus detalhes."""
    if janela_log is not None:
        janela_log.erro(mensagem, excecao)
