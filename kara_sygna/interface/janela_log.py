"""Janela de log em tempo real."""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from tkinter import scrolledtext

from kara_sygna.interface import tema
from kara_sygna.logs import NOME_RAIZ, FormatadorKara, categoria_do_logger

INTERVALO_MS = 100


class ManipuladorFilaLog(logging.Handler):
    """Handler do ``logging`` que guarda as linhas em uma fila.

    A fila é segura entre threads; a janela do Tk a esvazia na própria thread.
    """

    def __init__(self) -> None:
        """Cria a fila de linhas e o formato do log."""
        super().__init__()
        self.fila: queue.Queue[tuple[str, str]] = queue.Queue()
        self.setFormatter(FormatadorKara())

    def emit(self, record: logging.LogRecord) -> None:
        """Formata o registro e o coloca na fila com a etiqueta de cor adequada."""
        try:
            if record.levelno >= logging.ERROR:
                etiqueta = "ERROR"
            elif record.levelno >= logging.WARNING:
                etiqueta = "WARNING"
            else:
                etiqueta = categoria_do_logger(record.name)
            self.fila.put((self.format(record), etiqueta))
        except Exception:  # noqa: BLE001 - log nunca pode derrubar o programa
            self.handleError(record)


class JanelaLog:
    """Janela que mostra tudo o que o programa registra.

    Começa oculta; use :meth:`mostrar`. Fechar a janela só a oculta.

    Args:
        mestre: Janela principal do Tk.
    """

    def __init__(self, mestre: tk.Misc) -> None:
        """Cria a janela (oculta) e liga o handler de log do projeto."""
        self._manipulador = ManipuladorFilaLog()
        logging.getLogger(NOME_RAIZ).addHandler(self._manipulador)

        self._janela = tk.Toplevel(mestre)
        self._janela.title("Kara Sygna - Log")
        self._janela.geometry("850x500")
        self._janela.minsize(550, 300)
        self._janela.protocol("WM_DELETE_WINDOW", self.ocultar)

        self._texto = scrolledtext.ScrolledText(
            self._janela, wrap=tk.WORD, font=tema.FONTE_LOG, state=tk.DISABLED
        )
        self._texto.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        for etiqueta, cor in tema.COR_LOG.items():
            self._texto.tag_configure(etiqueta, foreground=cor)

        self._janela.withdraw()
        self._encerrada = False
        self._janela.after(INTERVALO_MS, self._esvaziar_fila)

    def _esvaziar_fila(self) -> None:
        """Move as linhas da fila para a caixa de texto."""
        if self._encerrada:
            return
        try:
            self._texto.configure(state=tk.NORMAL)
            while True:
                try:
                    linha, etiqueta = self._manipulador.fila.get_nowait()
                except queue.Empty:
                    break
                self._texto.insert(tk.END, linha + "\n", etiqueta)
            self._texto.see(tk.END)
            self._texto.configure(state=tk.DISABLED)
            self._janela.after(INTERVALO_MS, self._esvaziar_fila)
        except tk.TclError:
            pass  # janela destruída

    def mostrar(self) -> None:
        """Mostra a janela (trazendo-a para a frente)."""
        self._janela.deiconify()
        self._janela.lift()

    def ocultar(self) -> None:
        """Esconde a janela sem apagar o histórico."""
        self._janela.withdraw()

    def fechar(self) -> None:
        """Encerra a janela e desliga o handler (chamar ao sair do programa)."""
        self._encerrada = True
        logging.getLogger(NOME_RAIZ).removeHandler(self._manipulador)
        try:
            self._janela.destroy()
        except tk.TclError:
            pass
