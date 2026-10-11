"""Execução de tarefas demoradas fora da thread da interface."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

INTERVALO_VERIFICACAO_MS = 100


class ExecutorTk:
    """Roda uma função em uma thread e devolve o resultado na thread do Tk.

    O Tkinter não pode ser usado de outras threads. Por isso a tarefa só
    escreve em uma fila, e o Tk lê essa fila a cada 100 ms com ``after``.

    Args:
        widget: Qualquer widget do Tk (usado para agendar a leitura da fila).
    """

    def __init__(self, widget: Any) -> None:
        """Guarda o widget usado para agendar a leitura da fila."""
        self._widget = widget
        self._fila: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._ativo = False

    @property
    def ocupado(self) -> bool:
        """``True`` enquanto uma tarefa está em andamento."""
        return self._ativo

    def executar(
        self,
        trabalho: Callable[[Callable[[str], None]], Any],
        ao_concluir: Callable[[Any], None],
        ao_falhar: Callable[[BaseException], None],
        ao_progresso: Callable[[str], None] | None = None,
    ) -> None:
        """Inicia a tarefa.

        Args:
            trabalho: Função que recebe um callback de progresso (``str``) e
                devolve o resultado. Roda na thread de fundo.
            ao_concluir: Chamada na thread do Tk com o resultado.
            ao_falhar: Chamada na thread do Tk com a exceção, se a tarefa falhar.
            ao_progresso: Chamada na thread do Tk para cada mensagem de progresso.

        Raises:
            RuntimeError: Se já houver uma tarefa em andamento.
        """
        if self._ativo:
            raise RuntimeError("Já existe uma tarefa em andamento.")
        self._ativo = True

        def rodar() -> None:
            """Executa o trabalho na thread de fundo e envia o resultado (ou o erro) pela fila."""
            try:
                resultado = trabalho(lambda msg: self._fila.put(("progresso", msg)))
                self._fila.put(("ok", resultado))
            except BaseException as erro:  # noqa: BLE001 - repassa qualquer falha para a interface
                self._fila.put(("erro", erro))

        threading.Thread(target=rodar, daemon=True, name="kara-tarefa").start()
        self._verificar(ao_concluir, ao_falhar, ao_progresso)

    def _verificar(self, ao_concluir, ao_falhar, ao_progresso) -> None:
        """Lê a fila e agenda a próxima leitura (roda na thread do Tk)."""
        terminou = False
        try:
            while True:
                tipo, valor = self._fila.get_nowait()
                if tipo == "progresso":
                    if ao_progresso:
                        ao_progresso(valor)
                    continue
                self._ativo = False
                terminou = True
                (ao_concluir if tipo == "ok" else ao_falhar)(valor)
                break
        except queue.Empty:
            pass
        if not terminou:
            try:
                self._widget.after(
                    INTERVALO_VERIFICACAO_MS,
                    self._verificar,
                    ao_concluir,
                    ao_falhar,
                    ao_progresso,
                )
            except Exception:  # noqa: BLE001 - janela já foi destruída
                self._ativo = False
