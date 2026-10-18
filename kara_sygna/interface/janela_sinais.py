"""Janela com a lista dos sinais aprendidos."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox

from kara_sygna.aplicacao.controlador import PREFIXO_PRE, InfoSinal
from kara_sygna.interface import tema


class JanelaSinais(tk.Toplevel):
    """Lista de sinais com botões para visualizar e apagar.

    Args:
        mestre: Janela principal.
        obter_sinais: Devolve a lista atual de sinais.
        ao_ver: Chamada com a palavra escolhida para visualizar.
        ao_apagar: Chamada com a palavra escolhida para apagar (após confirmação).
    """

    def __init__(
        self,
        mestre: tk.Misc,
        obter_sinais: Callable[[], list[InfoSinal]],
        ao_ver: Callable[[str], None],
        ao_apagar: Callable[[str], None],
    ) -> None:
        """Cria a janela com a lista e os botões (veja a classe para os argumentos)."""
        super().__init__(mestre)
        self._obter_sinais = obter_sinais
        self._ao_ver = ao_ver
        self._ao_apagar = ao_apagar
        self._palavras: list[str] = []

        self.title("Sinais armazenados")
        self.geometry("520x500")

        conteudo = tk.Frame(self, padx=15, pady=15)
        conteudo.pack(fill="both", expand=True)
        tk.Label(
            conteudo, text="Sinais que a IA conhece", font=("Arial", 13, "bold")
        ).pack(anchor="w")
        tk.Label(
            conteudo, text="✓ treinado   |   • precisa treinar IA", fg="#666666"
        ).pack(anchor="w", pady=(3, 10))

        area = tk.Frame(conteudo)
        area.pack(fill="both", expand=True)
        self._lista = tk.Listbox(
            area, height=15, font=("Arial", 11), exportselection=False
        )
        self._lista.pack(side="left", fill="both", expand=True)
        self._lista.bind("<Double-Button-1>", lambda _e: self._ver())
        barra = tk.Scrollbar(area, command=self._lista.yview)
        barra.pack(side="right", fill="y")
        self._lista.config(yscrollcommand=barra.set)

        tk.Button(conteudo, text="ATUALIZAR LISTA", command=self.atualizar).pack(
            fill="x", pady=(10, 0)
        )
        tk.Button(conteudo, text="VER SINAL (SÓ AS MÃOS)", command=self._ver).pack(
            fill="x", pady=(7, 0)
        )
        tk.Button(
            conteudo,
            text="APAGAR SINAL SELECIONADO",
            command=self._apagar,
            fg=tema.PERIGO,
        ).pack(fill="x", pady=(7, 0))
        self.atualizar()

    def atualizar(self) -> None:
        """Recarrega a lista a partir do CSV."""
        self._lista.delete(0, tk.END)
        self._palavras = []
        try:
            sinais = self._obter_sinais()
        except Exception as erro:  # noqa: BLE001 - mostra o problema em vez de travar a janela
            messagebox.showerror(
                "Sinais", f"Não foi possível ler os sinais:\n{erro}", parent=self
            )
            return
        if not sinais:
            self._lista.insert(tk.END, "Nenhum sinal ensinado ainda.")
            return
        for sinal in sinais:
            estado = "✓ treinado" if sinal.treinado else "• precisa treinar"
            if sinal.eh_pre_sinal:
                nome = f"[pré-sinal] {sinal.palavra}"
            elif sinal.pre_sinal:
                nome = f"{sinal.palavra} (pré-sinal {sinal.pre_sinal})"
            else:
                nome = sinal.palavra
            self._lista.insert(tk.END, f"{nome} — {sinal.exemplos} exemplos — {estado}")
            # Chave: o pré-sinal "S" vira "pre:S" (veja InfoSinal.chave).
            self._palavras.append(sinal.chave)

    def _selecionada(self, acao: str) -> str | None:
        """Palavra selecionada, ou ``None`` (avisando o usuário) se não houver."""
        selecao = self._lista.curselection()
        if not selecao or not self._palavras:
            messagebox.showwarning(
                "Selecione um sinal",
                f"Selecione na lista o sinal que deseja {acao}.",
                parent=self,
            )
            return None
        return self._palavras[selecao[0]]

    def _ver(self) -> None:
        """Pede a pré-visualização do sinal selecionado."""
        palavra = self._selecionada("ver")
        if palavra:
            self._ao_ver(palavra)

    def _apagar(self) -> None:
        """Apaga o sinal selecionado depois de confirmar."""
        palavra = self._selecionada("apagar")
        if not palavra:
            return
        if messagebox.askyesno(
            "Apagar sinal",
            f"Apagar todos os exemplos de “{palavra.removeprefix(PREFIXO_PRE)}”?"
            "\n\nEsta ação não pode ser desfeita.",
            parent=self,
        ):
            self._ao_apagar(palavra)
