"""Diagnóstico do reconhecimento: o que a IA viu e decidiu em cada quadro.

Cada sessão de reconhecimento grava um CSV em ``config.PASTA_DIAGNOSTICO``
(abre no Excel), com uma linha por quadro da câmera: mãos, corpo e rosto
vistos, pré-sinal, os três sinais mais prováveis com as porcentagens e o
motivo da decisão. Ao parar, um resumo vai para o log.

Motivos:

* ``sem_mao``: nenhuma mão no quadro.
* ``preparando``: a janela de quadros ainda não encheu.
* ``incerto``: a IA não passou da confiança ou da margem mínimas.
* ``falta_pre``: a IA achou um sinal que tem pré-sinal, com a confiança
  normal, mas sem o pré-sinal ter sido visto ela exige mais
  (``config.CONFIANCA_SEM_PRE_SINAL``).
* ``ok``: sinal aceito neste quadro (vira voto).
* ``ok_sem_pre``: sinal aceito sem o pré-sinal ter sido visto, porque a IA
  teve muita certeza (``config.CONFIANCA_SEM_PRE_SINAL``).
* ``repete``: quadro igual ao anterior (o filtro de quadros não o usou); vale
  a decisão anterior.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from kara_sygna import config

logger = logging.getLogger(__name__)

CAMPOS = (
    "t_s",
    "mao_esq",
    "mao_dir",
    "corpo",
    "rosto",
    "pre_visto",
    "pre_pct",
    "pre_ativo",
    "pre_restante_s",
    "motivo",
    "top1",
    "top1_pct",
    "top2",
    "top2_pct",
    "top3",
    "top3_pct",
    "margem_pct",
    "limite_pct",
    "voto",
    "escrito",
)
MOTIVOS = (
    "sem_mao",
    "preparando",
    "incerto",
    "falta_pre",
    "ok",
    "ok_sem_pre",
    "repete",
)


class DiagnosticoReconhecimento:
    """Arquivo de diagnóstico de uma sessão de reconhecimento.

    Args:
        pasta: Onde criar o CSV. Padrão: ``config.PASTA_DIAGNOSTICO``.
        relogio: Função que devolve o tempo em segundos.
    """

    def __init__(
        self, pasta: Path | None = None, relogio: Callable[[], float] | None = None
    ) -> None:
        """Cria o arquivo (veja a classe para os argumentos)."""
        import time

        pasta = Path(pasta) if pasta is not None else config.PASTA_DIAGNOSTICO
        pasta.mkdir(parents=True, exist_ok=True)
        self._relogio = relogio or time.monotonic
        self._inicio = self._relogio()
        # Nome único: duas sessões no mesmo segundo não se sobrescrevem ("x"
        # nunca abre um arquivo que já existe). Fica aberto até fechar().
        momento = f"{datetime.now():%Y%m%d_%H%M%S}"
        for numero in range(1000):
            sufixo = f"_{numero}" if numero else ""
            self.caminho = pasta / f"reconhecimento_{momento}{sufixo}.csv"
            try:
                self._arquivo = open(  # noqa: SIM115
                    self.caminho, "x", newline="", encoding="utf-8"
                )
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError(f"Sem nome livre para o diagnóstico em {pasta}.")
        self._escritor = csv.DictWriter(self._arquivo, fieldnames=CAMPOS)
        self._escritor.writeheader()
        self._linhas = 0
        self._motivos: Counter[str] = Counter()
        self._incertos: Counter[str] = Counter()
        self._soma_pct: dict[str, float] = {}
        self._escritos: Counter[str] = Counter()
        self._pre_vistos: Counter[str] = Counter()

    def registrar(self, linha: dict) -> None:
        """Grava uma linha (campos que faltarem ficam vazios)."""
        linha = {"t_s": round(self._relogio() - self._inicio, 3), **linha}
        self._escritor.writerow({c: linha.get(c, "") for c in CAMPOS})
        self._linhas += 1
        if self._linhas % 30 == 0:
            self._arquivo.flush()

        motivo = linha.get("motivo", "")
        self._motivos[motivo] += 1
        if motivo in ("incerto", "falta_pre") and linha.get("top1"):
            self._incertos[linha["top1"]] += 1
            self._soma_pct[linha["top1"]] = self._soma_pct.get(
                linha["top1"], 0.0
            ) + float(linha.get("top1_pct") or 0)
        if linha.get("escrito"):
            self._escritos[linha["escrito"]] += 1
        if linha.get("pre_novo"):
            self._pre_vistos[linha["pre_novo"]] += 1

    def fechar(self) -> str:
        """Fecha o arquivo e devolve um resumo de uma linha para o log."""
        self._arquivo.close()
        total = max(1, self._linhas)
        partes = [
            f"{m} {100 * self._motivos[m] / total:.0f}%"
            for m in MOTIVOS
            if self._motivos[m]
        ]
        resumo = f"Resumo do reconhecimento: {self._linhas} quadros; " + ", ".join(
            partes
        )
        if self._incertos:
            sinal, vezes = self._incertos.most_common(1)[0]
            media = self._soma_pct[sinal] / vezes
            resumo += (
                f". Quando ficou incerto, achava mais “{sinal}” ({media:.0f}% em média)"
            )
        escritos = ", ".join(f"{s}×{n}" for s, n in self._escritos.items()) or "nada"
        pre = ", ".join(f"{s}×{n}" for s, n in self._pre_vistos.items()) or "nenhum"
        resumo += f". Pré-sinais vistos: {pre}. Escrito: {escritos}. Detalhes: {self.caminho.name}"
        return resumo
