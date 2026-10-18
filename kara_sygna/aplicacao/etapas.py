"""Etapas de uma gravação de ensino (um Enter grava tudo, em sequência).

Exemplo: sinal ``carro`` com pré-sinal ``S``::

    pré-sinal S          (uma vez)
    carro 1/3: devagar
    carro 2/3: médio
    carro 3/3: rápido

Cada etapa vira uma rodada separada no CSV. Assim a IA vê o sinal em ritmos
diferentes e consegue se testar (treina com umas rodadas, testa com outra).
Os rótulos que não viram texto (``neutro``) gravam ``1/3``, ``2/3``, ``3/3``,
sem ritmo.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from kara_sygna import config
from kara_sygna.dados.repositorio import TIPO_PRE, TIPO_SINAL


def sem_acentos(texto: str) -> str:
    """Tira os acentos (o OpenCV não desenha acentos sobre a câmera)."""
    return "".join(
        c
        for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


@dataclass(frozen=True)
class Etapa:
    """Uma etapa da gravação.

    Attributes:
        tipo: ``TIPO_SINAL`` ou ``TIPO_PRE``.
        nome: Palavra do sinal (ou nome do pré-sinal).
        pre_sinal: Pré-sinal do sinal ("" se nenhum).
        duracao: Segundos de gravação (``None`` = até parar).
        legenda: Texto da etapa para a barra de status (com acentos).
    """

    tipo: str
    nome: str
    pre_sinal: str
    duracao: float | None
    legenda: str

    @property
    def legenda_video(self) -> str:
        """Texto da etapa sobre a câmera (maiúsculas no ritmo, sem acentos)."""
        if self.tipo == TIPO_PRE:
            return sem_acentos(f"PRE-SINAL {self.nome}")
        nome, _, ritmo = self.legenda.partition(": ")
        return sem_acentos(f"{nome}: {ritmo.upper()}" if ritmo else nome)


def montar_etapas(
    palavra: str,
    pre_sinal: str = "",
    duracao: float | None = None,
    ritmos: tuple[str, ...] = config.RITMOS_ENSINO,
) -> list[Etapa]:
    """Monta as etapas de uma gravação: pré-sinal (se houver) e o sinal N vezes.

    Args:
        palavra: Palavra do sinal.
        pre_sinal: Pré-sinal ("" = sem pré-sinal).
        duracao: Segundos de cada etapa. Vazio = ``SEGUNDOS_PRE_SINAL_PADRAO``
            quando há mais de uma etapa (para a sequência andar sozinha); com
            uma etapa só, grava até parar.
        ritmos: Ritmo de cada repetição do sinal (o tamanho diz quantas são).

    Returns:
        Lista de :class:`Etapa`, na ordem.
    """
    vezes = max(1, len(ritmos))
    total = vezes + (1 if pre_sinal else 0)
    tempo = (
        duracao
        if duracao is not None or total == 1
        else config.SEGUNDOS_PRE_SINAL_PADRAO
    )
    etapas: list[Etapa] = []
    if pre_sinal:
        etapas.append(Etapa(TIPO_PRE, pre_sinal, "", tempo, f"pré-sinal {pre_sinal}"))
    sem_ritmo = palavra.strip().lower() in config.ROTULOS_IGNORADOS or not ritmos
    for numero in range(1, vezes + 1):
        legenda = f"{palavra} {numero}/{vezes}"
        if not sem_ritmo:
            legenda += f": {ritmos[numero - 1]}"
        etapas.append(Etapa(TIPO_SINAL, palavra, pre_sinal, tempo, legenda))
    return etapas
