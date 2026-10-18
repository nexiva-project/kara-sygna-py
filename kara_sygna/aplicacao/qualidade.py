"""Qualidade de uma gravação de ensino: diz na hora se vale gravar de novo.

Conta, em todos os quadros da câmera durante o ensino, em quantos apareceu
alguma mão e o rosto, e, nos quadros que foram para o CSV (depois de preencher
as mãos que sumiram por pouco tempo), em quantos estão as duas mãos. A IA
aprende mal com gravações em que as mãos somem.

Os avisos dependem do que está sendo gravado:

* **pré-sinal**: é de uma mão só, então não há aviso de "segunda mão";
* **neutro** (rótulos ignorados): mãos paradas no colo, fora da câmera, são
  normais, então não há aviso de mãos fora nem de segunda mão.
"""

from __future__ import annotations

import numpy as np

from kara_sygna import config
from kara_sygna.dados.repositorio import TIPO_PRE, TIPO_SINAL
from kara_sygna.visao.quadro import COLUNA_ROSTO_PRESENTE, maos_presentes

#: Abaixo disso, a gravação recebe um aviso.
MINIMO_COM_MAO = 0.6
MINIMO_ROSTO = 0.7
#: Se as duas mãos aparecem em parte dos quadros (entre os dois valores), o
#: sinal provavelmente é de duas mãos e a segunda sumiu muitas vezes. Abaixo
#: do mínimo, é um sinal de uma mão só e está tudo bem.
DUAS_MAOS_MINIMO_SINAL = 0.2
DUAS_MAOS_BOM = 0.75


class QualidadeGravacao:
    """Acumula a contagem de uma rodada de ensino.

    Args:
        tipo: ``TIPO_SINAL`` ou ``TIPO_PRE``.
        rotulo: Palavra gravada (o ``neutro`` tem avisos diferentes).
    """

    def __init__(self, tipo: str = TIPO_SINAL, rotulo: str = "") -> None:
        """Começa a contagem zerada (veja a classe para os argumentos)."""
        self.uma_mao = tipo == TIPO_PRE
        self.sem_sinal = rotulo.strip().lower() in config.ROTULOS_IGNORADOS
        self.quadros = 0
        self.com_mao = 0
        self.duas_maos = 0
        self.com_rosto = 0
        # Quadros que foram para o CSV, já com as mãos preenchidas.
        self.salvos_com_mao = 0
        self.salvos_duas_maos = 0

    def adicionar(self, vetor_quadro: np.ndarray) -> None:
        """Conta um quadro da câmera (vetor de 223 valores)."""
        vetor = np.asarray(vetor_quadro, dtype=np.float64).reshape(1, -1)
        presentes = maos_presentes(vetor)[0]
        self.quadros += 1
        self.com_mao += int(presentes.any())
        self.duas_maos += int(presentes.all())
        self.com_rosto += int(vetor[0, COLUNA_ROSTO_PRESENTE] > 0.5)

    def adicionar_salvos(self, quadros: np.ndarray) -> None:
        """Conta os quadros que foram para o CSV (depois do preenchimento)."""
        presentes = maos_presentes(quadros)
        self.salvos_com_mao += int(presentes.any(axis=1).sum())
        self.salvos_duas_maos += int(presentes.all(axis=1).sum())

    def _fracao_duas_maos(self) -> float | None:
        """Fração dos quadros com mão que têm as duas (salvos, se houver)."""
        if self.salvos_com_mao:
            return self.salvos_duas_maos / self.salvos_com_mao
        if self.com_mao:
            return self.duas_maos / self.com_mao
        return None

    def avisos(self, gravados: int) -> list[str]:
        """Problemas encontrados (lista vazia = gravação boa).

        Args:
            gravados: Quantos quadros foram para o CSV nesta rodada.
        """
        if not self.quadros or not gravados:
            return ["nenhuma mão foi gravada"]
        avisos = []
        if not self.sem_sinal and self.com_mao / self.quadros < MINIMO_COM_MAO:
            avisos.append("as mãos ficaram fora da câmera muitas vezes")
        duas = self._fracao_duas_maos()
        if (
            not (self.uma_mao or self.sem_sinal)
            and duas is not None
            and DUAS_MAOS_MINIMO_SINAL <= duas < DUAS_MAOS_BOM
        ):
            avisos.append("a segunda mão sumiu muitas vezes")
        if self.com_rosto / self.quadros < MINIMO_ROSTO:
            avisos.append("o rosto não apareceu bem")
        if gravados < 2 * config.QUADROS_POR_JANELA:
            avisos.append("a gravação foi curta")
        return avisos

    def resumo(self, gravados: int) -> str:
        """Texto curto para a barra de status."""
        avisos = self.avisos(gravados)
        if self.quadros:
            duas = self._fracao_duas_maos()
            numeros = (
                f"mãos em {self.com_mao / self.quadros:.0%}, "
                f"2 mãos em {duas or 0:.0%} dos quadros gravados, "
                f"rosto em {self.com_rosto / self.quadros:.0%}"
            )
        else:
            numeros = "nenhum quadro"
        if avisos:
            return f"{numeros}. ATENÇÃO, grave de novo: {'; '.join(avisos)}."
        return f"{numeros}. Boa gravação."
