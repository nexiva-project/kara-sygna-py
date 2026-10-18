"""Armazenamento dos exemplos de sinais em um arquivo CSV.

Formato: uma linha por quadro da câmera, com as colunas ``rotulo``, ``rodada``,
``quadro``, ``tipo``, ``pre_sinal`` e as 223 características de
:mod:`kara_sygna.visao.quadro`.

* ``rodada`` identifica a sessão de gravação (um clique em "COMEÇAR A
  ENSINAR"); a validação do modelo usa isso para nunca misturar quadros da
  mesma sessão no treino e no teste.
* ``quadro`` é a posição do quadro dentro da rodada. Os quadros são gravados
  em sequência, porque a IA aprende o movimento; quais quadros da câmera
  entram é decidido por :class:`kara_sygna.aplicacao.filtro_quadros.FiltroQuadros`.
* ``tipo`` é ``"sinal"`` ou ``"pre"`` (pré-sinal: uma forma de mão, como o
  "S", mostrada de todos os lados antes de sinais que começam com ela).
* ``pre_sinal`` diz, nas linhas de sinal, qual pré-sinal vem antes dele
  (vazio se nenhum).
* ``formato_vN`` guarda a versão do formato (veja ``VERSAO_FORMATO``).
"""

from __future__ import annotations

import csv
import logging
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kara_sygna import config
from kara_sygna.visao.quadro import (
    IDX_MAOS,
    TOTAL_FEATURES_QUADRO,
    maos_presentes,
    nomes_features_quadro,
)

logger = logging.getLogger(__name__)

COLUNA_ROTULO = "rotulo"
COLUNA_RODADA = "rodada"
COLUNA_QUADRO = "quadro"
COLUNA_TIPO = "tipo"
COLUNA_PRE_SINAL = "pre_sinal"
TIPO_SINAL = "sinal"
TIPO_PRE = "pre"
#: Versão do formato. Aumente quando o SIGNIFICADO das colunas mudar sem mudar
#: os nomes (ex.: v2 = a mão direita deixou de aparecer como "esq"): o nome da
#: coluna muda junto, e o CSV antigo vai sozinho para o backup.
VERSAO_FORMATO = 2
COLUNA_FORMATO = f"formato_v{VERSAO_FORMATO}"


def nome_pre_sinal(nome: str) -> str:
    """Nome padrão de um pré-sinal: sem espaços nas pontas e em maiúsculas.

    Assim "s" e "S" são o mesmo pré-sinal (antes viravam dois, e os exemplos
    se dividiam entre eles).
    """
    return nome.strip().upper()


# 4 casas bastam (as medidas são relativas, da ordem de 0,01 a 10) e deixam
# o CSV menor.
CASAS_DECIMAIS = 4


class DadosInvalidosError(ValueError):
    """O CSV existe, mas não tem o formato esperado."""


@dataclass(frozen=True)
class ConjuntoDados:
    """Todos os exemplos do CSV, já separados para o treinamento.

    Attributes:
        X: Matriz (n, colunas) de características: 223 por quadro, ao ler o
            CSV, ou as colunas de uma janela, depois de
            :func:`kara_sygna.ia.sequencia.montar_amostras`.
        y: Vetor (n,) com o rótulo (palavra) de cada linha.
        grupos: Vetor (n,) com a rodada de cada linha.
    """

    X: np.ndarray
    y: np.ndarray
    grupos: np.ndarray

    def __len__(self) -> int:
        """Quantidade de linhas."""
        return len(self.y)

    @property
    def classes(self) -> list[str]:
        """Rótulos distintos, em ordem alfabética."""
        return sorted(set(self.y.tolist()))


def cabecalho() -> list[str]:
    """Cabeçalho completo do CSV: identificação da linha e as 223 características."""
    return [
        COLUNA_ROTULO,
        COLUNA_RODADA,
        COLUNA_QUADRO,
        COLUNA_TIPO,
        COLUNA_PRE_SINAL,
        COLUNA_FORMATO,
    ] + nomes_features_quadro()


_ultimo_id_rodada = 0


def novo_id_rodada() -> int:
    """Gera o identificador de uma nova rodada (milissegundos desde 1970).

    Nunca repete um id já entregue, mesmo com dois cliques no mesmo instante.
    """
    global _ultimo_id_rodada
    _ultimo_id_rodada = max(time.time_ns() // 1_000_000, _ultimo_id_rodada + 1)
    return _ultimo_id_rodada


class RepositorioSinais:
    """Lê e escreve o CSV de exemplos.

    Args:
        caminho: Arquivo CSV. Padrão: ``config.ARQUIVO_CSV``.
    """

    def __init__(self, caminho: Path | None = None) -> None:
        """Define qual arquivo CSV será usado (veja a classe para o argumento)."""
        self.caminho = Path(caminho) if caminho is not None else config.ARQUIVO_CSV

    # -- estado do arquivo --------------------------------------------------

    def existe(self) -> bool:
        """Diz se o CSV existe e tem ao menos o cabeçalho."""
        return self.caminho.exists() and self.caminho.stat().st_size > 0

    def modificado_em(self) -> float:
        """Data da última modificação do CSV (``0.0`` se não existir)."""
        return self.caminho.stat().st_mtime if self.caminho.exists() else 0.0

    def garantir_arquivo(self) -> None:
        """Cria o CSV (só com o cabeçalho) se ainda não existir."""
        if self.existe():
            return
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        with open(self.caminho, "w", newline="", encoding="utf-8") as arquivo:
            csv.writer(arquivo).writerow(cabecalho())
        logger.info("CSV criado: %s", self.caminho)

    # -- escrita ------------------------------------------------------------

    def anexar_quadros(
        self,
        rotulo: str,
        rodada: int,
        quadros: Iterable[tuple[int, Iterable[float]]],
        tipo: str = TIPO_SINAL,
        pre_sinal: str = "",
    ) -> int:
        """Grava quadros de uma rodada.

        Args:
            rotulo: Palavra do sinal (ou nome do pré-sinal).
            rodada: Identificador da rodada (:func:`novo_id_rodada`).
            quadros: Pares ``(numero_do_quadro, vetor de 223 valores)``.
            tipo: ``TIPO_SINAL`` ou ``TIPO_PRE``.
            pre_sinal: Pré-sinal que vem antes deste sinal (vazio se nenhum).

        Returns:
            Quantas linhas foram gravadas.

        Raises:
            ValueError: Se algum vetor não tiver 223 valores (nada é gravado).
        """
        linhas = []
        for numero, vetor in quadros:
            vetor = np.asarray(list(vetor), dtype=np.float64)
            if vetor.size != TOTAL_FEATURES_QUADRO:
                raise ValueError(
                    f"O vetor tem {vetor.size} valores; esperava {TOTAL_FEATURES_QUADRO}."
                )
            linhas.append(
                [rotulo, rodada, int(numero), tipo, pre_sinal, VERSAO_FORMATO]
                + [round(float(x), CASAS_DECIMAIS) for x in vetor]
            )
        if not linhas:
            return 0

        self.garantir_arquivo()
        with open(self.caminho, "a", newline="", encoding="utf-8") as arquivo:
            csv.writer(arquivo).writerows(linhas)
        return len(linhas)

    def apagar_sinal(self, rotulo: str, tipo: str = TIPO_SINAL) -> int:
        """Remove todos os exemplos de um sinal (ou de um pré-sinal).

        Args:
            rotulo: Palavra a apagar.
            tipo: ``TIPO_SINAL`` ou ``TIPO_PRE``.

        Returns:
            Quantas linhas foram removidas.
        """
        tabela = self._ler_tabela()
        if tipo == TIPO_PRE:
            rotulo = nome_pre_sinal(rotulo)
        apagar = (tabela[COLUNA_ROTULO] == rotulo) & (tabela[COLUNA_TIPO] == tipo)
        restantes = tabela[~apagar]
        removidas = len(tabela) - len(restantes)
        restantes.to_csv(self.caminho, index=False, encoding="utf-8")
        logger.info("Sinal apagado: %s (%d linhas)", rotulo, removidas)
        return removidas

    # -- leitura ------------------------------------------------------------

    def _ler_tabela(self) -> pd.DataFrame:
        """Lê o CSV validando as colunas.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            DadosInvalidosError: Se faltarem colunas ou houver valores vazios.
        """
        if not self.existe():
            raise FileNotFoundError(f"CSV não encontrado: {self.caminho}")
        tabela = pd.read_csv(
            self.caminho,
            dtype={COLUNA_ROTULO: str, COLUNA_TIPO: str, COLUNA_PRE_SINAL: str},
        )
        if COLUNA_PRE_SINAL in tabela.columns:
            # Vazio = sinal sem pré-sinal (o pandas lê como NaN).
            tabela[COLUNA_PRE_SINAL] = (
                tabela[COLUNA_PRE_SINAL].fillna("").astype(str).map(nome_pre_sinal)
            )
        if COLUNA_TIPO in tabela.columns and COLUNA_ROTULO in tabela.columns:
            # Pré-sinais gravados com outra caixa ("s") juntam com "S".
            pre = tabela[COLUNA_TIPO] == TIPO_PRE
            tabela.loc[pre, COLUNA_ROTULO] = (
                tabela.loc[pre, COLUNA_ROTULO].astype(str).map(nome_pre_sinal)
            )

        faltando = [c for c in cabecalho() if c not in tabela.columns]
        if faltando:
            raise DadosInvalidosError(
                f"O CSV está em formato antigo (faltam colunas como '{faltando[0]}'). "
                "Abra o programa de novo: ele guarda o arquivo antigo em um backup "
                "e começa um novo."
            )
        if tabela[cabecalho()].isnull().any().any():
            raise DadosInvalidosError(
                "O CSV contém valores vazios. Corrija ou apague as linhas incompletas."
            )
        return tabela

    def carregar(self, tipo: str = TIPO_SINAL) -> ConjuntoDados:
        """Carrega os exemplos de um tipo (sinais ou pré-sinais) para o treino.

        Raises:
            FileNotFoundError: Se o CSV não existir.
            DadosInvalidosError: Se o formato estiver errado.
        """
        tabela = self._ler_tabela()
        tabela = tabela[tabela[COLUNA_TIPO] == tipo]
        # Garante a ordem de gravação dentro de cada rodada.
        tabela = tabela.sort_values([COLUNA_RODADA, COLUNA_QUADRO], kind="stable")
        return ConjuntoDados(
            X=tabela[nomes_features_quadro()].to_numpy(dtype=np.float64),
            y=tabela[COLUNA_ROTULO].to_numpy(dtype=str),
            grupos=tabela[COLUNA_RODADA].to_numpy(),
        )

    def pre_sinais_dos_sinais(self) -> dict[str, str]:
        """Qual pré-sinal vem antes de cada sinal que tem um.

        Se um sinal foi gravado com pré-sinais diferentes, vale o da gravação
        mais recente.

        Returns:
            ``{sinal: pre_sinal}``, só com os sinais que têm pré-sinal.
        """
        if not self.existe():
            return {}
        tabela = self._ler_tabela()
        tabela = tabela[tabela[COLUNA_TIPO] == TIPO_SINAL].sort_values(
            COLUNA_RODADA, kind="stable"
        )
        # A gravação mais recente vale, inclusive se foi SEM pré-sinal.
        ultimo = dict(zip(tabela[COLUNA_ROTULO], tabela[COLUNA_PRE_SINAL], strict=True))
        return {str(sinal): str(pre) for sinal, pre in ultimo.items() if pre}

    def contagem_por_sinal(self, tipo: str = TIPO_SINAL) -> dict[str, int]:
        """Quantos exemplos existem de cada sinal (ou pré-sinal), em ordem alfabética.

        Returns:
            Dicionário ``{palavra: quantidade}``; vazio se não houver dados.
        """
        if not self.existe():
            return {}
        tabela = self._ler_tabela()
        tabela = tabela[tabela[COLUNA_TIPO] == tipo]
        contagem = tabela[COLUNA_ROTULO].value_counts().sort_index()
        return {str(palavra): int(qtd) for palavra, qtd in contagem.items()}

    def exemplos_do_sinal(self, rotulo: str, tipo: str = TIPO_SINAL) -> np.ndarray:
        """Devolve a forma das mãos de um sinal, na ordem em que foi gravado.

        Quadros sem nenhuma mão ficam de fora.

        Returns:
            Matriz (n, 126); com ``n = 0`` se o sinal não existir.
        """
        tabela = self._ler_tabela()
        linhas = tabela[
            (tabela[COLUNA_ROTULO] == rotulo) & (tabela[COLUNA_TIPO] == tipo)
        ]
        quadros = linhas[nomes_features_quadro()].to_numpy(dtype=np.float64)
        com_mao = maos_presentes(quadros).any(axis=1)
        return quadros[com_mao][:, IDX_MAOS]


def remover_arquivo_se_existir(caminho: Path) -> bool:
    """Apaga um arquivo se ele existir.

    Returns:
        ``True`` se apagou, ``False`` se não havia o que apagar.
    """
    try:
        os.remove(caminho)
        return True
    except FileNotFoundError:
        return False
