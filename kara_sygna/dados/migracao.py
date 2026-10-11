"""Atualização de arquivos de versões anteriores do projeto.

* O CSV antigo não tem a coluna ``rodada``. Ela é inferida e o CSV original é
  guardado em um backup (nada é perdido).
* O CSV e o modelo antigos ficavam na pasta do projeto (ou em ``src/``); agora
  ficam em ``armazenamento/``. Eles são COPIADOS, nunca movidos.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from kara_sygna import config
from kara_sygna.dados.repositorio import COLUNA_RODADA, COLUNA_ROTULO, cabecalho
from kara_sygna.visao.caracteristicas import nomes_features

logger = logging.getLogger(__name__)


def inferir_rodadas(
    rotulos: np.ndarray, blocos_por_sinal: int = config.RODADAS_MINIMAS_INFERIDAS
) -> np.ndarray:
    """Cria uma coluna de rodadas para um CSV que não tinha.

    Cada sinal tem suas linhas (em ordem de gravação) divididas em blocos
    contíguos; cada bloco vira uma "rodada". Não é tão bom quanto rodadas reais
    (os blocos vizinhos são parecidos), mas impede que quadros quase idênticos
    caiam no treino e no teste ao mesmo tempo.

    Args:
        rotulos: Rótulo de cada linha, na ordem do arquivo.
        blocos_por_sinal: Em quantos blocos dividir cada sinal.

    Returns:
        Vetor de inteiros (mesmo tamanho) com o id da rodada de cada linha.
        Ids de sinais diferentes nunca se repetem.
    """
    rotulos = np.asarray(rotulos, dtype=str)
    rodadas = np.zeros(len(rotulos), dtype=np.int64)
    proximo_id = 1
    for rotulo in dict.fromkeys(rotulos.tolist()):
        posicoes = np.flatnonzero(rotulos == rotulo)
        blocos = max(1, min(blocos_por_sinal, len(posicoes) // 2))
        for numero, parte in enumerate(np.array_split(posicoes, blocos)):
            rodadas[parte] = proximo_id + numero
        proximo_id += blocos
    return rodadas


def _nome_do_backup(caminho: Path) -> Path:
    """Gera ``dados_sinais_backup_AAAAmmdd_HHMMSS.csv`` ao lado do original."""
    return caminho.with_name(
        f"{caminho.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}{caminho.suffix}"
    )


def migrar_csv_legado(caminho: Path) -> Path | None:
    """Acrescenta a coluna ``rodada`` a um CSV antigo.

    Args:
        caminho: CSV a verificar.

    Returns:
        Caminho do backup criado, ou ``None`` se nada precisou mudar (arquivo
        ausente, vazio ou já no formato novo).

    Raises:
        ValueError: Se o CSV tiver colunas diferentes das esperadas.
    """
    caminho = Path(caminho)
    if not caminho.exists() or caminho.stat().st_size == 0:
        return None

    tabela = pd.read_csv(caminho, dtype={COLUNA_ROTULO: str})
    if list(tabela.columns) == cabecalho():
        return None
    if list(tabela.columns) != [COLUNA_ROTULO] + nomes_features():
        raise ValueError(
            f"O CSV {caminho.name} tem colunas desconhecidas e não pôde ser convertido. "
            "Renomeie ou apague o arquivo para recomeçar."
        )

    backup = _nome_do_backup(caminho)
    shutil.copy2(caminho, backup)
    tabela = tabela.copy()  # desfragmenta antes de inserir a coluna
    tabela.insert(1, COLUNA_RODADA, inferir_rodadas(tabela[COLUNA_ROTULO].to_numpy()))
    stat = caminho.stat()
    tabela.to_csv(caminho, index=False, encoding="utf-8")
    # Mantém a data original: o modelo continua valendo (os dados não mudaram).
    os.utime(caminho, (stat.st_atime, stat.st_mtime))
    logger.info(
        "CSV convertido para o formato novo (coluna 'rodada'). Backup: %s", backup.name
    )
    return backup


def copiar_arquivos_legados(
    raiz: Path = config.RAIZ_PROJETO,
    destino: Path = config.PASTA_ARMAZENAMENTO,
) -> list[Path]:
    """Copia ``dados_sinais.csv`` e ``modelo_sinais.pkl`` antigos para ``armazenamento/``.

    Procura na raiz do projeto e em ``src/``. Só copia se o arquivo ainda não
    existir no destino, e preserva a data de modificação.

    Returns:
        Lista dos arquivos copiados (destinos).
    """
    destino.mkdir(parents=True, exist_ok=True)
    copiados: list[Path] = []
    for nome in (config.ARQUIVO_CSV.name, config.ARQUIVO_MODELO.name):
        novo = destino / nome
        if novo.exists():
            continue
        for pasta in (raiz, raiz / "src"):
            antigo = pasta / nome
            if antigo.is_file():
                shutil.copy2(antigo, novo)
                copiados.append(novo)
                logger.info("Arquivo antigo copiado: %s -> %s", antigo, novo)
                break
    return copiados


def migrar_arquivos_legados(
    raiz: Path = config.RAIZ_PROJETO,
    destino: Path = config.PASTA_ARMAZENAMENTO,
) -> None:
    """Faz toda a migração: copia os arquivos antigos e converte o CSV.

    Chamada uma vez na inicialização do programa. Se a conversão falhar, o
    erro é registrado no log e o programa continua (o CSV não é alterado).
    """
    copiar_arquivos_legados(raiz, destino)
    try:
        migrar_csv_legado(destino / config.ARQUIVO_CSV.name)
    except ValueError as erro:
        logger.warning("%s", erro)
