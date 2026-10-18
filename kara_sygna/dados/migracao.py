"""Atualização de arquivos de versões anteriores do projeto.

* O CSV e o modelo antigos ficavam na pasta do projeto (ou em ``src/``); agora
  ficam em ``armazenamento/``. Eles são COPIADOS, nunca movidos.
* CSVs de formatos antigos (só a forma das mãos, sem rosto nem sequência de
  quadros) não servem para a IA atual e não podem ser convertidos: faltam os
  dados do rosto e os quadros que foram pulados. Eles são guardados em um
  backup, junto com o modelo antigo, e o programa começa um CSV novo. Nada é
  apagado.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

from kara_sygna import config
from kara_sygna.dados.repositorio import cabecalho

logger = logging.getLogger(__name__)


def _nome_do_backup(caminho: Path, momento: datetime) -> Path:
    """Gera ``nome_formato_antigo_AAAAmmdd_HHMMSS.ext`` ao lado do original."""
    return caminho.with_name(
        f"{caminho.stem}_formato_antigo_{momento:%Y%m%d_%H%M%S}{caminho.suffix}"
    )


def _ler_cabecalho(caminho: Path) -> list[str]:
    with open(caminho, newline="", encoding="utf-8") as arquivo:
        return next(csv.reader(arquivo), [])


def guardar_formato_antigo(
    caminho_csv: Path = config.ARQUIVO_CSV,
    caminho_modelo: Path = config.ARQUIVO_MODELO,
) -> Path | None:
    """Move um CSV de formato antigo (e o modelo treinado com ele) para backups.

    Args:
        caminho_csv: CSV a verificar.
        caminho_modelo: Modelo ``.pkl``; o resumo ``.json`` e a IA de
            pré-sinais (``*_pre.pkl``) ao lado também vão.

    Returns:
        Caminho do backup do CSV, ou ``None`` se nada precisou mudar (arquivo
        ausente, vazio ou já no formato atual).
    """
    caminho_csv, caminho_modelo = Path(caminho_csv), Path(caminho_modelo)
    if not caminho_csv.exists() or caminho_csv.stat().st_size == 0:
        return None
    if _ler_cabecalho(caminho_csv) == cabecalho():
        return None

    momento = datetime.now()
    backup = _nome_do_backup(caminho_csv, momento)
    os.replace(caminho_csv, backup)
    pre = caminho_modelo.with_name(f"{caminho_modelo.stem}_pre{caminho_modelo.suffix}")
    for arquivo in (caminho_modelo, caminho_modelo.with_suffix(".json"), pre):
        if arquivo.exists():
            os.replace(arquivo, _nome_do_backup(arquivo, momento))

    logger.warning(
        "Os sinais gravados estavam num formato antigo, que a IA atual não usa "
        "(agora ela também olha o rosto e o movimento). Eles foram guardados em "
        "%s, junto com o modelo antigo. Grave os sinais de novo e treine a IA.",
        backup.name,
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
) -> Path | None:
    """Faz toda a migração: copia os arquivos antigos e guarda formatos antigos.

    Chamada uma vez na inicialização do programa.

    Returns:
        Caminho do backup do CSV, se um CSV de formato antigo foi guardado.
    """
    copiar_arquivos_legados(raiz, destino)
    try:
        return guardar_formato_antigo(
            destino / config.ARQUIVO_CSV.name, destino / config.ARQUIVO_MODELO.name
        )
    except OSError as erro:
        logger.warning("Não foi possível guardar o CSV antigo: %s", erro)
        return None
