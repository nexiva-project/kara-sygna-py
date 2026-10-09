"""Leitura/escrita do CSV de exemplos (um arquivo, uma linha por quadro)."""

import csv
import os
import time
from datetime import datetime

from utils import nomes_features

ARQUIVO_CSV = "dados_sinais.csv"


def cabecalho():
    # "rodada" = uma sessão de gravação. Serve para avaliar o modelo de
    # forma honesta (exemplos da mesma rodada nunca ficam em treino e teste).
    return ["rotulo", "rodada"] + nomes_features()


def garantir_csv(caminho=ARQUIVO_CSV):
    """Cria o CSV se preciso. Se existir em formato antigo, move para um
    backup e devolve o nome do backup (senão devolve None)."""
    if not os.path.exists(caminho) or os.path.getsize(caminho) == 0:
        _escrever_cabecalho(caminho)
        return None

    with open(caminho, newline="", encoding="utf-8") as f:
        primeira = next(csv.reader(f), [])
    if primeira == cabecalho():
        return None

    base, ext = os.path.splitext(caminho)
    backup = f"{base}_backup_{datetime.now():%Y%m%d_%H%M%S}{ext}"
    os.replace(caminho, backup)
    _escrever_cabecalho(caminho)
    return backup


def _escrever_cabecalho(caminho):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(cabecalho())


def novo_id_rodada():
    return int(time.time())


def anexar_linhas(caminho, rotulo, rodada, vetores):
    with open(caminho, "a", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        for vetor in vetores:
            escritor.writerow([rotulo, rodada] + [round(float(x), 6) for x in vetor])
