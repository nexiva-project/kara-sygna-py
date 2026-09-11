"""
kara-sygna-py — utilitário: aumentar dados já coletados com espelhamento.

Objetivo deste script:
    Corrigir dados coletados ANTES da atualização do coletar_dados.py
    que já gera a versão espelhada automaticamente. Para cada amostra
    do dados_sinais.csv que tem só UMA mão preenchida (a outra toda
    zero), gera uma linha nova com a versão espelhada (mesma forma,
    mão oposta) e adiciona ao final do arquivo.

    Depois de rodar este script, retreine o modelo com:
        python src/treinar_modelo.py

Como rodar (uma vez só, sobre os dados que você já tem):
    python src/aumentar_dados.py
"""

import csv
import os

from utils import FEATURES_POR_MAO, eh_vetor_zerado, espelhar_mao

ARQUIVO_CSV = "dados_sinais.csv"


def main():
    if not os.path.exists(ARQUIVO_CSV):
        print(f"Não encontrei {ARQUIVO_CSV}. Rode a partir da raiz do projeto.")
        return

    with open(ARQUIVO_CSV, "r", newline="", encoding="utf-8") as arquivo:
        leitor = csv.reader(arquivo)
        cabecalho = next(leitor)
        linhas_originais = list(leitor)

    novas_linhas = []
    for linha in linhas_originais:
        rotulo = linha[0]
        valores = [float(v) for v in linha[1:]]
        vetor_esquerda = valores[:FEATURES_POR_MAO]
        vetor_direita = valores[FEATURES_POR_MAO:]

        so_esquerda = (
            not eh_vetor_zerado(vetor_esquerda) and eh_vetor_zerado(vetor_direita)
        )
        so_direita = (
            not eh_vetor_zerado(vetor_direita) and eh_vetor_zerado(vetor_esquerda)
        )

        if so_esquerda:
            espelhado = [0.0] * FEATURES_POR_MAO + espelhar_mao(vetor_esquerda)
            novas_linhas.append([rotulo] + espelhado)
        elif so_direita:
            espelhado = espelhar_mao(vetor_direita) + [0.0] * FEATURES_POR_MAO
            novas_linhas.append([rotulo] + espelhado)
        # Se as duas mãos estavam presentes (sinal de duas mãos),
        # não espelhamos — a distinção esquerda/direita importa nesse caso.

    if not novas_linhas:
        print("Nenhuma amostra de mão única encontrada para espelhar.")
        return

    with open(ARQUIVO_CSV, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerows(novas_linhas)

    print(f"Adicionadas {len(novas_linhas)} amostras espelhadas ao {ARQUIVO_CSV}.")
    print("Agora rode: python src/treinar_modelo.py")


if __name__ == "__main__":
    main()