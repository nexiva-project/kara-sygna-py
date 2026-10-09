"""Kara Sygna — treino do classificador (com avaliação honesta).

    python treinar_modelo.py

O que mudou:
* Avaliação por RODADA (validação cruzada em grupos): exemplos da mesma
  sessão de gravação nunca aparecem no treino e no teste ao mesmo tempo.
  A acurácia mostrada passa a refletir o uso real.
* Aumento de dados só no treino de cada dobra (sem vazamento).
* class_weight balanceado: sinais com menos exemplos não são ignorados.
* A mesma função ``treinar`` é usada pela interface.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold

from aumentar_dados import expandir_treino
from dados import ARQUIVO_CSV
from utils import nomes_features

ARQUIVO_MODELO = "modelo_sinais.pkl"


def carregar_dados(caminho):
    df = pd.read_csv(caminho, dtype={"rotulo": str})
    nomes = nomes_features()
    faltando = [c for c in ["rotulo", "rodada"] + nomes if c not in df.columns]
    if faltando:
        raise ValueError(
            "O CSV está num formato antigo (faltam colunas como "
            f"'{faltando[0]}'). Grave os sinais novamente."
        )
    return (
        df[nomes].to_numpy(dtype=np.float64),
        df["rotulo"].to_numpy(),
        df["rodada"].to_numpy(),
    )


def _novo_modelo():
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )


def _variantes_para(n_linhas):
    # Mantém o treino em tamanho razoável mesmo com muitos exemplos.
    return int(max(2, min(8, 40000 // max(1, n_linhas))))


def _ajustar(X, y, nomes):
    n_var = _variantes_para(len(X))
    X_exp, y_exp = expandir_treino(X, y, n_variantes=n_var)
    modelo = _novo_modelo()
    modelo.fit(pd.DataFrame(X_exp, columns=nomes), y_exp)
    return modelo


def avaliar(X, y, grupos, nomes, log=print):
    """Validação cruzada por rodada. Devolve (acurácia, relatório) ou (None, aviso)."""
    classes, _ = np.unique(y, return_counts=True)
    rodadas_por_classe = [len(np.unique(grupos[y == c])) for c in classes]
    n_splits = min(5, min(rodadas_por_classe))
    if n_splits < 2:
        return None, (
            "Avaliação pulada: cada sinal precisa de pelo menos 2 rodadas de "
            "gravação (clique em COMEÇAR A ENSINAR mais de uma vez por sinal, "
            "de preferência variando posição, distância e luz)."
        )

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    previstos = np.empty(len(y), dtype=object)
    for i, (idx_treino, idx_teste) in enumerate(cv.split(X, y, grupos), 1):
        log(f"  validação {i}/{n_splits}...")
        modelo = _ajustar(X[idx_treino], y[idx_treino], nomes)
        previstos[idx_teste] = modelo.predict(
            pd.DataFrame(X[idx_teste], columns=nomes)
        )
    acuracia = accuracy_score(y, previstos)
    return acuracia, classification_report(y, previstos.astype(str), zero_division=0)


def treinar(caminho_csv=ARQUIVO_CSV, caminho_modelo=ARQUIVO_MODELO, log=print):
    X, y, grupos = carregar_dados(caminho_csv)
    if len(X) == 0:
        raise ValueError("O CSV está vazio. Ensine pelo menos um sinal.")

    nomes = nomes_features()
    classes = sorted(set(y))
    log(f"Amostras: {len(X)} | sinais: {len(classes)} | rodadas: {len(set(grupos))}")
    for c in classes:
        log(f"  {c}: {(y == c).sum()} quadros em {len(set(grupos[y == c]))} rodada(s)")

    if len(classes) < 2:
        raise ValueError(
            "Só há 1 sinal gravado. Grave pelo menos 2 sinais — de preferência "
            "também um sinal 'neutro' (mão relaxada) para a IA saber quando "
            "NÃO há sinal."
        )

    log("Avaliando (por rodada)...")
    acuracia, relatorio = avaliar(X, y, grupos, nomes, log)
    log(relatorio if acuracia is not None else f"AVISO: {relatorio}")

    log("Treinando o modelo final com todos os dados...")
    modelo = _ajustar(X, y, nomes)
    joblib.dump(modelo, caminho_modelo)

    resumo = {
        "amostras": int(len(X)),
        "classes": [str(c) for c in classes],
        "rodadas": int(len(set(grupos))),
        "acuracia_validacao": None if acuracia is None else round(float(acuracia), 4),
    }
    with open(os.path.splitext(caminho_modelo)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    log(f"Modelo salvo em: {caminho_modelo}")

    resumo["relatorio"] = relatorio
    return resumo


def main():
    if not os.path.exists(ARQUIVO_CSV):
        print(f"{ARQUIVO_CSV} não encontrado. Rode coletar_dados.py (ou a interface) primeiro.")
        return
    try:
        treinar()
    except ValueError as erro:
        print(erro)


if __name__ == "__main__":
    main()
