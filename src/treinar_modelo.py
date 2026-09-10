"""
kara-sygna-py — Passo 6: treinar um classificador simples.

Objetivo deste script:
    Ler o CSV gerado no Passo 6 (dados_sinais.csv), treinar um
    classificador de Random Forest para reconhecer cada sinal a
    partir das coordenadas dos 21 landmarks, e salvar o modelo
    treinado em um arquivo (modelo_sinais.pkl) para usarmos depois
    em tempo real no Passo 7.

Como rodar:
    python src/treinar_modelo.py
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

ARQUIVO_CSV = "dados_sinais.csv"
ARQUIVO_MODELO = "modelo_sinais.pkl"


def main():
    print(f"Lendo dados de: {ARQUIVO_CSV}")
    dados = pd.read_csv(ARQUIVO_CSV)

    print(f"Total de amostras: {len(dados)}")
    print("Amostras por sinal:")
    print(dados["rotulo"].value_counts())

    # X = todas as colunas de coordenadas (tudo, menos o rótulo).
    # y = a coluna com o nome do sinal (o que queremos prever).
    X = dados.drop(columns=["rotulo"])
    y = dados["rotulo"]

    # Separamos uma parte dos dados (20%) só para TESTAR o modelo
    # depois, com exemplos que ele nunca viu durante o treino.
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTreinando o classificador (Random Forest)...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_treino, y_treino)

    print("\nAvaliando o modelo com dados de teste:")
    predicoes = modelo.predict(X_teste)
    print(classification_report(y_teste, predicoes))

    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"Modelo treinado e salvo em: {ARQUIVO_MODELO}")


if __name__ == "__main__":
    main()