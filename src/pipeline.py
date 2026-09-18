"""
kara-sygna-py — pipeline.py: lógica reutilizável por trás do painel (app.py).

Este módulo junta, como FUNÇÕES chamáveis (em vez de scripts separados
que você roda um de cada vez), tudo que já construímos:
    - coletar_amostras(): grava landmarks por um tempo (padrão 60s)
    - aumentar_dados(): espelha sinais de mão única
    - treinar_modelo(): treina e salva o classificador
    - reconhecer_sinais(): reconhecimento em tempo real (com suavização)

O app.py (painel gráfico) chama essas funções. Você também pode
continuar chamando cada uma isoladamente, se preferir rodar por script.
"""

import csv
import os
import time
from collections import deque, Counter

import cv2
import joblib
import mediapipe as mp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import (
    montar_vetor_duas_maos,
    FEATURES_POR_MAO,
    eh_vetor_zerado,
    espelhar_mao,
)

ARQUIVO_CSV = "dados_sinais.csv"
ARQUIVO_MODELO = "modelo_sinais.pkl"
DURACAO_PADRAO_SEGUNDOS = 10
CONFIANCA_MINIMA = 0.7
TAMANHO_JANELA = 10
MINIMO_DE_VOTOS = 6


def _garantir_cabecalho_csv(caminho):
    if os.path.exists(caminho):
        return
    cabecalho = ["rotulo"]
    for lado in ("esq", "dir"):
        for i in range(21):
            cabecalho.extend([f"{lado}_x{i}", f"{lado}_y{i}", f"{lado}_z{i}"])
    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        csv.writer(arquivo).writerow(cabecalho)


def coletar_amostras(
    rotulo,
    duracao_segundos=DURACAO_PADRAO_SEGUNDOS,
    arquivo_csv=ARQUIVO_CSV,
    atualizar_status=None,
):
    """
    Abre a câmera e grava landmarks para o 'rotulo' informado, durante
    'duracao_segundos'. Faz o espelhamento automático quando só uma
    mão aparece. Retorna quantas amostras (linhas) foram salvas.

    'atualizar_status', se fornecido, é uma função chamada a cada
    frame com uma string de status (útil para o painel gráfico
    atualizar um rótulo de progresso).
    """
    mp_hands = mp.solutions.hands
    mp_desenho = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        raise RuntimeError("Não foi possível acessar a câmera.")

    _garantir_cabecalho_csv(arquivo_csv)

    total_salvas = 0
    inicio = time.time()
    nome_janela = "kara-sygna - Gravando"

    try:
        while True:
            tempo_restante = duracao_segundos - (time.time() - inicio)
            if tempo_restante <= 0:
                break

            ret, frame = captura.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = hands.process(frame_rgb)

            if resultado.multi_hand_landmarks:
                for landmarks_da_mao in resultado.multi_hand_landmarks:
                    mp_desenho.draw_landmarks(
                        frame, landmarks_da_mao, mp_hands.HAND_CONNECTIONS
                    )

                vetor = montar_vetor_duas_maos(resultado)
                vetor_esq = vetor[:FEATURES_POR_MAO]
                vetor_dir = vetor[FEATURES_POR_MAO:]

                linhas = [[rotulo] + vetor]

                so_esquerda = not eh_vetor_zerado(vetor_esq) and eh_vetor_zerado(vetor_dir)
                so_direita = not eh_vetor_zerado(vetor_dir) and eh_vetor_zerado(vetor_esq)

                if so_esquerda:
                    espelhado = [0.0] * FEATURES_POR_MAO + espelhar_mao(vetor_esq)
                    linhas.append([rotulo] + espelhado)
                elif so_direita:
                    espelhado = espelhar_mao(vetor_dir) + [0.0] * FEATURES_POR_MAO
                    linhas.append([rotulo] + espelhado)

                with open(arquivo_csv, "a", newline="", encoding="utf-8") as arquivo:
                    csv.writer(arquivo).writerows(linhas)

                total_salvas += len(linhas)

            texto = f"Gravando '{rotulo}': {tempo_restante:0.0f}s restantes | amostras: {total_salvas}"
            cv2.putText(
                frame, texto, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )
            cv2.imshow(nome_janela, frame)

            if atualizar_status is not None:
                atualizar_status(texto)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        captura.release()
        cv2.destroyWindow(nome_janela)
        hands.close()

    return total_salvas


def aumentar_dados(arquivo_csv=ARQUIVO_CSV):
    """
    Para cada amostra com só uma mão preenchida, adiciona uma versão
    espelhada (mesma forma, mão oposta). Retorna quantas linhas novas
    foram adicionadas.

    Só precisa rodar sobre dados coletados ANTES da versão atual do
    coletar_amostras() (que já espelha automaticamente). É seguro
    rodar de novo mesmo assim: linhas de duas mãos são ignoradas, e
    linhas de uma mão só geram no máximo uma espelhada cada.
    """
    if not os.path.exists(arquivo_csv):
        return 0

    with open(arquivo_csv, "r", newline="", encoding="utf-8") as arquivo:
        leitor = csv.reader(arquivo)
        next(leitor)  # cabeçalho
        linhas_originais = list(leitor)

    novas_linhas = []
    for linha in linhas_originais:
        rotulo = linha[0]
        valores = [float(v) for v in linha[1:]]
        vetor_esq = valores[:FEATURES_POR_MAO]
        vetor_dir = valores[FEATURES_POR_MAO:]

        so_esquerda = not eh_vetor_zerado(vetor_esq) and eh_vetor_zerado(vetor_dir)
        so_direita = not eh_vetor_zerado(vetor_dir) and eh_vetor_zerado(vetor_esq)

        if so_esquerda:
            espelhado = [0.0] * FEATURES_POR_MAO + espelhar_mao(vetor_esq)
            novas_linhas.append([rotulo] + espelhado)
        elif so_direita:
            espelhado = espelhar_mao(vetor_dir) + [0.0] * FEATURES_POR_MAO
            novas_linhas.append([rotulo] + espelhado)

    if novas_linhas:
        with open(arquivo_csv, "a", newline="", encoding="utf-8") as arquivo:
            csv.writer(arquivo).writerows(novas_linhas)

    return len(novas_linhas)


def treinar_modelo(arquivo_csv=ARQUIVO_CSV, arquivo_modelo=ARQUIVO_MODELO):
    """
    Treina o classificador Random Forest a partir do CSV e salva o
    modelo. Retorna um dicionário com a acurácia e a quantidade de
    amostras usadas, útil para mostrar no painel gráfico.
    """
    dados = pd.read_csv(arquivo_csv)
    X = dados.drop(columns=["rotulo"])
    y = dados["rotulo"]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_treino, y_treino)

    predicoes = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, predicoes)

    joblib.dump(modelo, arquivo_modelo)

    return {
        "acuracia": acuracia,
        "total_amostras": len(dados),
        "sinais": sorted(y.unique().tolist()),
    }


def reconhecer_sinais(arquivo_modelo=ARQUIVO_MODELO):
    """
    Abre a câmera e mostra o sinal reconhecido em tempo real, com
    suavização temporal (só mostra um sinal quando ele "vence a
    votação" nos últimos frames). Roda até 'q' ser pressionado ou a
    janela ser fechada.
    """
    modelo = joblib.load(arquivo_modelo)

    mp_hands = mp.solutions.hands
    mp_desenho = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        raise RuntimeError("Não foi possível acessar a câmera.")

    historico_previsoes = deque(maxlen=TAMANHO_JANELA)
    nome_janela = "kara-sygna - Reconhecimento em Tempo Real"

    try:
        while True:
            ret, frame = captura.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resultado = hands.process(frame_rgb)

            texto = "Nenhuma mao detectada"
            cor = (0, 0, 255)

            if resultado.multi_hand_landmarks:
                for landmarks_da_mao in resultado.multi_hand_landmarks:
                    mp_desenho.draw_landmarks(
                        frame, landmarks_da_mao, mp_hands.HAND_CONNECTIONS
                    )

                vetor = montar_vetor_duas_maos(resultado)
                entrada = pd.DataFrame([vetor], columns=modelo.feature_names_in_)

                probabilidades = modelo.predict_proba(entrada)[0]
                indice_melhor = probabilidades.argmax()
                sinal_previsto = modelo.classes_[indice_melhor]
                confianca = probabilidades[indice_melhor]

                historico_previsoes.append(
                    sinal_previsto if confianca >= CONFIANCA_MINIMA else None
                )

                votos = Counter(v for v in historico_previsoes if v is not None)
                if votos:
                    sinal_mais_votado, quantidade_votos = votos.most_common(1)[0]
                else:
                    sinal_mais_votado, quantidade_votos = None, 0

                if sinal_mais_votado is not None and quantidade_votos >= MINIMO_DE_VOTOS:
                    texto = f"Sinal: {sinal_mais_votado} ({quantidade_votos}/{TAMANHO_JANELA})"
                    cor = (0, 255, 0)
                else:
                    texto = f"Incerto (frame atual: {sinal_previsto}, {confianca:.0%})"
                    cor = (0, 165, 255)
            else:
                historico_previsoes.clear()

            cv2.putText(
                frame, texto, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2,
            )
            cv2.imshow(nome_janela, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            # Fecha também se o usuário clicar no X da janela.
            if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        captura.release()
        cv2.destroyWindow(nome_janela)
        hands.close()