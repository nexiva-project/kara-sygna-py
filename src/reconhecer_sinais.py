"""
kara-sygna-py — Passo 7: rodar o reconhecimento em tempo real.

Objetivo deste script:
    Juntar tudo que construímos: captura de vídeo (Passo 1),
    detecção de mão (Passo 2), extração de landmarks (Passo 3) e o
    classificador treinado (Passo 6), para mostrar o sinal reconhecido
    ao vivo, na tela.

Pré-requisito:
    Ter rodado antes o src/treinar_modelo.py, gerando o arquivo
    modelo_sinais.pkl.

Como rodar:
    python src/reconhecer_sinais.py
"""

from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import pandas as pd

from utils import montar_vetor_duas_maos

ARQUIVO_MODELO = "modelo_sinais.pkl"

# Só mostramos o sinal na tela quando o modelo está bem confiante,
# para evitar "piscar" um sinal errado por um instante.
CONFIANCA_MINIMA = 0.7

# --- Suavização temporal ---
# Em vez de confiar 100% na previsão de um único frame (que pode ser
# afetado por ruído momentâneo da câmera ou um ângulo levemente
# diferente da mão), guardamos as últimas N previsões e mostramos a
# que apareceu com mais frequência entre elas. Isso reduz bastante a
# confusão entre sinais parecidos, como "3" e "4".
TAMANHO_JANELA = 10  # quantos frames recentes considerar
MINIMO_DE_VOTOS = 6  # de quantos desses frames precisam concordar


def main():
    print(f"Carregando modelo de: {ARQUIVO_MODELO}")
    modelo = joblib.load(ARQUIVO_MODELO)

    mp_hands = mp.solutions.hands
    mp_desenho = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("Não foi possível acessar a câmera.")
        return

    # Guarda as últimas previsões (só as que passaram do CONFIANCA_MINIMA).
    historico_previsoes = deque(maxlen=TAMANHO_JANELA)

    print("Câmera aberta! Mostre um sinal treinado. Pressione 'q' para sair.")

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

            # Uma predição por FRAME, com o vetor das duas mãos juntas
            # (a mesma lógica usada na coleta de dados).
            vetor = montar_vetor_duas_maos(resultado)

            # Empacotamos o vetor num DataFrame com os MESMOS nomes de
            # coluna usados no treino (evita o warning do scikit-learn
            # e garante que a ordem das colunas bate certinho).
            entrada = pd.DataFrame([vetor], columns=modelo.feature_names_in_)

            # predict_proba retorna a probabilidade do modelo para
            # cada sinal que ele conhece; pegamos a maior delas.
            probabilidades = modelo.predict_proba(entrada)[0]
            indice_melhor = probabilidades.argmax()
            sinal_previsto = modelo.classes_[indice_melhor]
            confianca = probabilidades[indice_melhor]

            if confianca >= CONFIANCA_MINIMA:
                historico_previsoes.append(sinal_previsto)
            else:
                # Uma previsão de baixa confiança não "vota" em nada,
                # mas também não é adicionada como ruído no histórico.
                historico_previsoes.append(None)

            # Conta qual sinal apareceu mais vezes na janela recente
            # (ignorando os None, que representam frames incertos).
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
            # Sem mão no frame: limpamos o histórico para não "carregar"
            # uma previsão antiga quando a mão voltar a aparecer.
            historico_previsoes.clear()

        cv2.putText(
            frame, texto, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2,
        )
        cv2.imshow("kara-sygna-py - Passo 8: Reconhecimento em Tempo Real", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()