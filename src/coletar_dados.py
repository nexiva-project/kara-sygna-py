"""
kara-sygna-py — Passo 5 (v1): fix ‘bug’.
kara-sygna-py — Passo 5 (v2): coletar dados para sinais mais específicos.

Diferença da v1:
    Agora usamos `extrair_landmarks_normalizados` (de src/utils.py) em
    vez das coordenadas absolutas. Isso torna o sinal reconhecível
    mesmo quando a mão muda de posição na tela ou de distância da
    câmera.

    IMPORTANTE: se você já tinha um dados_sinais.csv gravado com a
    versão antiga deste script, apague-o antes de rodar esta versão
    (os formatos não são compatíveis) e colete os sinais de novo.

Como usar: igual à v1 — veja as instruções no README.md.
"""

import csv
import os

import cv2
import mediapipe as mp

from utils import montar_vetor_duas_maos, FEATURES_POR_MAO

ARQUIVO_CSV = "dados_sinais.csv"
AMOSTRAS_POR_RODADA = 60


def garantir_cabecalho_csv(caminho):
    if os.path.exists(caminho):
        return

    cabecalho = ["rotulo"]
    for lado in ("esq", "dir"):
        for i in range(21):
            cabecalho.extend([f"{lado}_x{i}", f"{lado}_y{i}", f"{lado}_z{i}"])

    with open(caminho, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(cabecalho)


def main():
    mp_hands = mp.solutions.hands
    mp_desenho = mp.solutions.drawing_utils

    # max_num_hands=2: agora detectamos até duas mãos ao mesmo tempo,
    # para sinais que usam as duas mãos (ex: "carro").
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("Não foi possível acessar a câmera.")
        return

    garantir_cabecalho_csv(ARQUIVO_CSV)

    print(f"Dados serão salvos em: {os.path.abspath(ARQUIVO_CSV)}")
    print("Pressione 'q' na janela de vídeo a qualquer momento para encerrar.\n")

    rotulo_atual = None
    amostras_restantes = 0

    while True:
        ret, frame = captura.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(frame_rgb)

        if amostras_restantes == 0:
            cv2.imshow("kara-sygna-py - Passo 6: Coleta de Dados", frame)
            cv2.waitKey(1)

            entrada = input(
                "Digite o rótulo do sinal a gravar (ou Enter vazio para sair): "
            ).strip()

            if entrada == "":
                break

            rotulo_atual = entrada
            amostras_restantes = AMOSTRAS_POR_RODADA
            print(f"Gravando {AMOSTRAS_POR_RODADA} amostras para '{rotulo_atual}'"
                  f"... varie um pouco a posição da(s) mão(s) durante a gravação.")
            continue

        if resultado.multi_hand_landmarks:
            for landmarks_da_mao in resultado.multi_hand_landmarks:
                mp_desenho.draw_landmarks(
                    frame, landmarks_da_mao, mp_hands.HAND_CONNECTIONS
                )

            # Um vetor por FRAME (não por mão!), já com as duas mãos
            # combinadas — isso corrige o bug do contador ficar negativo.
            vetor = montar_vetor_duas_maos(resultado)

            with open(ARQUIVO_CSV, "a", newline="", encoding="utf-8") as arquivo:
                escritor = csv.writer(arquivo)
                escritor.writerow([rotulo_atual] + vetor)

            amostras_restantes -= 1

        texto = f"Gravando '{rotulo_atual}': faltam {amostras_restantes}"
        cv2.putText(
            frame, texto, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
        cv2.imshow("kara-sygna-py - Passo 6: Coleta de Dados", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"\nDados salvos em: {os.path.abspath(ARQUIVO_CSV)}")


if __name__ == "__main__":
    main()