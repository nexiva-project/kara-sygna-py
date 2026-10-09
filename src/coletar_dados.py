"""
Kara Sygna — coletar exemplos de sinais pelo terminal (alternativa à interface).

Como usar:
    python coletar_dados.py

Para cada sinal: digite o nome, espere a contagem e faça o sinal. Cada
gravação é uma RODADA. Grave pelo menos 3 rodadas por sinal, mudando
posição, distância, inclinação e luz entre elas — é isso que torna o
modelo confiável (e permite avaliá-lo de forma honesta).

Dica: grave também um sinal "neutro" (mão relaxada, em movimento) para a
IA aprender a dizer "isto não é um sinal".

Se o CSV existente for do formato antigo, ele é movido para um backup.
"""

import os
import time

import cv2
import mediapipe as mp

from dados import ARQUIVO_CSV, anexar_linhas, garantir_csv, novo_id_rodada
from utils import extrair_features

AMOSTRAS_POR_RODADA = 80
INTERVALO_ENTRE_AMOSTRAS = 0.10   # segundos (evita quadros quase idênticos)
SEGUNDOS_CONTAGEM = 3
JANELA = "kara-sygna - Coleta de Dados"


def gravar_rodada(captura, hands, face, rotulo):
    rodada = novo_id_rodada()
    inicio = time.monotonic() + SEGUNDOS_CONTAGEM
    ultima = 0.0
    gravadas = 0

    while gravadas < AMOSTRAS_POR_RODADA:
        ret, frame = captura.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        altura, largura = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        agora = time.monotonic()

        if agora < inicio:
            texto = f"Prepare '{rotulo}': {int(inicio - agora) + 1}"
            cor = (0, 200, 255)
        else:
            res_maos = hands.process(rgb)
            if res_maos.multi_hand_landmarks:
                for lm in res_maos.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
                if agora - ultima >= INTERVALO_ENTRE_AMOSTRAS:
                    vetor = extrair_features(res_maos, face.process(rgb), largura, altura)
                    anexar_linhas(ARQUIVO_CSV, rotulo, rodada, [vetor])
                    gravadas += 1
                    ultima = agora
                texto = f"Gravando '{rotulo}': {gravadas}/{AMOSTRAS_POR_RODADA}"
                cor = (0, 0, 255)
            else:
                texto = "Mostre a mao para a camera"
                cor = (0, 165, 255)

        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        cv2.imshow(JANELA, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def main():
    backup = garantir_csv(ARQUIVO_CSV)
    if backup:
        print(f"CSV antigo (formato incompatível) salvo como: {backup}")

    mp_hands, mp_face = mp.solutions.hands, mp.solutions.face_mesh
    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("Não foi possível acessar a câmera.")
        return

    print(f"Dados serão salvos em: {os.path.abspath(ARQUIVO_CSV)}")
    print("Durante a gravação, 'q' na janela interrompe.\n")

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands, \
         mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.5) as face:
        while True:
            rotulo = input("Nome do sinal (Enter vazio para sair): ").strip().lower()
            if not rotulo:
                break
            if not gravar_rodada(captura, hands, face, rotulo):
                break
            print(f"Rodada de '{rotulo}' concluída. Repita variando posição/distância.\n")

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
