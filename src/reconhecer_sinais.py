"""
Kara Sygna — reconhecimento em tempo real pelo terminal/janela OpenCV.

Pré-requisito: ter treinado (treinar_modelo.py ou botão TREINAR IA).
    python reconhecer_sinais.py
"""

import joblib
import cv2
import mediapipe as mp

from reconhecimento import IGNORAR, Estabilizador, prever
from treinar_modelo import ARQUIVO_MODELO
from utils import extrair_features, nomes_features


def main():
    modelo = joblib.load(ARQUIVO_MODELO)
    if list(modelo.feature_names_in_) != nomes_features():
        print("Este modelo é de uma versão antiga. Grave os sinais de novo e treine.")
        return

    mp_hands, mp_face = mp.solutions.hands, mp.solutions.face_mesh
    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("Não foi possível acessar a câmera.")
        return

    estabilizador = Estabilizador()
    frase = []
    print("Câmera aberta! Faça um sinal. 'q' sai, 'c' limpa o texto.")

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands, \
         mp_face.FaceMesh(max_num_faces=1, min_detection_confidence=0.7,
                          min_tracking_confidence=0.5) as face:
        while True:
            ret, frame = captura.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            altura, largura = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res_maos = hands.process(rgb)

            aceito, texto, cor = None, "Nenhuma mao detectada", (0, 0, 255)
            if res_maos.multi_hand_landmarks:
                for lm in res_maos.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, lm, mp_hands.HAND_CONNECTIONS)
                vetor = extrair_features(res_maos, face.process(rgb), largura, altura)
                sinal, confianca, _, ok = prever(modelo, vetor)
                if ok:
                    aceito, cor = sinal, (0, 255, 0)
                    texto = f"Sinal: {sinal} ({confianca:.0%})"
                else:
                    texto, cor = f"Incerto ({confianca:.0%})", (0, 165, 255)

            confirmado = estabilizador.atualizar(aceito)
            if confirmado and confirmado not in IGNORAR:
                frase.append(confirmado)

            cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)
            cv2.putText(frame, " ".join(frase[-8:]), (10, altura - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow("kara-sygna - Reconhecimento", frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("c"):
                frase.clear()

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
