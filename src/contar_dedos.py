"""Kara Sygna — contar dedos levantados (demonstração).

Melhoria: o critério agora é a DISTÂNCIA até o punho, não o eixo Y.
Funciona com a mão inclinada/girada e não depende de qual mão é.
"""

import math

import cv2
import mediapipe as mp

# nome: (ponta, articulação intermediária)
DEDOS = {
    "indicador": (8, 6),
    "medio": (12, 10),
    "anelar": (16, 14),
    "mindinho": (20, 18),
}


def _dist(pontos, a, b):
    return math.hypot(pontos[a].x - pontos[b].x, pontos[a].y - pontos[b].y)


def dedos_levantados(landmarks_da_mao):
    p = landmarks_da_mao.landmark
    levantados = [
        nome for nome, (ponta, junta) in DEDOS.items()
        if _dist(p, ponta, 0) > _dist(p, junta, 0) * 1.1
    ]
    # Polegar: a ponta fica mais longe da base do mindinho que a junta IP.
    if _dist(p, 4, 17) > _dist(p, 3, 17) * 1.05:
        levantados.append("polegar")
    return levantados


def main():
    mp_hands = mp.solutions.hands
    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("Não foi possível acessar a câmera.")
        return
    print("Mostre a mão. Pressione 'q' para sair.")

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = captura.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, lm, mp_hands.HAND_CONNECTIONS)
                    lev = dedos_levantados(lm)
                    cv2.putText(frame, f"Dedos: {len(lev)} {lev}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("kara-sygna - Contar Dedos", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
