import cv2
import mediapipe as mp

def main():
    # --- Configuração do MediaPipe ---
    mp_hands = mp.solutions.hands
    mp_desenho = mp.solutions.drawing_utils

    # max_num_hands=1: por enquanto detectamos só uma mão, para simplificar.
    # min_detection_confidence: quão confiante o modelo precisa estar para
    # considerar que "achou" uma mão (0.0 a 1.0).
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    captura = cv2.VideoCapture(0)

    if not captura.isOpened():
        print("Não foi possível acessar a câmera.")
        return

    print("Câmera aberta! Mostre sua mão para a câmera. Pressione 'q' para sair.")

    while True:
        ret, frame = captura.read()
        if not ret:
            print("Não foi possível ler o frame da câmera. Encerrando...")
            break

        frame = cv2.flip(frame, 1)

        # O MediaPipe espera imagens no formato RGB, mas o OpenCV
        # captura em BGR — por isso convertemos antes de processar.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(frame_rgb)

        # Se alguma mão foi detectada no frame...
        if resultado.multi_hand_landmarks:
            for landmarks_da_mao in resultado.multi_hand_landmarks:
                # Desenha os 21 pontos e as linhas conectando-os
                # (esqueleto da mão) diretamente sobre o frame original.
                mp_desenho.draw_landmarks(
                    frame,
                    landmarks_da_mao,
                    mp_hands.HAND_CONNECTIONS,
                )

        cv2.imshow("kara-sygna - Passo 2: Deteccao de Mao", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()