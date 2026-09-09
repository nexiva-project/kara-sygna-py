import cv2
import mediapipe as mp

# Nomes amigáveis para os pontos que mais vamos usar.
NOMES_DOS_PONTOS = {
    0: "pulso",
    4: "ponta_polegar",
    8: "ponta_indicador",
    12: "ponta_medio",
    16: "ponta_anelar",
    20: "ponta_mindinho",
}


def extrair_landmarks(landmarks_da_mao):
    """
    Recebe o objeto de landmarks que o MediaPipe devolve para UMA mão
    e retorna uma lista de dicionários, um por ponto, no formato:
        {"id": 8, "nome": "ponta_indicador", "x": 0.52, "y": 0.31, "z": -0.02}

    x e y são coordenadas normalizadas (de 0.0 a 1.0) relativas ao
    tamanho da imagem. z é uma estimativa de profundidade (quanto mais
    negativo, mais perto da câmera em relação ao pulso).
    """
    pontos = []
    for indice, ponto in enumerate(landmarks_da_mao.landmark):
        pontos.append({
            "id": indice,
            "nome": NOMES_DOS_PONTOS.get(indice, f"ponto_{indice}"),
            "x": round(ponto.x, 3),
            "y": round(ponto.y, 3),
            "z": round(ponto.z, 3),
        })
    return pontos


def main():
    mp_hands = mp.solutions.hands
    mp_desenho = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        max_num_hands=1,
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
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(frame_rgb)

        if resultado.multi_hand_landmarks:
            for landmarks_da_mao in resultado.multi_hand_landmarks:
                mp_desenho.draw_landmarks(
                    frame, landmarks_da_mao, mp_hands.HAND_CONNECTIONS
                )

                # Aqui está a parte nova: transformamos os landmarks
                # em uma lista de dicionários organizada.
                pontos = extrair_landmarks(landmarks_da_mao)

                # Mostra no terminal só a ponta do indicador, como exemplo.
                ponta_indicador = pontos[8]
                print(
                    f"Ponta do indicador -> x={ponta_indicador['x']} "
                    f"y={ponta_indicador['y']} z={ponta_indicador['z']}"
                )

                # Desenha essa informação também na janela de vídeo.
                texto = f"Indicador: x={ponta_indicador['x']} y={ponta_indicador['y']}"
                cv2.putText(
                    frame, texto, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

        cv2.imshow("kara-sygna - Passo 3: Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()