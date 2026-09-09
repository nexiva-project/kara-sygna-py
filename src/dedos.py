import cv2
import mediapipe as mp

# Para cada dedo (exceto o polegar), guardamos:
#   (indice_da_ponta, indice_da_articulacao_de_comparacao)
DEDOS = {
    "indicador": (8, 6),
    "medio": (12, 10),
    "anelar": (16, 14),
    "mindinho": (20, 18),
}

# O polegar é tratado à parte, devido ao movimento lateral.
POLEGAR_PONTA = 4
POLEGAR_ARTICULACAO = 2


def dedos_levantados(landmarks_da_mao, mao_e_direita):
    """
    Recebe os landmarks de UMA mão e retorna uma lista com os nomes
    dos dedos que estão esticados/levantados.

    'mao_e_direita' importa só para o polegar, porque a direção
    lateral que indica "esticado" se inverte dependendo da mão.
    """
    pontos = landmarks_da_mao.landmark
    levantados = []

    # Dedos indicador, médio, anelar, mindinho: compara eixo Y.
    for nome_dedo, (ponta, articulacao) in DEDOS.items():
        if pontos[ponta].y < pontos[articulacao].y:
            levantados.append(nome_dedo)

    # Polegar: compara eixo X, e a direção depende de qual mão é.
    x_ponta = pontos[POLEGAR_PONTA].x
    x_articulacao = pontos[POLEGAR_ARTICULACAO].x

    if mao_e_direita:
        polegar_esticado = x_ponta > x_articulacao
    else:
        polegar_esticado = x_ponta < x_articulacao

    if polegar_esticado:
        levantados.append("polegar")

    return levantados


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

        if resultado.multi_hand_landmarks and resultado.multi_handedness:
            # 'multi_handedness' diz se cada mão detectada é a
            # esquerda ou a direita (do ponto de vista de quem
            # está na frente da câmera, já considerando o espelho).
            zipped = zip(resultado.multi_hand_landmarks, resultado.multi_handedness)
            for landmarks_da_mao, info_da_mao in zipped:
                mp_desenho.draw_landmarks(
                    frame, landmarks_da_mao, mp_hands.HAND_CONNECTIONS
                )

                rotulo = info_da_mao.classification[0].label  # "Left" ou "Right"
                mao_e_direita = rotulo == "Left"

                levantados = dedos_levantados(landmarks_da_mao, mao_e_direita)
                quantidade = len(levantados)

                texto = f"Dedos levantados: {quantidade} {levantados}"
                cv2.putText(
                    frame, texto, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
                )
                print(texto)

        cv2.imshow("kara-sygna - Passo 4: Contar Dedos", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    captura.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()