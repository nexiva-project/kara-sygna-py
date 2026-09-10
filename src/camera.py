import cv2

def main():
    # 0 = webcam padrão do computador.
    # Se você tiver mais de uma câmera, pode testar 1, 2, etc.
    captura = cv2.VideoCapture(0)

    if not captura.isOpened():
        print("Não foi possível acessar a câmera. Verifique se ela está conectada"
              " e se nenhum outro programa está usando ela.")
        return

    print("Câmera aberta com sucesso! Pressione 'q' na janela de vídeo para sair.")

    while True:
        # 'ret' indica se o frame foi lido com sucesso.
        # 'frame' é a imagem capturada naquele instante.
        ret, frame = captura.read()

        if not ret:
            print("Não foi possível ler o frame da câmera. Encerrando...")
            break

        # Espelha a imagem horizontalmente (efeito "espelho"),
        # que é mais natural para quem está se vendo na tela.
        frame = cv2.flip(frame, 1)

        cv2.imshow("kara-sygna - Passo 1: Camera", frame)

        # Aguarda 1ms por uma tecla; se for 'q', encerra o loop.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Libera a câmera e fecha as janelas abertas.
    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()