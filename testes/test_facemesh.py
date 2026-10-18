import cv2

from kara_sygna.visao.detector_rosto import DetectorRosto


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Não foi possível abrir a webcam.")
        return

    try:
        with DetectorRosto(desenhar_malha=True) as detector:
            while True:
                sucesso, quadro = camera.read()

                if not sucesso:
                    print("Falha ao ler a webcam.")
                    break

                resultado = detector.detectar(quadro)
                tela = detector.desenhar(quadro, resultado)

                cv2.imshow("Kara Sygna - FaceMesh", tela)

                if resultado.presente:
                    print(
                        "Rosto detectado:",
                        resultado.quantidade_pontos,
                        "pontos;",
                        resultado.vetor.shape[0],
                        "características",
                    )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
