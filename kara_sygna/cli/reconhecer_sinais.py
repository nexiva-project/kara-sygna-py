"""Reconhece sinais ao vivo em uma janela do OpenCV (sem a interface Tk).

Uso:  python -m kara_sygna.cli.reconhecer_sinais

Pré-requisito: ter treinado a IA. Teclas: 'q' sai, 'c' limpa o texto.
"""

from __future__ import annotations

import sys

import cv2

from kara_sygna.aplicacao.controlador import ControladorSinais, Nivel
from kara_sygna.aplicacao.erros import ErroAplicacao
from kara_sygna.aplicacao.inicializacao import preparar_ambiente
from kara_sygna.camera import CameraIndisponivelError, abrir_camera
from kara_sygna.ia.modelo_io import ModeloInvalidoError
from kara_sygna.visao.caracteristicas import montar_vetor_duas_maos
from kara_sygna.visao.detector_maos import DetectorMaos, tem_maos

JANELA = "kara-sygna - Reconhecimento"
COR_NIVEL = {
    Nivel.NEUTRO: (200, 200, 200),
    Nivel.OK: (0, 255, 0),
    Nivel.INCERTO: (0, 165, 255),
    Nivel.ENSINO: (0, 220, 255),
}


def main() -> int:
    """Abre a câmera e mostra o sinal reconhecido e a frase formada.

    Returns:
        0 se terminou normalmente, 1 se não foi possível começar.
    """
    preparar_ambiente()
    controlador = ControladorSinais()
    try:
        controlador.iniciar_reconhecimento()
        camera = abrir_camera()
    except (ErroAplicacao, ModeloInvalidoError, CameraIndisponivelError) as erro:
        print(erro, file=sys.stderr)
        return 1

    frase: list[str] = []
    print("Câmera aberta! Faça um sinal. 'q' sai, 'c' limpa o texto.")
    try:
        with DetectorMaos() as detector:
            while True:
                quadro = camera.ler()
                if quadro is None:
                    break
                quadro = cv2.flip(quadro, 1)
                resultado = detector.detectar(quadro)
                detector.desenhar(quadro, resultado)
                vetor = (
                    montar_vetor_duas_maos(resultado) if tem_maos(resultado) else None
                )
                resposta = controlador.processar_quadro(vetor)
                if resposta.sinal_confirmado:
                    frase.append(resposta.sinal_confirmado)

                cv2.putText(
                    quadro,
                    resposta.texto_video or "Nenhuma mao detectada",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    COR_NIVEL[resposta.nivel],
                    2,
                )
                cv2.putText(
                    quadro,
                    " ".join(frase[-8:]),
                    (10, quadro.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(JANELA, quadro)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                if tecla == ord("c"):
                    frase.clear()
                    controlador.limpar_reconhecimento()
    finally:
        camera.liberar()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
