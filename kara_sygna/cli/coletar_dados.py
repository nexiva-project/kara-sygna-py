"""Grava exemplos de sinais pelo terminal (alternativa à interface gráfica).

Uso:  python -m kara_sygna.cli.coletar_dados

Para cada sinal: digite o nome, espere a contagem e faça o sinal. Cada
gravação é uma RODADA. Grave pelo menos 3 rodadas por sinal, mudando posição,
distância, inclinação e luz entre elas. Grave também um sinal "neutro" (mão
relaxada) para a IA aprender a dizer "isto não é um sinal".
"""

from __future__ import annotations

import sys
import time

import cv2

from kara_sygna.aplicacao.controlador import ControladorSinais
from kara_sygna.aplicacao.erros import ErroAplicacao
from kara_sygna.aplicacao.inicializacao import preparar_ambiente
from kara_sygna.camera import Camera, CameraIndisponivelError, abrir_camera
from kara_sygna.visao.caracteristicas import montar_vetor_duas_maos
from kara_sygna.visao.detector_maos import DetectorMaos, tem_maos

QUADROS_POR_RODADA = 80
INTERVALO_ENTRE_QUADROS_S = 0.10  # evita gravar quadros quase idênticos
SEGUNDOS_CONTAGEM = 3
JANELA = "kara-sygna - Coleta de Dados"


def gravar_rodada(
    camera: Camera, detector: DetectorMaos, controlador: ControladorSinais, palavra: str
) -> bool:
    """Grava uma rodada de um sinal.

    Args:
        camera: Câmera aberta.
        detector: Detector de mãos.
        controlador: Controlador (grava no CSV).
        palavra: Nome do sinal.

    Returns:
        ``False`` se o usuário interrompeu com 'q' (ou a câmera falhou).
    """
    controlador.iniciar_ensino(palavra)
    inicio = time.monotonic() + SEGUNDOS_CONTAGEM
    ultimo, gravados = 0.0, 0
    try:
        while gravados < QUADROS_POR_RODADA:
            quadro = camera.ler()
            if quadro is None:
                return False
            quadro = cv2.flip(quadro, 1)
            agora = time.monotonic()
            if agora < inicio:
                texto, cor = (
                    f"Prepare '{palavra}': {int(inicio - agora) + 1}",
                    (0, 200, 255),
                )
            else:
                resultado = detector.detectar(quadro)
                detector.desenhar(quadro, resultado)
                if tem_maos(resultado):
                    if agora - ultimo >= INTERVALO_ENTRE_QUADROS_S:
                        controlador.processar_quadro(montar_vetor_duas_maos(resultado))
                        gravados, ultimo = gravados + 1, agora
                    texto, cor = (
                        f"Gravando '{palavra}': {gravados}/{QUADROS_POR_RODADA}",
                        (0, 0, 255),
                    )
                else:
                    texto, cor = "Mostre a mao para a camera", (0, 165, 255)
            cv2.putText(quadro, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
            cv2.imshow(JANELA, quadro)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
    finally:
        controlador.parar()
    return True


def main() -> int:
    """Pergunta o nome de cada sinal e grava uma rodada por vez.

    Returns:
        0 se terminou normalmente, 1 se não conseguiu abrir a câmera.
    """
    preparar_ambiente()
    controlador = ControladorSinais()
    try:
        camera = abrir_camera()
    except CameraIndisponivelError as erro:
        print(erro, file=sys.stderr)
        return 1

    print(f"Dados serão salvos em: {controlador.repositorio.caminho}")
    print("Durante a gravação, 'q' na janela interrompe.\n")
    try:
        with DetectorMaos() as detector:
            while True:
                palavra = (
                    input("Nome do sinal (Enter vazio para sair): ").strip().lower()
                )
                if not palavra:
                    break
                try:
                    if not gravar_rodada(camera, detector, controlador, palavra):
                        break
                except ErroAplicacao as erro:
                    print(erro)
                    continue
                print(
                    f"Rodada de '{palavra}' concluída. Repita variando posição/distância.\n"
                )
    finally:
        camera.liberar()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
