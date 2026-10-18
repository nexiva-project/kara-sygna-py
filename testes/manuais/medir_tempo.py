"""Mede o tempo médio de cada etapa por quadro.

python -m testes.manuais.medir_tempo     (mostre as mãos para a câmera)
"""

import time
from collections import defaultdict
from contextlib import contextmanager

import cv2

from kara_sygna import config
from kara_sygna.camera import abrir_camera
from kara_sygna.ia.modelo_io import carregar_modelo
from kara_sygna.ia.reconhecimento import prever
from kara_sygna.ia.sequencia import BufferQuadros
from kara_sygna.visao.detector_maos import DetectorMaos, tem_maos
from kara_sygna.visao.detector_rosto import DetectorRosto
from kara_sygna.visao.quadro import montar_vetor_quadro

QUADROS = 200
tempos: dict[str, float] = defaultdict(float)


@contextmanager
def medir(nome):
    inicio = time.perf_counter()
    yield
    tempos[nome] += time.perf_counter() - inicio


def main():
    modelo = carregar_modelo(config.ARQUIVO_MODELO)
    quadros = previsoes = 0
    buffer = BufferQuadros()
    with abrir_camera() as camera, DetectorMaos() as maos, DetectorRosto() as rosto:
        for _ in range(QUADROS):
            with medir("leitura"):
                quadro = camera.ler()
            if quadro is None:
                break
            quadro = cv2.flip(quadro, 1)
            quadros += 1
            with medir("mãos (detectar)"):
                res_maos = maos.detectar(quadro)
            with medir("rosto (detectar)"):
                res_rosto = rosto.detectar(quadro)
            with medir("rosto (desenhar)"):
                rosto.desenhar(quadro, res_rosto)
            with medir("montar vetor do quadro"):
                altura, largura = quadro.shape[:2]
                buffer.adicionar(
                    montar_vetor_quadro(res_maos, res_rosto, largura, altura)
                )
            if tem_maos(res_maos) and buffer.cheio:
                previsoes += 1
                with medir("prever (janela)"):
                    prever(modelo, buffer.features())

    for nome, total in tempos.items():
        n = previsoes if nome.startswith("prever") else quadros
        print(f"{nome:24s} {1000 * total / max(1, n):7.2f} ms")


if __name__ == "__main__":
    main()
