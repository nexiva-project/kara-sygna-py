"""Apoio dos testes manuais: abre a câmera e mostra o vídeo (ou salva uma foto)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from kara_sygna.camera import Camera, CameraIndisponivelError, abrir_camera
from kara_sygna.logs import configurar_logging

#: Função que recebe o quadro espelhado e devolve o quadro a exibir.
Processador = Callable[[np.ndarray], np.ndarray]


def executar(titulo: str, processar: Processador | None = None) -> int:
    """Roda um teste manual de câmera.

    Opções da linha de comando: ``--foto ARQUIVO`` salva um único quadro e sai
    (útil em computador sem tela ou para conferir a imagem); ``--indice N``
    força um índice de câmera.

    Args:
        titulo: Título da janela.
        processar: Função que desenha algo sobre o quadro. ``None`` mostra o
            vídeo puro.

    Returns:
        Código de saída (0 = ok).
    """
    analisador = argparse.ArgumentParser(description=titulo)
    analisador.add_argument(
        "--foto", type=Path, help="salva um quadro neste arquivo e termina"
    )
    analisador.add_argument(
        "--indice", type=int, help="usa só este índice de câmera (padrão: tenta todos)"
    )
    argumentos = analisador.parse_args()

    configurar_logging()
    try:
        camera: Camera = abrir_camera(
            indices=None if argumentos.indice is None else [argumentos.indice]
        )
    except CameraIndisponivelError as erro:
        print(erro, file=sys.stderr)
        return 1

    print(f"Usando {camera.descricao}. Pressione 'q' na janela para sair.")
    try:
        while True:
            quadro = camera.ler()
            if quadro is None:
                print("Não foi possível ler um quadro da câmera.", file=sys.stderr)
                return 1
            quadro = cv2.flip(quadro, 1)
            if processar is not None:
                quadro = processar(quadro)
            if argumentos.foto:
                cv2.imwrite(str(argumentos.foto), quadro)
                print(f"Foto salva em {argumentos.foto}")
                return 0
            cv2.imshow(titulo, quadro)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return 0
    finally:
        camera.liberar()
        cv2.destroyAllWindows()


__all__ = ["executar"]
