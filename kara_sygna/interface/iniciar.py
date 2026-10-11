"""Ponto de entrada da interface gráfica."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

MENSAGEM_SEM_TK = (
    "O Tkinter não está instalado. No Ubuntu/Debian, instale com:\n"
    "    sudo apt install python3-tk\n"
    "(depois reinicie o programa; se usar uma venv, ela precisa ter sido criada com o Python do sistema)."
)


def executar() -> int:
    """Abre o Kara Sygna.

    Returns:
        Código de saída: 0 se encerrou normalmente, 1 se não conseguiu iniciar.
    """
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        print(MENSAGEM_SEM_TK, file=sys.stderr)
        return 1

    # Importados aqui porque dependem do Tk.
    from kara_sygna.aplicacao.controlador import ControladorSinais
    from kara_sygna.aplicacao.inicializacao import preparar_ambiente
    from kara_sygna.camera import CameraIndisponivelError, abrir_camera
    from kara_sygna.interface.janela_principal import JanelaPrincipal
    from kara_sygna.visao.detector_maos import DetectorMaos

    preparar_ambiente()
    raiz = tk.Tk()
    raiz.withdraw()  # só aparece quando câmera e MediaPipe estiverem prontos

    try:
        detector = DetectorMaos()
    except Exception as erro:
        logger.error("Falha ao iniciar o MediaPipe", exc_info=erro)
        messagebox.showerror(
            "MediaPipe", f"Não foi possível iniciar o MediaPipe:\n\n{erro}"
        )
        return 1

    try:
        camera = abrir_camera()
    except CameraIndisponivelError as erro:
        logger.error("%s", erro)
        detector.fechar()
        messagebox.showerror("Câmera", str(erro))
        return 1

    janela = JanelaPrincipal(raiz, ControladorSinais(), camera, detector)
    raiz.deiconify()
    janela.iniciar()
    return 0
