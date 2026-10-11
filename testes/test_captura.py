"""Testes da abertura de câmera (com câmeras falsas)."""

import numpy as np
import pytest

from kara_sygna.camera import (
    Backend,
    CameraIndisponivelError,
    abrir_camera,
    backends_da_plataforma,
)
from kara_sygna.camera.captura import _mensagem_de_falha


class CapturaFalsa:
    """Imita ``cv2.VideoCapture``: abre ou não, e entrega quadros após N leituras vazias."""

    def __init__(self, aberta=True, leituras_vazias=0):
        self.aberta, self.vazias, self.liberada = aberta, leituras_vazias, False

    def isOpened(self):
        return self.aberta

    def set(self, *_):
        return True

    def read(self):
        if self.vazias > 0:
            self.vazias -= 1
            return False, None
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        self.liberada = True


def fabrica_de(mapa, chamadas=None):
    """Fábrica que devolve a captura do mapa ``{(indice, codigo): CapturaFalsa}``."""

    def fabrica(indice, codigo):
        if chamadas is not None:
            chamadas.append((indice, codigo))
        return mapa.get((indice, codigo), CapturaFalsa(aberta=False))

    return fabrica


V4L2, AUTO = Backend("V4L2", 200), Backend("auto", 0)
RAPIDO = {"tentativas_leitura": 3, "pausa_s": 0}


def test_backends_por_sistema():
    assert next(b.nome for b in backends_da_plataforma("Windows")) == "DirectShow"
    assert [b.nome for b in backends_da_plataforma("Linux")] == ["V4L2", "automático"]
    assert "DirectShow" not in [b.nome for b in backends_da_plataforma("Linux")]
    assert backends_da_plataforma("Darwin")[0].nome == "AVFoundation"
    assert [b.nome for b in backends_da_plataforma("Outro")] == ["automático"]


def test_pula_dispositivo_que_abre_mas_nao_entrega_imagem():
    # Caso comum no Linux: /dev/video0 é a imagem e /dev/video1 é só metadados.
    metadados = CapturaFalsa(aberta=True, leituras_vazias=999)
    mapa = {(0, 200): metadados, (1, 200): CapturaFalsa()}
    camera = abrir_camera([0, 1], [V4L2], fabrica=fabrica_de(mapa), **RAPIDO)
    assert camera.indice == 1 and metadados.liberada


def test_aceita_camera_que_precisa_aquecer():
    mapa = {(0, 200): CapturaFalsa(leituras_vazias=2)}
    camera = abrir_camera([0], [V4L2], fabrica=fabrica_de(mapa), **RAPIDO)
    assert camera.indice == 0 and (camera.largura, camera.altura) == (640, 480)


def test_usa_proximo_backend_quando_o_primeiro_falha():
    mapa = {(0, 0): CapturaFalsa()}
    camera = abrir_camera([0], [V4L2, AUTO], fabrica=fabrica_de(mapa), **RAPIDO)
    assert camera.backend == AUTO
    assert "automático" not in camera.descricao and "auto" in camera.descricao


def test_erro_lista_as_tentativas():
    with pytest.raises(CameraIndisponivelError) as info:
        abrir_camera([0, 1], [V4L2], fabrica=fabrica_de({}), **RAPIDO)
    assert "câmera 0 (V4L2): não abriu" in str(info.value)
    assert "câmera 1 (V4L2): não abriu" in str(info.value)


def test_excecao_do_driver_nao_derruba_a_busca():
    def fabrica(indice, codigo):
        if indice == 0:
            raise OSError("driver quebrado")
        return CapturaFalsa()

    camera = abrir_camera([0, 1], [V4L2], fabrica=fabrica, **RAPIDO)
    assert camera.indice == 1


def test_dicas_do_linux_so_aparecem_no_linux():
    assert "/dev/video" in _mensagem_de_falha(["x"], "Linux")
    assert "/dev/video" not in _mensagem_de_falha(["x"], "Windows")


def test_liberar_e_contexto():
    captura = CapturaFalsa()
    with abrir_camera(
        [0], [V4L2], fabrica=fabrica_de({(0, 200): captura}), **RAPIDO
    ) as camera:
        assert camera.ler().shape == (480, 640, 3)
    assert captura.liberada
