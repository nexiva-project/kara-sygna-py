"""O app segue sem os braços se o MediaPipe Pose não abrir.

O desenho do corpo (ombro, braço, antebraço, postura) só aparece na câmera.
"""

from types import SimpleNamespace

import numpy as np

from kara_sygna.visao import detector_corpo


def test_sem_pose_o_app_continua(monkeypatch):
    def falha(*_a, **_k):
        raise OSError("sem internet para baixar o modelo")

    monkeypatch.setattr(detector_corpo, "DetectorCorpo", falha)
    corpo = detector_corpo.criar_detector_corpo()
    assert isinstance(corpo, detector_corpo.SemCorpo)
    with corpo:
        assert corpo.detectar(None) is None


def _ponto(x: float, y: float, visivel: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=visivel)


def _corpo_de_pe() -> SimpleNamespace:
    pontos = [_ponto(0.0, 0.0, 0.0) for _ in range(33)]
    pontos[11] = _ponto(0.35, 0.30)
    pontos[12] = _ponto(0.65, 0.30)
    pontos[13] = _ponto(0.25, 0.48)
    pontos[14] = _ponto(0.75, 0.48)
    pontos[15] = _ponto(0.18, 0.66)
    pontos[16] = _ponto(0.82, 0.66)
    pontos[23] = _ponto(0.40, 0.78)
    pontos[24] = _ponto(0.60, 0.78)
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=pontos))


def test_desenha_ombro_braco_antebraco_e_postura():
    tela = np.zeros((100, 100, 3), dtype=np.uint8)
    detector_corpo.desenhar_corpo(tela, _corpo_de_pe())
    # O círculo do ombro fica branco. As linhas saem suavizadas, então a
    # cor pura do traço não cai num pixel só: o braço é o verde e o
    # antebraço é o rosa.
    assert tuple(tela[29, 34]) == detector_corpo.COR_OMBRO_BGR
    verde = (tela[:, :, 1] > 140) & (tela[:, :, 1] > tela[:, :, 2])
    rosa = (tela[:, :, 2] > 140) & (tela[:, :, 2] > tela[:, :, 1])
    assert verde.any()
    assert rosa.any()


def test_quadril_na_borda_ainda_desenha_o_tronco():
    pontos = [_ponto(0.0, 0.0, 0.0) for _ in range(33)]
    pontos[11] = _ponto(0.35, 0.25)
    pontos[12] = _ponto(0.65, 0.25)
    pontos[23] = _ponto(0.40, 1.08, 0.25)
    pontos[24] = _ponto(0.60, 1.08, 0.25)
    resultado = SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=pontos))
    tela = np.zeros((100, 100, 3), dtype=np.uint8)
    detector_corpo.desenhar_corpo(tela, resultado)
    assert tela[90:, 45:55].any()


def test_borda_do_pose_volta_para_o_quadro_original():
    pontos = [_ponto(0.5, 0.5), _ponto(0.2, 0.8)]
    resultado = SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=pontos))
    detector_corpo.reposicionar_landmarks(resultado, largura=100, altura=80, borda=20)
    # O centro da imagem com borda é o centro do quadro.
    assert pontos[0].x == 0.5
    assert pontos[0].y == 0.5
    # x = 0.2 na imagem de 140 px, menos 20 de borda, sobre 100.
    assert pontos[1].x == (0.2 * 140 - 20) / 100
    assert pontos[1].y == (0.8 * 120 - 20) / 80


def test_sem_corpo_deixa_a_imagem_intacta():
    tela = np.zeros((20, 20, 3), dtype=np.uint8)
    detector_corpo.desenhar_corpo(tela, None)
    assert not tela.any()
