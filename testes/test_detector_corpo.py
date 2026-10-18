"""O app segue sem os braços se o MediaPipe Pose não abrir."""

from kara_sygna.visao import detector_corpo


def test_sem_pose_o_app_continua(monkeypatch):
    def falha(*_a, **_k):
        raise OSError("sem internet para baixar o modelo")

    monkeypatch.setattr(detector_corpo, "DetectorCorpo", falha)
    corpo = detector_corpo.criar_detector_corpo()
    assert isinstance(corpo, detector_corpo.SemCorpo)
    with corpo:
        assert corpo.detectar(None) is None
