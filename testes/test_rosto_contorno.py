"""O desenho do rosto segue o contorno da imagem de referência."""

from types import SimpleNamespace

import mediapipe as mp
import numpy as np

from kara_sygna.visao.detector_rosto import (
    COR_ROSTO_LEVE,
    conexoes_rosto_referencia,
    desenhar_contorno_rosto,
)


def test_contorno_inclui_oval_olhos_sobrancelhas_labios_e_nariz():
    face_mesh = mp.solutions.face_mesh
    conexoes = conexoes_rosto_referencia(face_mesh)
    for conjunto in (
        face_mesh.FACEMESH_FACE_OVAL,
        face_mesh.FACEMESH_LEFT_EYE,
        face_mesh.FACEMESH_RIGHT_EYE,
        face_mesh.FACEMESH_LEFT_EYEBROW,
        face_mesh.FACEMESH_RIGHT_EYEBROW,
        face_mesh.FACEMESH_LIPS,
        face_mesh.FACEMESH_NOSE,
    ):
        assert set(conjunto) <= set(conexoes)


def test_desenha_linha_e_ponto_do_contorno():
    pontos = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(468)]
    pontos[1] = SimpleNamespace(x=0.2, y=0.2, z=0.0)
    pontos[4] = SimpleNamespace(x=0.2, y=0.8, z=0.0)
    landmarks = SimpleNamespace(landmark=pontos)
    tela = np.zeros((50, 50, 3), dtype=np.uint8)
    desenhar_contorno_rosto(
        tela,
        landmarks,
        {(1, 4)},
        COR_ROSTO_LEVE,
        espessura=1,
        raio=1,
        desenhar_linhas=True,
        desenhar_pontos=True,
    )
    assert tela.any()
    assert tuple(tela[10, 10]) == COR_ROSTO_LEVE
