"""Testes do realce de pele (imagem que vai para o detector de mãos)."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from kara_sygna.visao.realce_pele import PONTOS_PELE, RealcePele

PELE_BGR = (120, 150, 200)  # tom de pele em BGR
ROUPA_BGR = (128, 128, 128)  # cinza (Cr, Cb perto de 128)


def _quadro():
    """Metade esquerda "pele", metade direita "roupa cinza"."""
    quadro = np.zeros((120, 160, 3), np.uint8)
    quadro[:, :80] = PELE_BGR
    quadro[:, 80:] = ROUPA_BGR
    return quadro


def _rosto_na_pele():
    """Rosto falso com todos os pontos de pele no meio da metade "pele"."""
    pontos = [SimpleNamespace(x=0.25, y=0.5, z=0.0) for _ in range(468)]
    return SimpleNamespace(presente=True, landmarks=SimpleNamespace(landmark=pontos))


def _brilho(imagem, colunas):
    return float(imagem[:, colunas].mean())


def test_desligado_devolve_o_mesmo_quadro():
    quadro = _quadro()
    assert RealcePele(ativo=False).aplicar(quadro) is quadro


def test_com_cor_do_rosto_a_pele_fica_e_a_roupa_escurece():
    quadro = _quadro()
    realce = RealcePele(ativo=True, escuro=0.1)
    realce.calibrar(quadro, _rosto_na_pele())
    assert realce.calibrado
    saida = realce.aplicar(quadro)
    assert saida.shape == quadro.shape and saida.dtype == np.uint8
    # Longe da borda (o borrão da máscara invade alguns pixels).
    assert _brilho(saida, slice(0, 50)) == pytest.approx(
        _brilho(quadro, slice(0, 50)), abs=3
    )
    assert _brilho(saida, slice(120, 160)) < 0.2 * _brilho(quadro, slice(120, 160))


def test_cor_medida_segue_o_tom_do_rosto():
    realce = RealcePele(ativo=True)
    quadro = _quadro()
    realce.calibrar(quadro, _rosto_na_pele())
    _, cr, cb = cv2.cvtColor(np.uint8([[PELE_BGR]]), cv2.COLOR_BGR2YCrCb)[0, 0]
    assert realce.media == pytest.approx([cr, cb], abs=1)


def test_sem_rosto_nao_calibra_e_usa_a_faixa_padrao():
    realce = RealcePele(ativo=True)
    realce.calibrar(_quadro(), SimpleNamespace(presente=False, landmarks=None))
    assert not realce.calibrado
    # A faixa padrão também separa a pele da roupa cinza.
    mascara = realce.mascara(_quadro())
    assert mascara[:, :50].mean() > 200 and mascara[:, 120:].mean() < 20


def test_escuro_invalido():
    with pytest.raises(ValueError):
        RealcePele(ativo=True, escuro=1.5)


def test_pontos_de_pele_existem_no_facemesh():
    assert max(PONTOS_PELE) < 468
