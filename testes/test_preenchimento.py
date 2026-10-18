"""Testes do preenchimento das mãos que somem por pouco tempo."""

import numpy as np
import pytest

from kara_sygna.visao.preenchimento import preencher_buracos
from kara_sygna.visao.quadro import COLUNAS_POSICAO, IDX_FORMA, maos_presentes
from testes.conftest import quadro_falso, vetor_de_uma_mao


def _gravacao(rng, presentes, passo_s=1 / 30):
    """Mão direita andando em x; ``presentes`` diz em que quadros ela aparece."""
    forma = vetor_de_uma_mao(rng, "dir")
    quadros = np.array(
        [
            quadro_falso(forma if p else None, (i / 10, 0.0))
            for i, p in enumerate(presentes)
        ]
    )
    return quadros, np.arange(len(presentes)) * passo_s


def test_buraco_curto_vira_o_caminho_entre_antes_e_depois(rng):
    quadros, tempos = _gravacao(rng, [1, 1, 1, 0, 0, 1, 1, 1])
    saida, preenchidos = preencher_buracos(quadros, tempos, 0.5)
    assert preenchidos == 2
    assert maos_presentes(saida)[:, 1].all()
    x = COLUNAS_POSICAO["dir"][0]
    assert saida[3, x] == pytest.approx(0.3) and saida[4, x] == pytest.approx(0.4)
    # A forma da mão também é preenchida (fica entre as vizinhas).
    assert np.allclose(saida[3, IDX_FORMA["dir"]], quadros[2, IDX_FORMA["dir"]])
    assert not maos_presentes(quadros)[3, 1]  # a entrada não muda


def test_buraco_longo_e_pontas_ficam_sem_mao(rng):
    quadros, tempos = _gravacao(rng, [0, 1, 0, 0, 0, 1, 0], passo_s=0.2)
    saida, preenchidos = preencher_buracos(quadros, tempos, 0.5)
    assert preenchidos == 0
    assert np.array_equal(maos_presentes(saida), maos_presentes(quadros))


def test_poucos_pontos_usa_reta_e_zero_desliga(rng):
    quadros, tempos = _gravacao(rng, [1, 0, 1])
    saida, preenchidos = preencher_buracos(quadros, tempos, 0.5)
    assert preenchidos == 1
    x = COLUNAS_POSICAO["dir"][0]
    assert saida[1, x] == pytest.approx(0.1)
    assert preencher_buracos(quadros, tempos, 0)[1] == 0


def test_tempos_errados_sao_recusados(rng):
    quadros, tempos = _gravacao(rng, [1, 0, 1])
    with pytest.raises(ValueError):
        preencher_buracos(quadros, tempos[:2])
