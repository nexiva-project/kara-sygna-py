"""Testes da extração de características."""

import numpy as np
import pytest

from kara_sygna.visao import caracteristicas as c
from testes.conftest import (
    mao_falsa,
    resultado_falso,
    vetor_de_duas_maos,
    vetor_de_uma_mao,
)


def test_nomes_tem_126_colunas_unicas():
    nomes = c.nomes_features()
    assert len(nomes) == c.TOTAL_FEATURES == len(set(nomes))
    assert nomes[0] == "esq_x0" and nomes[-1] == "dir_z20"


def test_sem_maos_gera_vetor_zerado(rng):
    vetor = c.montar_vetor_duas_maos(resultado_falso())
    assert vetor.shape == (126,) and c.eh_vetor_zerado(vetor)


@pytest.mark.parametrize("rotulo, slot", [("Left", "esq"), ("Right", "dir")])
def test_uma_mao_vai_para_o_slot_da_sua_lateralidade(rng, rotulo, slot):
    vetor = c.montar_vetor_duas_maos(resultado_falso((mao_falsa(rng), rotulo)))
    esquerda, direita = c.dividir_maos(vetor)
    assert c.quantidade_de_maos(vetor) == 1
    assert not c.eh_vetor_zerado(esquerda if slot == "esq" else direita)


def test_duas_maos_usam_a_posicao_horizontal_mesmo_com_rotulos_iguais(rng):
    # Os dois rótulos "Left": a posição na imagem decide quem é quem.
    a, b = mao_falsa(rng, -0.3), mao_falsa(rng, 0.3)
    vetor = c.montar_vetor_duas_maos(resultado_falso((b, "Left"), (a, "Left")))
    assert c.quantidade_de_maos(vetor) == 2


def test_vetor_nao_depende_de_posicao_nem_de_tamanho(rng):
    base = mao_falsa(rng)
    movida = type(base)(
        landmark=[
            type(p)(x=p.x * 0.5 + 0.2, y=p.y * 0.5 + 0.1, z=p.z * 0.5)
            for p in base.landmark
        ]
    )
    a = c.montar_vetor_duas_maos(resultado_falso((base, "Right")))
    b = c.montar_vetor_duas_maos(resultado_falso((movida, "Right")))
    assert np.allclose(a, b, atol=1e-9)


def test_mao_degenerada_vira_zeros():
    ponto = type("P", (), {"x": 0.5, "y": 0.5, "z": 0.0})
    mao = type("M", (), {"landmark": [ponto() for _ in range(21)]})()
    assert c.eh_vetor_zerado(c.montar_vetor_duas_maos(resultado_falso((mao, "Left"))))


def test_espelhar_mao_duas_vezes_devolve_o_original(rng):
    mao = vetor_de_uma_mao(rng)[63:]
    assert np.allclose(c.espelhar_mao(c.espelhar_mao(mao)), mao)


def test_espelhar_features_troca_as_maos(rng):
    v = vetor_de_uma_mao(rng, "dir")
    espelhado = c.espelhar_features(v)
    esquerda, direita = c.dividir_maos(espelhado)
    assert c.eh_vetor_zerado(direita) and not c.eh_vetor_zerado(esquerda)
    assert np.allclose(c.espelhar_features(espelhado), v)


def test_espelhar_features_aceita_matriz(rng):
    matriz = np.vstack([vetor_de_uma_mao(rng, "esq"), vetor_de_duas_maos(rng)])
    assert c.espelhar_features(matriz).shape == (2, 126)


def test_tamanho_errado_gera_erro():
    with pytest.raises(ValueError):
        c.dividir_maos(np.zeros(10))
    with pytest.raises(ValueError):
        c.espelhar_mao(np.zeros(10))
