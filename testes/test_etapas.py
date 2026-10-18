"""Testes das etapas de uma gravação (pré-sinal uma vez, sinal em 3 ritmos)."""

from kara_sygna.aplicacao.etapas import montar_etapas, sem_acentos
from kara_sygna.dados.repositorio import TIPO_PRE, TIPO_SINAL


def test_pre_sinal_uma_vez_e_sinal_em_tres_ritmos():
    etapas = montar_etapas("carro", "S", 10)
    assert [e.legenda for e in etapas] == [
        "pré-sinal S",
        "carro 1/3: devagar",
        "carro 2/3: médio",
        "carro 3/3: rápido",
    ]
    assert [e.tipo for e in etapas] == [TIPO_PRE] + [TIPO_SINAL] * 3
    assert all(e.pre_sinal == "S" for e in etapas[1:]) and etapas[0].pre_sinal == ""
    assert all(e.duracao == 10 for e in etapas)
    assert [e.legenda_video for e in etapas] == [
        "PRE-SINAL S",
        "carro 1/3: DEVAGAR",
        "carro 2/3: MEDIO",
        "carro 3/3: RAPIDO",
    ]


def test_neutro_sem_ritmo_e_tempo_padrao_quando_vazio():
    etapas = montar_etapas("neutro")
    assert [e.legenda for e in etapas] == ["neutro 1/3", "neutro 2/3", "neutro 3/3"]
    assert etapas[0].legenda_video == "neutro 1/3"
    assert all(e.duracao is not None for e in etapas)  # a sequência anda sozinha


def test_uma_vez_so_grava_ate_parar():
    etapas = montar_etapas("oi", ritmos=())
    assert len(etapas) == 1 and etapas[0].duracao is None
    assert etapas[0].legenda == "oi 1/1"


def test_sem_acentos():
    assert sem_acentos("médio rápido pré") == "medio rapido pre"
