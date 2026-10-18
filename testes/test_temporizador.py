"""Testes da contagem antes do ensino e da parada automática."""

import pytest

from kara_sygna.aplicacao.erros import DuracaoInvalidaError
from kara_sygna.aplicacao.temporizador import (
    Evento,
    Fase,
    TemporizadorEnsino,
    ler_duracao,
)
from testes.test_filtro_quadros import Relogio


@pytest.fixture
def relogio():
    return Relogio()


def test_ler_duracao():
    assert ler_duracao("10") == 10
    assert ler_duracao(" 7,5 ") == 7.5
    assert ler_duracao("8s") == 8
    assert ler_duracao("") is None
    for invalido in ("abc", "0", "-3", "nan", "100000"):
        with pytest.raises(DuracaoInvalidaError):
            ler_duracao(invalido)


def test_conta_3_s_grava_o_tempo_escolhido_e_para(relogio):
    t = TemporizadorEnsino(3, relogio)
    t.iniciar(10)
    assert t.fase is Fase.CONTAGEM and t.segundos_restantes() == 3
    relogio.agora = 2.5
    assert t.atualizar() is Evento.NADA and t.segundos_restantes() == 1
    relogio.agora = 3.0
    assert t.atualizar() is Evento.COMECAR and t.fase is Fase.GRAVANDO
    assert t.segundos_restantes() == 10
    relogio.agora = 12.9
    assert t.atualizar() is Evento.NADA and t.segundos_restantes() == 1
    relogio.agora = 13.0
    assert t.atualizar() is Evento.PARAR and not t.ativo
    assert t.atualizar() is Evento.NADA


def test_sem_tempo_grava_ate_parar(relogio):
    t = TemporizadorEnsino(3, relogio)
    t.iniciar(None)
    relogio.agora = 3
    assert t.atualizar() is Evento.COMECAR
    relogio.agora = 1000
    assert t.atualizar() is Evento.NADA and t.segundos_restantes() is None
    t.cancelar()
    assert not t.ativo


def test_cancelar_na_contagem_nao_comeca(relogio):
    t = TemporizadorEnsino(3, relogio)
    t.iniciar(5)
    t.cancelar()
    relogio.agora = 10
    assert t.atualizar() is Evento.NADA and t.fase is Fase.PARADO
