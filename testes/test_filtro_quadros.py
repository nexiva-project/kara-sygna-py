"""Testes do filtro que escolhe quais quadros da câmera são usados."""

import numpy as np
import pytest

from kara_sygna import config
from kara_sygna.aplicacao.filtro_quadros import FiltroQuadros
from kara_sygna.visao.quadro import COLUNAS_POSICAO
from testes.conftest import quadro_falso, vetor_de_duas_maos, vetor_de_uma_mao


class Relogio:
    """Relógio de mentira: o teste decide quanto tempo passou."""

    def __init__(self):
        self.agora = 0.0

    def __call__(self):
        return self.agora


@pytest.fixture
def relogio():
    return Relogio()


def _mudanca(relogio, **opcoes):
    return FiltroQuadros("mudanca", relogio=relogio, **opcoes)


def test_primeiro_quadro_e_sempre_usado(rng, relogio):
    assert _mudanca(relogio).aceitar(quadro_falso(vetor_de_duas_maos(rng)))


def test_mao_parada_nao_e_usada_de_novo(rng, relogio):
    filtro = _mudanca(relogio, parado_a_cada_s=0)
    quadro = quadro_falso(vetor_de_duas_maos(rng))
    assert filtro.aceitar(quadro)
    relogio.agora = 10.0
    assert not filtro.aceitar(quadro + 0.001)  # mudança menor que o limiar


def test_mudanca_na_forma_e_usada(rng, relogio):
    filtro = _mudanca(relogio, parado_a_cada_s=0)
    maos = vetor_de_duas_maos(rng)
    filtro.aceitar(quadro_falso(maos))
    assert filtro.aceitar(quadro_falso(maos + 0.1 * (maos != 0)))


def test_mudanca_no_lugar_e_usada_mesmo_com_a_mesma_forma(rng, relogio):
    filtro = _mudanca(relogio, parado_a_cada_s=0)
    maos = vetor_de_uma_mao(rng, "dir")
    filtro.aceitar(quadro_falso(maos, posicao_dir=(0.0, 0.0)))
    assert not filtro.aceitar(quadro_falso(maos, posicao_dir=(0.01, 0.0)))
    assert filtro.aceitar(quadro_falso(maos, posicao_dir=(0.0, 0.5)))


def test_mao_que_aparece_ou_some_e_usada(rng, relogio):
    filtro = _mudanca(relogio, parado_a_cada_s=0)
    sem_mao = quadro_falso(None)
    assert filtro.aceitar(sem_mao)
    assert not filtro.aceitar(sem_mao)
    assert filtro.aceitar(quadro_falso(vetor_de_uma_mao(rng)))
    assert filtro.aceitar(sem_mao)


def test_mao_parada_e_usada_de_tempos_em_tempos(rng, relogio):
    filtro = _mudanca(relogio, parado_a_cada_s=0.5)
    quadro = quadro_falso(vetor_de_duas_maos(rng))
    assert filtro.aceitar(quadro)
    relogio.agora = 0.3
    assert not filtro.aceitar(quadro)
    relogio.agora = 0.5
    assert filtro.aceitar(quadro)


def test_reiniciar_esquece_o_ultimo_quadro(rng, relogio):
    filtro = _mudanca(relogio, parado_a_cada_s=0)
    quadro = quadro_falso(vetor_de_duas_maos(rng))
    filtro.aceitar(quadro)
    filtro.reiniciar()
    assert filtro.aceitar(quadro)


def test_modo_fixo_usa_1_a_cada_n(rng):
    filtro = FiltroQuadros("fixo", a_cada=3)
    quadro = quadro_falso(vetor_de_duas_maos(rng))
    assert [filtro.aceitar(quadro) for _ in range(7)] == [
        True, False, False, True, False, False, True,
    ]  # fmt: skip


def test_valores_padrao_vem_do_config(monkeypatch):
    monkeypatch.setattr(config, "MODO_QUADROS", "fixo")
    monkeypatch.setattr(config, "USAR_1_A_CADA_QUADROS", 4)
    monkeypatch.setattr(config, "LIMIAR_MUDANCA_FORMA", 0.2)
    filtro = FiltroQuadros()
    assert (filtro.modo, filtro.a_cada, filtro.limiar) == ("fixo", 4, 0.2)


def test_configuracao_invalida_e_recusada():
    with pytest.raises(ValueError, match="MODO_QUADROS"):
        FiltroQuadros("outro")
    with pytest.raises(ValueError):
        FiltroQuadros("mudanca", limiar=-1)


def test_tamanho_errado_e_recusado_e_valores_invalidos_sao_ignorados(rng, relogio):
    filtro = _mudanca(relogio)
    with pytest.raises(ValueError):
        filtro.aceitar(np.zeros(126))
    quadro = quadro_falso(vetor_de_duas_maos(rng))
    quadro[COLUNAS_POSICAO["dir"][0]] = np.nan
    assert not filtro.aceitar(quadro)
