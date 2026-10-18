"""Testes da previsão e da votação do Estabilizador."""

import numpy as np
import pytest

from kara_sygna import config
from kara_sygna.ia.reconhecimento import Estabilizador, prever


class ModeloFalso:
    """Devolve sempre as mesmas probabilidades; guarda o que recebeu."""

    classes_ = np.array(["a", "b", "c"])

    def __init__(self, probabilidades):
        self.probabilidades = np.asarray(probabilidades, dtype=float)
        self.entrada = None

    def predict_proba(self, X):
        self.entrada = np.asarray(X)
        return np.tile(self.probabilidades, (len(self.entrada), 1))


def test_aceita_quando_confianca_e_margem_passam():
    p = prever(ModeloFalso([0.8, 0.15, 0.05]), np.zeros(126))
    assert (p.sinal, p.aceito) == ("a", True)
    assert p.margem == pytest.approx(0.65)


@pytest.mark.parametrize("probabilidades", [[0.6, 0.3, 0.1], [0.74, 0.7, 0.0]])
def test_recusa_confianca_baixa_ou_margem_pequena(probabilidades):
    assert not prever(ModeloFalso(probabilidades), np.zeros(126)).aceito


def test_avalia_o_vetor_uma_unica_vez():
    modelo = ModeloFalso([1, 0, 0])
    prever(modelo, np.arange(232.0))
    assert modelo.entrada.shape == (1, 232)
    assert np.array_equal(modelo.entrada[0], np.arange(232.0))


def test_estabilizador_confirma_depois_de_votos_suficientes():
    e = Estabilizador(janela=10, votos_minimos=3, quadros_para_liberar=2)
    assert [e.atualizar("a") for _ in range(3)] == [None, None, "a"]


def test_estabilizador_nao_repete_sinal_segurado():
    e = Estabilizador(janela=10, votos_minimos=3, quadros_para_liberar=2)
    confirmados = [e.atualizar("a") for _ in range(20)]
    assert confirmados.count("a") == 1


def test_estabilizador_permite_repetir_depois_de_soltar():
    e = Estabilizador(janela=4, votos_minimos=3, quadros_para_liberar=2)
    saida = [e.atualizar(s) for s in ["a"] * 3 + [None] * 4 + ["a"] * 3]
    assert saida.count("a") == 2


def test_estabilizador_ignora_ruido_isolado():
    e = Estabilizador(janela=10, votos_minimos=6, quadros_para_liberar=8)
    assert [e.atualizar(s) for s in ["a", "b", "a", "c", "a", "b"]] == [None] * 6


def test_estabilizador_reiniciar_esquece_o_travamento():
    e = Estabilizador(janela=4, votos_minimos=2, quadros_para_liberar=8)
    e.atualizar("a")
    assert e.atualizar("a") == "a"
    e.reiniciar()
    e.atualizar("a")
    assert e.atualizar("a") == "a"


def test_valores_padrao_vem_da_configuracao():
    e = Estabilizador()
    assert (e.janela, e.votos_minimos, e.quadros_para_liberar) == (
        config.JANELA_ESTABILIZADOR,
        config.VOTOS_MINIMOS,
        config.QUADROS_PARA_LIBERAR,
    )
