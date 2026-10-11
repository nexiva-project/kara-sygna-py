"""Testes das regras da aplicação (sem Tk, câmera ou MediaPipe)."""

import os

import numpy as np
import pytest

from kara_sygna.aplicacao.controlador import ControladorSinais, Modo, Nivel
from kara_sygna.aplicacao.erros import (
    ModeloAusenteError,
    ModeloDesatualizadoError,
    PalavraInvalidaError,
    SemDadosError,
)
from kara_sygna.aplicacao.previsualizacao import Reproducao, combinar_espelhos
from kara_sygna.dados.repositorio import RepositorioSinais
from kara_sygna.visao.caracteristicas import linhas_para_gravar
from testes.conftest import vetor_de_duas_maos, vetor_de_uma_mao
from testes.test_reconhecimento import ModeloFalso


@pytest.fixture
def ctrl(tmp_path):
    return ControladorSinais(
        RepositorioSinais(tmp_path / "d.csv"),
        tmp_path / "m.pkl",
        carregador_modelo=lambda _caminho: ModeloFalso([0.9, 0.05, 0.05]),
    )


def _preparar_modelo(ctrl, rng):
    ctrl.iniciar_ensino("a")
    ctrl.processar_quadro(vetor_de_duas_maos(rng))
    ctrl.parar()
    ctrl.caminho_modelo.write_text("modelo")
    os.utime(
        ctrl.caminho_modelo, (os.path.getmtime(ctrl.repositorio.caminho) + 10,) * 2
    )


def test_palavra_vazia_e_recusada(ctrl):
    with pytest.raises(PalavraInvalidaError):
        ctrl.iniciar_ensino("   ")
    assert ctrl.modo is Modo.PARADO


def test_ensino_grava_quadros_e_conta_linhas(ctrl, rng):
    ctrl.iniciar_ensino("  OI ")
    r1 = ctrl.processar_quadro(vetor_de_uma_mao(rng))  # 2 linhas (original + espelho)
    r2 = ctrl.processar_quadro(vetor_de_duas_maos(rng))  # 1 linha
    assert (
        ctrl.total_gravado == 3
        and r2.texto_video == "ENSINANDO: oi (3)"
        and r1.nivel is Nivel.ENSINO
    )
    assert ctrl.repositorio.contagem_por_sinal() == {"oi": 3}
    assert ctrl.parar().startswith("Ensino de “oi” parado: 3")


def test_quadro_sem_mao_nao_grava(ctrl):
    ctrl.iniciar_ensino("oi")
    assert ctrl.processar_quadro(None).texto_video == ""
    assert ctrl.repositorio.contagem_por_sinal() == {}


def test_modo_parado_ignora_quadros(ctrl, rng):
    assert ctrl.processar_quadro(vetor_de_duas_maos(rng)).texto_video == ""


def test_rodadas_diferentes_a_cada_ensino(ctrl, rng, monkeypatch):
    ids = iter([111, 222])
    monkeypatch.setattr(
        "kara_sygna.aplicacao.controlador.novo_id_rodada", lambda: next(ids)
    )
    for _ in range(2):
        ctrl.iniciar_ensino("a")
        ctrl.processar_quadro(vetor_de_duas_maos(rng))
        ctrl.parar()
    assert sorted(set(ctrl.repositorio.carregar().grupos.tolist())) == [111, 222]


def test_reconhecer_exige_dados_modelo_e_treino_atual(ctrl, rng):
    with pytest.raises(SemDadosError):
        ctrl.iniciar_reconhecimento()
    ctrl.iniciar_ensino("a")
    ctrl.processar_quadro(vetor_de_duas_maos(rng))
    ctrl.parar()
    with pytest.raises(ModeloAusenteError):
        ctrl.iniciar_reconhecimento()
    ctrl.caminho_modelo.write_text("x")
    os.utime(ctrl.caminho_modelo, (1, 1))
    with pytest.raises(ModeloDesatualizadoError):
        ctrl.iniciar_reconhecimento()


def test_reconhecimento_confirma_sinal_e_ignora_neutro(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    vetor = vetor_de_duas_maos(rng)
    confirmados = [ctrl.processar_quadro(vetor).sinal_confirmado for _ in range(10)]
    assert confirmados.count("a") == 1

    ctrl.limpar_reconhecimento()
    ctrl._modelo.classes_ = np.array(["neutro", "b", "c"])
    assert all(ctrl.processar_quadro(vetor).sinal_confirmado is None for _ in range(10))


def test_reconhecimento_sem_mao_e_incerto(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    assert ctrl.processar_quadro(None).texto_video == "MOSTRE UM SINAL"
    ctrl._modelo.probabilidades = np.array([0.4, 0.3, 0.3])
    r = ctrl.processar_quadro(vetor_de_duas_maos(rng))
    assert r.nivel is Nivel.INCERTO and r.texto_video.startswith("INCERTO")


def test_texto_do_video_e_so_ascii(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    ctrl._modelo = type(
        "M",
        (),
        {
            "classes_": np.array(["a", "b", "c"]),
            "predict_proba": lambda self, X: np.tile([0.9, 0.05, 0.05], (len(X), 1)),
        },
    )()
    assert ctrl.processar_quadro(vetor_de_duas_maos(rng)).texto_video.isascii()


def test_listar_sinais_marca_treinados_so_se_modelo_esta_atualizado(
    ctrl, rng, monkeypatch
):
    _preparar_modelo(ctrl, rng)
    monkeypatch.setattr(
        "kara_sygna.aplicacao.controlador.classes_do_modelo", lambda _c: {"a"}
    )
    assert [(s.palavra, s.treinado) for s in ctrl.listar_sinais()] == [("a", True)]
    os.utime(ctrl.caminho_modelo, (1, 1))
    assert [s.treinado for s in ctrl.listar_sinais()] == [False]


def test_apagar_sinal_remove_modelo_antigo(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    mensagem = ctrl.apagar_sinal("a")
    assert (
        "apagado" in mensagem
        and not ctrl.caminho_modelo.exists()
        and ctrl.listar_sinais() == []
    )


def test_treinar_sem_dados(ctrl):
    with pytest.raises(SemDadosError):
        ctrl.treinar()


def test_criar_reproducao(ctrl, rng):
    with pytest.raises(SemDadosError):
        ctrl.repositorio.garantir_arquivo() or ctrl.criar_reproducao("x")
    ctrl.iniciar_ensino("a")
    ctrl.processar_quadro(vetor_de_uma_mao(rng))
    ctrl.parar()
    reproducao = ctrl.criar_reproducao("a")
    assert (
        len(reproducao.vetores) == 1
    )  # original + espelho viram um quadro com as duas mãos


def test_combinar_espelhos_junta_so_pares_de_uma_mao(rng):
    um = vetor_de_uma_mao(rng, "dir")
    linhas = linhas_para_gravar(um)
    duas = vetor_de_duas_maos(rng)
    resultado = combinar_espelhos([*linhas, duas, linhas[0]])
    assert len(resultado) == 3
    assert np.count_nonzero(resultado[0][:63]) and np.count_nonzero(resultado[0][63:])
    assert np.array_equal(resultado[1], duas)
    assert np.array_equal(resultado[2], linhas[0])  # sem par: passa sem alteração


def test_reproducao_avanca_em_ciclo_e_gira():
    r = Reproducao("oi", [np.zeros(126), np.ones(126)])
    assert r.legenda == "oi - exemplo 1/2" and r.legenda.isascii()
    r.avancar()
    r.avancar()
    assert r.indice == 0 and r.angulo > 0
    with pytest.raises(ValueError):
        Reproducao("x", [])
