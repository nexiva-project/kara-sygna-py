"""Testes das regras da aplicação (sem Tk, câmera ou MediaPipe)."""

import os

import numpy as np
import pytest

from kara_sygna import config
from kara_sygna.aplicacao.controlador import ControladorSinais, Modo, Nivel
from kara_sygna.aplicacao.erros import (
    ModeloAusenteError,
    ModeloDesatualizadoError,
    PalavraInvalidaError,
    SemDadosError,
)
from kara_sygna.aplicacao.previsualizacao import Reproducao, completar_com_espelho
from kara_sygna.dados.repositorio import TIPO_PRE, RepositorioSinais
from kara_sygna.ia.pre_sinais import ModeloPreSinais
from kara_sygna.visao.caracteristicas import IDX_DIR, IDX_ESQ
from testes.conftest import quadro_falso, vetor_de_duas_maos, vetor_de_uma_mao
from testes.test_filtro_quadros import Relogio
from testes.test_reconhecimento import ModeloFalso


def _quadro(rng, uma_mao=False):
    """Vetor de quadro (223 valores) com uma ou duas mãos."""
    return quadro_falso(vetor_de_uma_mao(rng) if uma_mao else vetor_de_duas_maos(rng))


@pytest.fixture
def ctrl(tmp_path, monkeypatch):
    # A maioria dos testes usa todos os quadros; os modos de escolha de
    # quadros têm testes próprios.
    monkeypatch.setattr(config, "MODO_QUADROS", "fixo")
    monkeypatch.setattr(config, "USAR_1_A_CADA_QUADROS", 1)
    return ControladorSinais(
        RepositorioSinais(tmp_path / "d.csv"),
        tmp_path / "m.pkl",
        carregador_modelo=lambda _caminho: ModeloFalso([0.9, 0.05, 0.05]),
    )


def _preparar_modelo(ctrl, rng):
    ctrl.iniciar_ensino("a")
    ctrl.processar_quadro(_quadro(rng))
    ctrl.parar()
    ctrl.caminho_modelo.write_text("modelo")
    os.utime(
        ctrl.caminho_modelo, (os.path.getmtime(ctrl.repositorio.caminho) + 10,) * 2
    )


def test_palavra_vazia_e_recusada(ctrl):
    with pytest.raises(PalavraInvalidaError):
        ctrl.iniciar_ensino("   ")
    assert ctrl.modo is Modo.PARADO


def test_ensino_grava_os_quadros_em_sequencia(ctrl, rng):
    ctrl.iniciar_ensino("  OI ")
    r1 = ctrl.processar_quadro(_quadro(rng, uma_mao=True))
    r2 = ctrl.processar_quadro(_quadro(rng))
    r3 = ctrl.processar_quadro(quadro_falso(None))  # sem mão no meio do sinal
    ctrl.processar_quadro(_quadro(rng))
    assert ctrl.total_gravado == 4 and r1.nivel is Nivel.ENSINO
    assert r2.texto_video == "ENSINANDO: oi (2)"
    assert "Mostre as mãos" in r3.status and "Mostre as mãos" not in r2.status
    assert ctrl.parar().startswith("“oi”: 4 quadros gravados")
    assert ctrl.repositorio.contagem_por_sinal() == {"oi": 4}
    assert ctrl.repositorio.carregar().X.shape[0] == 4


def test_ensino_nao_grava_o_comeco_e_o_fim_sem_mao(ctrl, rng):
    ctrl.iniciar_ensino("oi")
    for _ in range(5):
        ctrl.processar_quadro(quadro_falso(None))  # antes de levantar a mão
    ctrl.processar_quadro(_quadro(rng))
    ctrl.processar_quadro(_quadro(rng))
    for _ in range(5):
        ctrl.processar_quadro(quadro_falso(None))  # depois de baixar a mão
    ctrl.parar()
    assert ctrl.repositorio.contagem_por_sinal() == {"oi": 2}
    assert ctrl.repositorio.carregar().X.shape[0] == 2


def test_ensino_usa_1_a_cada_n_quadros(ctrl, rng, monkeypatch):
    monkeypatch.setattr(config, "USAR_1_A_CADA_QUADROS", 2)  # modo "fixo"
    ctrl.iniciar_ensino("oi")
    for _ in range(10):
        ctrl.processar_quadro(_quadro(rng))
    ctrl.parar()
    assert ctrl.repositorio.contagem_por_sinal() == {"oi": 5}


def test_reconhecimento_usa_1_a_cada_n_quadros_sem_repetir_confirmacao(
    ctrl, rng, monkeypatch
):
    monkeypatch.setattr(config, "USAR_1_A_CADA_QUADROS", 2)  # modo "fixo"
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    vetor = _quadro(rng)
    total = 2 * (config.QUADROS_POR_JANELA + 10)
    respostas = [ctrl.processar_quadro(vetor) for _ in range(total)]
    assert [r.sinal_confirmado for r in respostas].count("a") == 1
    # o quadro pulado repete o que está na tela
    assert respostas[-1].texto_video == respostas[-2].texto_video


def test_ensino_grava_em_lotes_e_o_resto_ao_parar(ctrl, rng):
    lote = config.QUADROS_POR_LOTE_GRAVACAO
    ctrl.iniciar_ensino("oi")
    for _ in range(lote + 2):
        ctrl.processar_quadro(_quadro(rng))
    assert ctrl.repositorio.contagem_por_sinal() == {"oi": lote}
    ctrl.parar()
    assert ctrl.repositorio.contagem_por_sinal() == {"oi": lote + 2}


def test_quadro_que_nao_pode_ser_analisado_nao_grava(ctrl):
    ctrl.iniciar_ensino("oi")
    assert ctrl.processar_quadro(None).texto_video == ""
    ctrl.parar()
    assert ctrl.repositorio.contagem_por_sinal() == {}


def test_modo_parado_ignora_quadros(ctrl, rng):
    assert ctrl.processar_quadro(_quadro(rng)).texto_video == ""


def test_rodadas_diferentes_a_cada_ensino(ctrl, rng, monkeypatch):
    ids = iter([111, 222])
    monkeypatch.setattr(
        "kara_sygna.aplicacao.controlador.novo_id_rodada", lambda: next(ids)
    )
    for _ in range(2):
        ctrl.iniciar_ensino("a")
        ctrl.processar_quadro(_quadro(rng))
        ctrl.parar()
    assert sorted(set(ctrl.repositorio.carregar().grupos.tolist())) == [111, 222]


def test_reconhecer_exige_dados_modelo_e_treino_atual(ctrl, rng):
    with pytest.raises(SemDadosError):
        ctrl.iniciar_reconhecimento()
    ctrl.iniciar_ensino("a")
    ctrl.processar_quadro(_quadro(rng))
    ctrl.parar()
    with pytest.raises(ModeloAusenteError):
        ctrl.iniciar_reconhecimento()
    ctrl.caminho_modelo.write_text("x")
    os.utime(ctrl.caminho_modelo, (1, 1))
    with pytest.raises(ModeloDesatualizadoError):
        ctrl.iniciar_reconhecimento()


def test_reconhecimento_espera_a_janela_encher(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    vetor = _quadro(rng)
    respostas = [ctrl.processar_quadro(vetor) for _ in range(config.QUADROS_POR_JANELA)]
    assert all(r.texto_video == "PREPARANDO..." for r in respostas[:-1])
    assert respostas[-1].texto_video.startswith("SINAL: a")
    assert ctrl._modelo.entrada.shape[0] == 1  # uma previsão por quadro, sem rotações


def test_reconhecimento_confirma_sinal_e_ignora_neutro(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    vetor = _quadro(rng)
    total = config.QUADROS_POR_JANELA + 10
    confirmados = [ctrl.processar_quadro(vetor).sinal_confirmado for _ in range(total)]
    assert confirmados.count("a") == 1

    ctrl.limpar_reconhecimento()
    ctrl._modelo.classes_ = np.array(["neutro", "b", "c"])
    assert all(ctrl.processar_quadro(vetor).sinal_confirmado is None for _ in range(10))


def test_reconhecimento_sem_mao_e_incerto(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    assert ctrl.processar_quadro(None).texto_video == "MOSTRE UM SINAL"
    assert ctrl.processar_quadro(quadro_falso(None)).texto_video == "MOSTRE UM SINAL"
    ctrl._modelo.probabilidades = np.array([0.4, 0.3, 0.3])
    for _ in range(config.QUADROS_POR_JANELA):
        r = ctrl.processar_quadro(_quadro(rng))
    assert r.nivel is Nivel.INCERTO and r.texto_video.startswith("INCERTO")


def test_iniciar_reconhecimento_esvazia_a_janela(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    for _ in range(config.QUADROS_POR_JANELA):
        ctrl.processar_quadro(_quadro(rng))
    ctrl.parar()
    ctrl.iniciar_reconhecimento()
    assert ctrl.processar_quadro(_quadro(rng)).texto_video == "PREPARANDO..."


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
    for _ in range(config.QUADROS_POR_JANELA):
        resposta = ctrl.processar_quadro(_quadro(rng))
    assert resposta.texto_video.startswith("SINAL") and resposta.texto_video.isascii()


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


def test_apagar_sinal_durante_o_ensino_grava_antes_e_para(ctrl, rng):
    ctrl.iniciar_ensino("b")
    ctrl.processar_quadro(_quadro(rng))
    ctrl.apagar_sinal("b")
    assert ctrl.modo is Modo.PARADO and ctrl.repositorio.contagem_por_sinal() == {}


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
    relogio = Relogio()
    ctrl._relogio = relogio
    ctrl.iniciar_ensino("a")
    ctrl.processar_quadro(_quadro(rng, uma_mao=True))
    relogio.agora = 2.0
    ctrl.processar_quadro(quadro_falso(None))  # sem mão por muito tempo: fica fora
    relogio.agora = 4.0
    ctrl.processar_quadro(_quadro(rng))
    ctrl.parar()
    reproducao = ctrl.criar_reproducao("a")
    assert len(reproducao.vetores) == 2


def test_ensino_preenche_a_mao_que_sumiu_por_pouco_tempo(ctrl, rng):
    relogio = Relogio()
    ctrl._relogio = relogio
    ctrl.iniciar_ensino("a")
    for t, quadro in enumerate(
        [_quadro(rng), quadro_falso(None), quadro_falso(None), _quadro(rng)]
    ):
        relogio.agora = t / 30
        ctrl.processar_quadro(quadro)
    ctrl.parar()
    assert len(ctrl.criar_reproducao("a").vetores) == 4  # os 2 do meio preenchidos


def test_completar_com_espelho_so_muda_quadros_de_uma_mao(rng):
    um = vetor_de_uma_mao(rng, "dir")
    duas = vetor_de_duas_maos(rng)
    resultado = completar_com_espelho([um, duas, np.zeros(126)])
    assert len(resultado) == 3
    assert np.array_equal(resultado[0][IDX_DIR], um[IDX_DIR])
    espelho = resultado[0][IDX_ESQ].reshape(21, 3)
    assert np.allclose(espelho[:, 0], -um[IDX_DIR].reshape(21, 3)[:, 0])
    assert np.array_equal(resultado[1], duas)
    assert np.array_equal(resultado[2], np.zeros(126))


def test_reproducao_avanca_em_ciclo_e_gira():
    r = Reproducao("oi", [np.zeros(126), np.ones(126)])
    assert r.legenda == "oi - exemplo 1/2" and r.legenda.isascii()
    r.avancar()
    r.avancar()
    assert r.indice == 0 and r.angulo > 0
    with pytest.raises(ValueError):
        Reproducao("x", [])


def test_modo_mudanca_nao_grava_mao_parada(ctrl, rng, monkeypatch):
    monkeypatch.setattr(config, "MODO_QUADROS", "mudanca")
    monkeypatch.setattr(config, "PARADO_USAR_A_CADA_S", 0)
    parado = _quadro(rng)
    ctrl.iniciar_ensino("oi")
    for _ in range(20):
        ctrl.processar_quadro(parado)
    for _ in range(3):
        ctrl.processar_quadro(_quadro(rng))  # mudou: grava
    ctrl.parar()
    assert ctrl.repositorio.contagem_por_sinal() == {"oi": 4}


def test_modo_mudanca_confirma_sinal_parado(ctrl, rng, monkeypatch):
    """Com a mão parada a janela quase não recebe quadros, mas cada quadro vota."""
    monkeypatch.setattr(config, "MODO_QUADROS", "mudanca")
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    for _ in range(config.QUADROS_POR_JANELA):
        ctrl.processar_quadro(_quadro(rng))  # quadros diferentes enchem a janela
    parado = _quadro(rng)
    confirmados = [ctrl.processar_quadro(parado).sinal_confirmado for _ in range(10)]
    assert confirmados.count("a") == 1


# ---------------------------------------------------------------- pré-sinal


def test_ensino_com_pre_sinal_grava_as_duas_etapas(ctrl, rng):
    ctrl.iniciar_ensino("S", TIPO_PRE)
    for _ in range(3):
        ctrl.processar_quadro(_quadro(rng, uma_mao=True))
    assert (
        "PRE-SINAL: S" in ctrl.processar_quadro(_quadro(rng, uma_mao=True)).texto_video
    )
    assert ctrl.parar().startswith("Pré-sinal “S”: 4 quadros")
    ctrl.iniciar_ensino("Carro", pre_sinal="S")
    assert "SINAL: carro" in ctrl.processar_quadro(_quadro(rng)).texto_video
    ctrl.parar()

    sinais = ctrl.listar_sinais()
    assert [(s.chave, s.exemplos, s.pre_sinal) for s in sinais] == [
        ("pre:S", 4, ""),
        ("carro", 1, "S"),
    ]
    assert ctrl.criar_reproducao("pre:S").legenda.startswith("S ")
    assert "Pré-sinal" in ctrl.apagar_sinal("pre:S")
    assert [s.chave for s in ctrl.listar_sinais()] == ["carro"]


def test_pre_sinal_vazio_e_recusado(ctrl):
    with pytest.raises(PalavraInvalidaError, match="pré-sinal"):
        ctrl.iniciar_ensino("  ", TIPO_PRE)


class _ClassificadorFalso:
    """IA de pré-sinais falsa: diz "S" enquanto ``mostrando`` for True."""

    classes_ = np.array(["", "S"])

    def __init__(self):
        self.mostrando = False

    def predict_proba(self, X):
        linha = [0.0, 1.0] if self.mostrando else [1.0, 0.0]
        return np.tile(linha, (len(X), 1))


def _reconhecer_com_pre(ctrl, rng):
    """Reconhecimento com sinal "a" (sempre 90%) que exige o pré-sinal S."""
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento(espera_sinal_s=3)
    classificador = _ClassificadorFalso()
    ctrl._pre = ModeloPreSinais(classificador, {"a": "S"})
    # 75%: passa da confiança mínima, mas não da certeza para dispensar o S.
    ctrl._modelo = ModeloFalso([0.75, 0.2, 0.05])
    relogio = Relogio()
    ctrl._relogio = relogio
    return classificador, relogio


def _confirmados(ctrl, rng, n):
    return [ctrl.processar_quadro(_quadro(rng)).sinal_confirmado for _ in range(n)]


def test_sinal_com_pre_sinal_so_vale_depois_do_pre_sinal(ctrl, rng):
    classificador, _ = _reconhecer_com_pre(ctrl, rng)
    assert not any(_confirmados(ctrl, rng, 30))  # sem o S, "a" não é permitido

    classificador.mostrando = True
    _confirmados(ctrl, rng, config.VOTOS_MINIMOS)
    assert ctrl.pre_ativo == "S"
    resposta = ctrl.processar_quadro(_quadro(rng))
    assert "PRE-SINAL" not in resposta.texto_video  # não aparece na tela

    classificador.mostrando = False
    assert _confirmados(ctrl, rng, 30).count("a") == 1
    assert ctrl.pre_ativo is None  # depois do sinal, espera outro pré-sinal


def test_pre_sinal_expira_sem_sinal(ctrl, rng):
    classificador, relogio = _reconhecer_com_pre(ctrl, rng)
    classificador.mostrando = True
    _confirmados(ctrl, rng, config.VOTOS_MINIMOS)
    assert ctrl.pre_ativo == "S"
    relogio.agora = 3.5
    ctrl.processar_quadro(_quadro(rng))
    assert ctrl.pre_ativo is None


def test_diagnostico_registra_cada_quadro_e_resume(ctrl, rng, tmp_path):
    import csv

    classificador, _ = _reconhecer_com_pre(ctrl, rng)
    assert ctrl.diagnostico is not None
    caminho = ctrl.diagnostico.caminho
    _confirmados(ctrl, rng, config.QUADROS_POR_JANELA + 2)  # sem o S: não permitido
    classificador.mostrando = True
    _confirmados(ctrl, rng, config.VOTOS_MINIMOS)
    classificador.mostrando = False
    _confirmados(ctrl, rng, 20)
    resumo = ctrl.diagnostico.fechar()
    ctrl.diagnostico = None
    with open(caminho, encoding="utf-8") as arquivo:
        linhas = list(csv.DictReader(arquivo))
    assert len(linhas) == config.QUADROS_POR_JANELA + 2 + config.VOTOS_MINIMOS + 20
    assert linhas[0]["motivo"] == "preparando"  # a janela ainda não encheu
    linha = linhas[config.QUADROS_POR_JANELA]
    assert linha["motivo"] == "falta_pre" and linha["top1"] == "a"
    assert linha["top1_pct"] == "75" and linha["mao_dir"] == "1"
    assert any(linha["escrito"] == "a" for linha in linhas)
    assert "Escrito: a×1" in resumo and "Pré-sinais vistos: S×1" in resumo
    assert "falta_pre" in resumo


def test_sinal_com_muita_certeza_vale_sem_o_pre_sinal(ctrl, rng):
    _reconhecer_com_pre(ctrl, rng)
    ctrl._modelo = ModeloFalso([0.9, 0.05, 0.05])  # 90%: passa dos 85%
    confirmados = _confirmados(ctrl, rng, 30)
    assert confirmados.count("a") == 1 and ctrl.pre_ativo is None


def test_pre_sinal_visto_nao_bloqueia_outro_sinal(ctrl, rng):
    """Depois do S (carro), a casa (pré-sinal B) sai na hora, sem esperar."""
    classificador, _ = _reconhecer_com_pre(ctrl, rng)
    ctrl._pre = ModeloPreSinais(classificador, {"a": "S", "b": "B"})
    classificador.mostrando = True
    _confirmados(ctrl, rng, config.VOTOS_MINIMOS)
    assert ctrl.pre_ativo == "S"
    classificador.mostrando = False
    ctrl._modelo = ModeloFalso([0.05, 0.9, 0.05])  # agora a IA acha "b" com 90%
    assert _confirmados(ctrl, rng, 20).count("b") == 1


def test_pre_sinal_visto_baixa_o_limite_so_dos_sinais_dele(ctrl, rng):
    classificador, _ = _reconhecer_com_pre(ctrl, rng)
    ctrl._pre = ModeloPreSinais(classificador, {"a": "S", "b": "B"})
    assert ctrl._limite_do_sinal("a") == config.CONFIANCA_SEM_PRE_SINAL
    assert ctrl._limite_do_sinal("c") == config.CONFIANCA_MINIMA  # sem pré-sinal
    classificador.mostrando = True
    _confirmados(ctrl, rng, config.VOTOS_MINIMOS)
    assert ctrl._limite_do_sinal("a") == config.CONFIANCA_MINIMA
    assert ctrl._limite_do_sinal("b") == config.CONFIANCA_SEM_PRE_SINAL


def test_pre_sinal_nao_diferencia_maiuscula(ctrl, rng):
    ctrl.iniciar_ensino("s", TIPO_PRE)
    ctrl.processar_quadro(_quadro(rng, uma_mao=True))
    ctrl.parar()
    ctrl.iniciar_ensino(" S ", TIPO_PRE)
    ctrl.processar_quadro(_quadro(rng, uma_mao=True))
    ctrl.parar()
    ctrl.iniciar_ensino("carro", pre_sinal="s")
    ctrl.processar_quadro(_quadro(rng))
    ctrl.parar()
    assert ctrl.repositorio.contagem_por_sinal(TIPO_PRE) == {"S": 2}
    assert ctrl.repositorio.pre_sinais_dos_sinais() == {"carro": "S"}


def test_apagar_sinal_durante_o_reconhecimento_fecha_o_diagnostico(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    ctrl.processar_quadro(_quadro(rng))
    arquivo = ctrl.diagnostico
    ctrl.apagar_sinal("a")
    assert ctrl.diagnostico is None and arquivo._arquivo.closed


def test_ensinar_durante_o_reconhecimento_fecha_o_diagnostico(ctrl, rng):
    _preparar_modelo(ctrl, rng)
    ctrl.iniciar_reconhecimento()
    arquivo = ctrl.diagnostico
    ctrl.iniciar_ensino("b")
    assert ctrl.diagnostico is None and arquivo._arquivo.closed
    ctrl.parar()


def test_falta_pre_so_quando_a_confianca_fica_entre_o_minimo_e_o_limite(ctrl, rng):
    _reconhecer_com_pre(ctrl, rng)
    # 80% mas margem pequena (80 - 70 = 10%): é dúvida, não falta de pré-sinal.
    ctrl._modelo = ModeloFalso([0.80, 0.70, 0.0])
    _confirmados(ctrl, rng, config.QUADROS_POR_JANELA + 2)
    assert ctrl._diag["motivo"] == "incerto"
