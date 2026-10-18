"""Testes de rotação, aumento de dados e treinamento."""

import numpy as np
import pytest

from kara_sygna import config
from kara_sygna.dados.repositorio import ConjuntoDados
from kara_sygna.ia import treinamento
from kara_sygna.ia.aumento import expandir_treino, numero_de_variantes
from kara_sygna.ia.geometria import matrizes_rotacao
from kara_sygna.ia.modelo_io import (
    ModeloInvalidoError,
    carregar_modelo,
    modelo_esta_atualizado,
    validar_modelo,
)
from kara_sygna.ia.sequencia import total_features_janela
from kara_sygna.visao.quadro import TOTAL_FEATURES_QUADRO
from testes.conftest import gravacao_falsa, vetor_de_duas_maos, vetor_de_uma_mao


def test_rotacao_nula_e_identidade_e_matrizes_sao_ortogonais():
    m = matrizes_rotacao([0, 30, -45], [0, 10, 20], [0, 5, 15])
    assert np.allclose(m[0], np.eye(3))
    for matriz in m:
        assert np.allclose(matriz @ matriz.T, np.eye(3), atol=1e-12)


def test_aumento_multiplica_linhas_e_mantem_rotulos(rng):
    X = np.vstack([vetor_de_duas_maos(rng) for _ in range(6)])
    y = np.array(list("aabbcc"))
    X2, y2 = expandir_treino(X, y, 3, np.random.default_rng(0))
    assert X2.shape == (24, 126) and y2.shape == (24,)
    assert np.allclose(X2[:6], X) and (y2[6:12] == y).all()


def test_aumento_nao_cria_mao_que_nao_existia_e_mantem_o_pulso_na_origem(rng):
    X = np.vstack([vetor_de_uma_mao(rng, "dir") for _ in range(5)])
    X2, _ = expandir_treino(X, np.arange(5), 4, np.random.default_rng(0))
    assert np.all(X2[:, :63] == 0)
    assert np.allclose(X2[:, 63:66], 0)


def test_aumento_so_mexe_na_forma_das_maos_das_janelas(rng):
    X = np.tile(np.arange(total_features_janela(), dtype=float), (4, 1))
    X[:, :126] = vetor_de_duas_maos(rng)
    X2, _ = expandir_treino(X, np.arange(4), 2, np.random.default_rng(0))
    assert X2.shape == (12, total_features_janela())
    assert np.array_equal(X2[:, 126:], np.tile(X[0, 126:], (12, 1)))


def test_aumento_zero_variantes_devolve_o_original(rng):
    X = np.vstack([vetor_de_duas_maos(rng)])
    X2, y2 = expandir_treino(X, np.array(["a"]), 0)
    assert X2 is X or np.array_equal(X2, X)
    assert y2.tolist() == ["a"]


def test_numero_de_variantes_respeita_limites():
    assert numero_de_variantes(10) == config.MAX_VARIANTES_AUMENTO
    assert numero_de_variantes(10_000_000) == config.MIN_VARIANTES_AUMENTO


def _dados_sinteticos(
    rng, rodadas_por_classe=3, por_rodada=20, classes=("a", "b", "c")
):
    """Quadros gravados: cada classe é uma mão diferente; cada rodada, uma gravação."""
    X, y, g = [], [], []
    identificador = 0
    for k, classe in enumerate(classes):
        forma = vetor_de_duas_maos(np.random.default_rng(100 + k))
        for _ in range(rodadas_por_classe):
            identificador += 1
            X.append(gravacao_falsa(forma, rng, por_rodada))
            y += [classe] * por_rodada
            g += [identificador] * por_rodada
    return ConjuntoDados(np.vstack(X), np.array(y), np.array(g))


@pytest.fixture
def config_rapida(monkeypatch):
    monkeypatch.setattr(config, "ARVORES", 10)


def test_treinar_salva_modelo_e_resumo(tmp_path, rng, config_rapida):
    caminho = tmp_path / "m.pkl"
    resultado = treinamento.treinar(_dados_sinteticos(rng), caminho)
    assert caminho.exists() and caminho.with_suffix(".json").exists()
    assert resultado.classes == ["a", "b", "c"] and resultado.rodadas == 9
    assert (
        resultado.acuracia_validacao is not None and resultado.acuracia_validacao > 0.9
    )
    modelo = carregar_modelo(caminho)
    assert list(modelo.classes_) == ["a", "b", "c"] and modelo.n_jobs == 1


def test_treinar_com_um_sinal_so(tmp_path, rng, config_rapida):
    from kara_sygna.ia.reconhecimento import prever

    caminho = tmp_path / "m.pkl"
    resultado = treinamento.treinar(_dados_sinteticos(rng, classes=("a",)), caminho)
    assert resultado.classes == ["a"] and resultado.acuracia_validacao is None
    modelo = carregar_modelo(caminho)
    predicao = prever(modelo, np.zeros(modelo.n_features_in_))
    assert predicao.sinal == "a" and predicao.aceito


def test_treinar_com_csv_vazio(tmp_path):
    vazio = ConjuntoDados(
        np.empty((0, TOTAL_FEATURES_QUADRO)), np.array([], dtype=str), np.array([])
    )
    with pytest.raises(treinamento.TreinoImpossivelError, match="vazio"):
        treinamento.treinar(vazio, tmp_path / "m.pkl")


def test_gravacoes_curtas_demais_nao_treinam(tmp_path, rng, config_rapida):
    dados = _dados_sinteticos(rng, por_rodada=config.QUADROS_POR_JANELA - 1)
    with pytest.raises(treinamento.TreinoImpossivelError, match="curtas demais"):
        treinamento.treinar(dados, tmp_path / "m.pkl")


def test_com_uma_rodada_por_sinal_pula_a_avaliacao_mas_treina(
    tmp_path, rng, config_rapida
):
    resultado = treinamento.treinar(
        _dados_sinteticos(rng, rodadas_por_classe=1), tmp_path / "m.pkl"
    )
    assert (
        resultado.acuracia_validacao is None
        and "Avaliação pulada" in resultado.relatorio
    )
    assert (tmp_path / "m.pkl").exists()


def test_sinal_com_uma_rodada_fica_fora_so_da_avaliacao(tmp_path, rng, config_rapida):
    base = _dados_sinteticos(rng, classes=("a", "b"))
    extra = _dados_sinteticos(rng, rodadas_por_classe=1, classes=("z",))
    juntos = ConjuntoDados(
        np.vstack([base.X, extra.X]),
        np.concatenate([base.y, extra.y]),
        np.concatenate([base.grupos, extra.grupos + 1000]),
    )
    resultado = treinamento.treinar(juntos, tmp_path / "m.pkl")
    assert resultado.acuracia_validacao is not None
    assert "Fora da avaliação" in resultado.relatorio and "z" in resultado.relatorio
    assert "z" in carregar_modelo(tmp_path / "m.pkl").classes_


def test_validar_modelo_recusa_incompativeis():
    class SemProba: ...

    class OutroTamanho:
        n_features_in_ = 10

        def predict_proba(self, X): ...

    with pytest.raises(ModeloInvalidoError):
        validar_modelo(SemProba())
    with pytest.raises(ModeloInvalidoError, match="espera 10"):
        validar_modelo(OutroTamanho())


def test_modelo_atualizado_compara_datas(tmp_path):
    import os

    csv, pkl = tmp_path / "d.csv", tmp_path / "m.pkl"
    csv.write_text("x")
    pkl.write_text("y")
    os.utime(csv, (100, 100))
    os.utime(pkl, (200, 200))
    assert modelo_esta_atualizado(pkl, csv)
    os.utime(csv, (300, 300))
    assert not modelo_esta_atualizado(pkl, csv)
    assert not modelo_esta_atualizado(tmp_path / "nao.pkl", csv)
