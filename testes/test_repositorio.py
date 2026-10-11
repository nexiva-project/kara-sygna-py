"""Testes do CSV de exemplos e da migração."""

import numpy as np
import pandas as pd
import pytest

from kara_sygna.dados import migracao
from kara_sygna.dados.repositorio import (
    DadosInvalidosError,
    RepositorioSinais,
    cabecalho,
)
from kara_sygna.visao.caracteristicas import nomes_features
from testes.conftest import vetor_de_duas_maos, vetor_de_uma_mao


@pytest.fixture
def repositorio(tmp_path):
    repo = RepositorioSinais(tmp_path / "dados.csv")
    repo.garantir_arquivo()
    return repo


def test_cria_arquivo_com_cabecalho(repositorio):
    assert list(pd.read_csv(repositorio.caminho).columns) == cabecalho()
    assert repositorio.contagem_por_sinal() == {}


def test_uma_mao_grava_duas_linhas_e_duas_maos_uma(repositorio, rng):
    assert repositorio.anexar_quadro("oi", 1, vetor_de_uma_mao(rng)) == 2
    assert repositorio.anexar_quadro("tchau", 2, vetor_de_duas_maos(rng)) == 1
    assert repositorio.contagem_por_sinal() == {"oi": 2, "tchau": 1}


def test_carregar_devolve_x_y_e_grupos(repositorio, rng):
    repositorio.anexar_quadro("oi", 7, vetor_de_duas_maos(rng))
    dados = repositorio.carregar()
    assert (
        dados.X.shape == (1, 126)
        and dados.y.tolist() == ["oi"]
        and dados.grupos.tolist() == [7]
    )
    assert dados.classes == ["oi"]


def test_vetor_de_tamanho_errado_e_recusado(repositorio):
    with pytest.raises(ValueError):
        repositorio.anexar_quadro("oi", 1, [0.0] * 10)


def test_apagar_sinal(repositorio, rng):
    repositorio.anexar_quadro("a", 1, vetor_de_duas_maos(rng))
    repositorio.anexar_quadro("b", 2, vetor_de_duas_maos(rng))
    assert repositorio.apagar_sinal("a") == 1
    assert repositorio.contagem_por_sinal() == {"b": 1}


def test_exemplos_do_sinal_mantem_a_ordem(repositorio, rng):
    vetores = [vetor_de_duas_maos(rng) for _ in range(3)]
    for v in vetores:
        repositorio.anexar_quadro("a", 1, v)
    assert np.allclose(repositorio.exemplos_do_sinal("a"), np.round(vetores, 6))
    assert repositorio.exemplos_do_sinal("inexistente").shape == (0, 126)


def _csv_antigo(caminho, rotulos):
    tabela = pd.DataFrame(
        np.random.default_rng(0).random((len(rotulos), 126)), columns=nomes_features()
    )
    tabela.insert(0, "rotulo", rotulos)
    tabela.to_csv(caminho, index=False)


def test_csv_sem_rodada_pede_migracao(tmp_path):
    _csv_antigo(tmp_path / "d.csv", ["a"] * 4)
    with pytest.raises(DadosInvalidosError, match="formato antigo"):
        RepositorioSinais(tmp_path / "d.csv").carregar()


def test_migracao_acrescenta_rodada_faz_backup_e_preserva_dados(tmp_path):
    caminho = tmp_path / "d.csv"
    _csv_antigo(caminho, ["a"] * 20 + ["b"] * 20)
    antes = pd.read_csv(caminho)

    backup = migracao.migrar_csv_legado(caminho)

    assert backup is not None and backup.exists()
    pd.testing.assert_frame_equal(pd.read_csv(backup), antes)
    dados = RepositorioSinais(caminho).carregar()
    assert len(dados) == 40 and len(set(dados.grupos)) == 10  # 5 blocos por sinal
    assert migracao.migrar_csv_legado(caminho) is None  # idempotente


def test_migracao_nao_mexe_na_data_do_arquivo(tmp_path):
    import os

    caminho = tmp_path / "d.csv"
    _csv_antigo(caminho, ["a"] * 10)
    os.utime(caminho, (1_000_000_000, 1_000_000_000))
    migracao.migrar_csv_legado(caminho)
    assert caminho.stat().st_mtime == 1_000_000_000


def test_inferir_rodadas_nao_repete_ids_entre_sinais():
    rotulos = np.array(["a"] * 10 + ["b"] * 10 + ["a"] * 10)
    rodadas = migracao.inferir_rodadas(rotulos, blocos_por_sinal=5)
    assert len(set(rodadas[rotulos == "a"]) & set(rodadas[rotulos == "b"])) == 0
    assert len(set(rodadas)) == 10


def test_inferir_rodadas_com_poucas_linhas():
    assert migracao.inferir_rodadas(np.array(["a", "a"])).tolist() == [1, 1]


def test_copia_arquivos_antigos_sem_apagar_os_originais(tmp_path):
    raiz, destino = tmp_path / "raiz", tmp_path / "arm"
    (raiz / "src").mkdir(parents=True)
    (raiz / "dados_sinais.csv").write_text("x")
    (raiz / "src" / "modelo_sinais.pkl").write_text("y")
    copiados = migracao.copiar_arquivos_legados(raiz, destino)
    assert {p.name for p in copiados} == {"dados_sinais.csv", "modelo_sinais.pkl"}
    assert (raiz / "dados_sinais.csv").exists()
    assert migracao.copiar_arquivos_legados(raiz, destino) == []  # não sobrescreve
