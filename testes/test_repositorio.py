"""Testes do CSV de exemplos e da migração."""

import numpy as np
import pandas as pd
import pytest

from kara_sygna.dados import migracao
from kara_sygna.dados.repositorio import (
    TIPO_PRE,
    DadosInvalidosError,
    RepositorioSinais,
    cabecalho,
    novo_id_rodada,
)
from kara_sygna.visao.caracteristicas import nomes_features
from kara_sygna.visao.quadro import TOTAL_FEATURES_QUADRO
from testes.conftest import quadro_falso, vetor_de_duas_maos


@pytest.fixture
def repositorio(tmp_path):
    repo = RepositorioSinais(tmp_path / "dados.csv")
    repo.garantir_arquivo()
    return repo


def _quadros(rng, n):
    return [(i, quadro_falso(vetor_de_duas_maos(rng))) for i in range(n)]


def test_cria_arquivo_com_cabecalho(repositorio):
    colunas = list(pd.read_csv(repositorio.caminho).columns)
    assert colunas == cabecalho()
    assert colunas[:5] == ["rotulo", "rodada", "quadro", "tipo", "pre_sinal"]
    assert colunas[5].startswith("formato_v")
    assert len(colunas) == 6 + TOTAL_FEATURES_QUADRO
    assert repositorio.contagem_por_sinal() == {}


def test_pre_sinais_ficam_separados_dos_sinais(repositorio, rng):
    repositorio.anexar_quadros("S", 1, _quadros(rng, 4), tipo=TIPO_PRE)
    repositorio.anexar_quadros("carro", 2, _quadros(rng, 3), pre_sinal="S")
    repositorio.anexar_quadros("oi", 3, _quadros(rng, 2))
    assert repositorio.contagem_por_sinal() == {"carro": 3, "oi": 2}
    assert repositorio.contagem_por_sinal(TIPO_PRE) == {"S": 4}
    assert repositorio.pre_sinais_dos_sinais() == {"carro": "S"}
    assert len(repositorio.carregar()) == 5
    assert repositorio.carregar(TIPO_PRE).classes == ["S"]
    assert repositorio.exemplos_do_sinal("S", TIPO_PRE).shape == (4, 126)
    # Apagar o pré-sinal não apaga os sinais (e vice-versa).
    assert repositorio.apagar_sinal("S", TIPO_PRE) == 4
    assert repositorio.contagem_por_sinal() == {"carro": 3, "oi": 2}
    assert repositorio.pre_sinais_dos_sinais() == {"carro": "S"}


def test_grava_um_quadro_por_linha(repositorio, rng):
    assert repositorio.anexar_quadros("oi", 1, _quadros(rng, 3)) == 3
    assert repositorio.anexar_quadros("tchau", 2, _quadros(rng, 1)) == 1
    assert repositorio.anexar_quadros("tchau", 2, []) == 0
    assert repositorio.contagem_por_sinal() == {"oi": 3, "tchau": 1}


def test_carregar_devolve_x_y_e_grupos_na_ordem_dos_quadros(repositorio, rng):
    quadros = _quadros(rng, 3)
    repositorio.anexar_quadros("oi", 7, [quadros[2], quadros[0], quadros[1]])
    dados = repositorio.carregar()
    assert dados.X.shape == (3, TOTAL_FEATURES_QUADRO)
    assert dados.y.tolist() == ["oi"] * 3 and dados.grupos.tolist() == [7] * 3
    assert np.allclose(dados.X, np.round([q for _, q in quadros], 4))
    assert dados.classes == ["oi"]


def test_vetor_de_tamanho_errado_e_recusado_sem_gravar_nada(repositorio, rng):
    with pytest.raises(ValueError):
        repositorio.anexar_quadros("oi", 1, [*_quadros(rng, 1), (1, [0.0] * 10)])
    assert repositorio.contagem_por_sinal() == {}


def test_apagar_sinal(repositorio, rng):
    repositorio.anexar_quadros("a", 1, _quadros(rng, 1))
    repositorio.anexar_quadros("b", 2, _quadros(rng, 2))
    assert repositorio.apagar_sinal("a") == 1
    assert repositorio.contagem_por_sinal() == {"b": 2}


def test_exemplos_do_sinal_devolve_so_as_maos_e_pula_quadros_sem_mao(repositorio, rng):
    maos = [vetor_de_duas_maos(rng) for _ in range(2)]
    repositorio.anexar_quadros(
        "a",
        1,
        [
            (0, quadro_falso(maos[0])),
            (1, quadro_falso(None)),
            (2, quadro_falso(maos[1])),
        ],
    )
    assert np.allclose(repositorio.exemplos_do_sinal("a"), np.round(maos, 4))
    assert repositorio.exemplos_do_sinal("inexistente").shape == (0, 126)


def test_ids_de_rodada_nunca_se_repetem():
    ids = [novo_id_rodada() for _ in range(50)]
    assert len(set(ids)) == 50 and ids == sorted(ids)


def _csv_antigo(caminho, rotulos, com_rodada=True):
    tabela = pd.DataFrame(
        np.random.default_rng(0).random((len(rotulos), 126)), columns=nomes_features()
    )
    if com_rodada:
        tabela.insert(0, "rodada", 1)
    tabela.insert(0, "rotulo", rotulos)
    tabela.to_csv(caminho, index=False)


def test_csv_antigo_e_recusado_na_leitura(tmp_path):
    _csv_antigo(tmp_path / "d.csv", ["a"] * 4)
    with pytest.raises(DadosInvalidosError, match="formato antigo"):
        RepositorioSinais(tmp_path / "d.csv").carregar()


@pytest.mark.parametrize("com_rodada", [True, False])
def test_formato_antigo_vai_para_backup_com_o_modelo(tmp_path, com_rodada):
    csv, modelo = tmp_path / "dados.csv", tmp_path / "modelo.pkl"
    _csv_antigo(csv, ["a", "b"], com_rodada)
    original = csv.read_bytes()
    modelo.write_text("modelo antigo")
    modelo.with_suffix(".json").write_text("{}")

    backup = migracao.guardar_formato_antigo(csv, modelo)

    assert backup is not None and backup.read_bytes() == original
    assert not csv.exists() and not modelo.exists()
    nomes = sorted(p.name for p in tmp_path.iterdir())
    assert any(
        n.startswith("modelo_formato_antigo_") and n.endswith(".pkl") for n in nomes
    )
    assert any(
        n.startswith("modelo_formato_antigo_") and n.endswith(".json") for n in nomes
    )


def test_formato_atual_nao_e_mexido(repositorio, rng, tmp_path):
    repositorio.anexar_quadros("a", 1, _quadros(rng, 2))
    assert (
        migracao.guardar_formato_antigo(repositorio.caminho, tmp_path / "m.pkl") is None
    )
    assert repositorio.contagem_por_sinal() == {"a": 2}
    assert migracao.guardar_formato_antigo(tmp_path / "nao_existe.csv") is None


def test_copia_arquivos_antigos_sem_apagar_os_originais(tmp_path):
    raiz, destino = tmp_path / "raiz", tmp_path / "arm"
    (raiz / "src").mkdir(parents=True)
    (raiz / "dados_sinais.csv").write_text("x")
    (raiz / "src" / "modelo_sinais.pkl").write_text("y")
    copiados = migracao.copiar_arquivos_legados(raiz, destino)
    assert {p.name for p in copiados} == {"dados_sinais.csv", "modelo_sinais.pkl"}
    assert (raiz / "dados_sinais.csv").exists()
    assert migracao.copiar_arquivos_legados(raiz, destino) == []  # não sobrescreve


def test_pre_sinal_antigo_em_minuscula_junta_com_maiuscula(repositorio, rng):
    repositorio.anexar_quadros("s", 1, _quadros(rng, 2), tipo=TIPO_PRE)
    repositorio.anexar_quadros("S", 2, _quadros(rng, 3), tipo=TIPO_PRE)
    repositorio.anexar_quadros("carro", 3, _quadros(rng, 1), pre_sinal="s")
    assert repositorio.contagem_por_sinal(TIPO_PRE) == {"S": 5}
    assert repositorio.pre_sinais_dos_sinais() == {"carro": "S"}
    assert repositorio.apagar_sinal("s", TIPO_PRE) == 5


def test_regravar_sem_pre_sinal_tira_o_pre_sinal_antigo(repositorio, rng):
    repositorio.anexar_quadros("carro", 1, _quadros(rng, 2), pre_sinal="S")
    assert repositorio.pre_sinais_dos_sinais() == {"carro": "S"}
    repositorio.anexar_quadros("carro", 2, _quadros(rng, 2))  # sem pré-sinal
    assert repositorio.pre_sinais_dos_sinais() == {}
