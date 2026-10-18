"""Testes da IA de pré-sinais (forma de uma mão, de qualquer lado)."""

import numpy as np

from kara_sygna.dados.repositorio import ConjuntoDados
from kara_sygna.ia.pre_sinais import (
    caminho_modelo_pre,
    carregar_pre_sinais,
    maos_dos_quadros,
    treinar_pre_sinais,
)
from kara_sygna.visao.caracteristicas import FEATURES_POR_MAO, IDX_DIR, IDX_ESQ
from kara_sygna.visao.quadro import IDX_FORMA, espelhar_quadros
from testes.conftest import quadro_falso


def _quadro_uma_mao(forma_dir, lado="dir"):
    """Quadro com uma mão de forma ``forma_dir`` (63 valores, como mão direita)."""
    maos = np.zeros(126)
    maos[IDX_DIR] = forma_dir
    quadro = quadro_falso(maos)
    return espelhar_quadros(quadro) if lado == "esq" else quadro


def test_mao_esquerda_vira_direita(rng):
    forma = rng.normal(0, 1, FEATURES_POR_MAO)
    direita = maos_dos_quadros(_quadro_uma_mao(forma, "dir"))
    esquerda = maos_dos_quadros(_quadro_uma_mao(forma, "esq"))
    assert direita.shape == esquerda.shape == (1, FEATURES_POR_MAO)
    assert np.allclose(direita, esquerda)
    duas = quadro_falso(np.concatenate([forma, forma]))
    assert len(maos_dos_quadros(duas)) == 2
    assert len(maos_dos_quadros(quadro_falso(None))) == 0


def _gravacao(rng, base, n, rotulo):
    quadros = [
        _quadro_uma_mao(
            base + rng.normal(0, 0.05, FEATURES_POR_MAO), rng.choice(["esq", "dir"])
        )
        for _ in range(n)
    ]
    return np.array(quadros), np.array([rotulo] * n)


def test_treina_reconhece_e_salva(tmp_path, rng):
    forma_s, forma_outra = rng.normal(0, 1, (2, FEATURES_POR_MAO))
    X_pre, y_pre = _gravacao(rng, forma_s, 60, "S")
    X_sem, y_sem = _gravacao(rng, forma_outra, 60, "oi")
    X_com, y_com = _gravacao(rng, forma_s, 30, "carro")  # sinal do S: não é "nenhum"
    pre = ConjuntoDados(X_pre, y_pre, np.ones(60))
    sinais = ConjuntoDados(
        np.vstack([X_sem, X_com]), np.concatenate([y_sem, y_com]), np.ones(90)
    )
    caminho = caminho_modelo_pre(tmp_path / "m.pkl")
    assert caminho.name == "m_pre.pkl"

    modelo = treinar_pre_sinais(pre, sinais, {"carro": "S"}, caminho)
    assert modelo.pre_sinais == ["S"] and modelo.sinais_do_pre("S") == {"carro"}
    assert modelo.classificar(_quadro_uma_mao(forma_s, "esq"))[0] == "S"
    assert modelo.classificar(_quadro_uma_mao(forma_outra))[0] is None
    assert modelo.classificar(quadro_falso(None)) == (None, 0.0)

    carregado = carregar_pre_sinais(caminho)
    assert carregado is not None and carregado.sinal_para_pre == {"carro": "S"}

    vazio = ConjuntoDados(np.empty((0, 223)), np.array([], dtype=str), np.array([]))
    assert treinar_pre_sinais(vazio, sinais, {}, caminho) is None
    assert not caminho.exists() and carregar_pre_sinais(caminho) is None


def test_idx_forma_bate_com_as_maos():
    assert IDX_FORMA == {"esq": IDX_ESQ, "dir": IDX_DIR}


def test_nomes_dos_pre_sinais_fica_guardado_ate_o_arquivo_mudar(
    tmp_path, rng, monkeypatch
):
    from kara_sygna.ia import pre_sinais

    forma = rng.normal(0, 1, FEATURES_POR_MAO)
    X, y = _gravacao(rng, forma, 20, "S")
    caminho = tmp_path / "m_pre.pkl"
    vazio = ConjuntoDados(
        np.empty((0, X.shape[1])), np.array([], dtype=str), np.array([])
    )
    treinar_pre_sinais(ConjuntoDados(X, y, np.ones(20)), vazio, {}, caminho)
    assert pre_sinais.nomes_dos_pre_sinais(caminho) == ["S"]
    chamadas = []
    monkeypatch.setattr(pre_sinais, "carregar_pre_sinais", lambda c: chamadas.append(c))
    assert pre_sinais.nomes_dos_pre_sinais(caminho) == ["S"] and not chamadas
    assert pre_sinais.nomes_dos_pre_sinais(tmp_path / "nao_existe.pkl") == []
