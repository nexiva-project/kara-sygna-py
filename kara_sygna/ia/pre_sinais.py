"""Pré-sinais: formas de mão (como o "S") que vêm antes de alguns sinais.

Fluxo
-----
* **Ensino:** o pré-sinal é gravado sozinho, com UMA mão girando para todos
  os lados (frente, trás, lados). Depois, na mesma gravação, vem o sinal que
  começa com ele (ex.: "carro" começa com "S").
* **Treino:** esta IA aprende a forma de cada pré-sinal olhando uma mão de
  cada vez. A mão esquerda é espelhada para parecer uma direita, então o
  pré-sinal vale para qualquer mão.
* **Reconhecimento:** quando a IA vê um pré-sinal, o próximo sinal é
  escolhido só entre os sinais que começam com ele (veja
  :class:`kara_sygna.aplicacao.controlador.ControladorSinais`).

Para a IA saber quando uma mão NÃO está em nenhum pré-sinal, ela também
aprende exemplos "de nenhum": as mãos dos sinais sem pré-sinal (inclusive o
``neutro``). Sem esses exemplos, qualquer mão vira o único pré-sinal gravado.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from kara_sygna import config
from kara_sygna.dados.repositorio import ConjuntoDados
from kara_sygna.visao.caracteristicas import FEATURES_POR_MAO, TOTAL_LANDMARKS_MAO
from kara_sygna.visao.quadro import IDX_FORMA, TOTAL_FEATURES_QUADRO, maos_presentes

logger = logging.getLogger(__name__)

#: Rótulo dos exemplos de mão que não estão em nenhum pré-sinal.
NENHUM = ""
#: Máximo de exemplos "de nenhum" por exemplo de pré-sinal (equilíbrio).
_NENHUM_POR_PRE = 3


def caminho_modelo_pre(caminho_modelo: Path) -> Path:
    """Arquivo da IA de pré-sinais, ao lado do modelo dos sinais."""
    caminho_modelo = Path(caminho_modelo)
    return caminho_modelo.with_name(f"{caminho_modelo.stem}_pre{caminho_modelo.suffix}")


def maos_dos_quadros(quadros: np.ndarray) -> np.ndarray:
    """Separa cada mão presente nos quadros, todas "viradas" para a direita.

    Args:
        quadros: Matriz (n, 223).

    Returns:
        Matriz (m, 63): uma linha por mão presente. A mão esquerda é espelhada
        (x invertido), para o mesmo pré-sinal valer com qualquer mão.
    """
    quadros = np.asarray(quadros, dtype=np.float64).reshape(-1, TOTAL_FEATURES_QUADRO)
    presentes = maos_presentes(quadros)
    maos = []
    for coluna, lado in enumerate(("esq", "dir")):
        forma = quadros[presentes[:, coluna]][:, IDX_FORMA[lado]]
        if lado == "esq" and len(forma):
            forma = forma.reshape(-1, TOTAL_LANDMARKS_MAO, 3).copy()
            forma[:, :, 0] *= -1.0
            forma = forma.reshape(-1, FEATURES_POR_MAO)
        maos.append(forma)
    return np.vstack(maos) if maos else np.empty((0, FEATURES_POR_MAO))


@dataclass
class ModeloPreSinais:
    """IA de pré-sinais pronta para uso, com o mapa dos sinais de cada um.

    Attributes:
        modelo: Classificador de uma mão (63 valores).
        sinal_para_pre: ``{sinal: pre_sinal}`` dos sinais que têm pré-sinal.
    """

    modelo: Any
    sinal_para_pre: dict[str, str] = field(default_factory=dict)

    @property
    def pre_sinais(self) -> list[str]:
        """Nomes dos pré-sinais que a IA conhece."""
        return [str(c) for c in self.modelo.classes_ if c != NENHUM]

    def sinais_do_pre(self, pre_sinal: str) -> set[str]:
        """Sinais que começam com este pré-sinal."""
        return {s for s, p in self.sinal_para_pre.items() if p == pre_sinal}

    def classificar(self, vetor_quadro: np.ndarray) -> tuple[str | None, float]:
        """Diz se alguma mão do quadro está num pré-sinal.

        Returns:
            ``(nome, confiança)`` do pré-sinal mais provável entre as mãos do
            quadro, ou ``(None, confiança)`` se nenhuma mão passou de
            ``config.CONFIANCA_PRE_SINAL``.
        """
        maos = maos_dos_quadros(vetor_quadro)
        if not len(maos):
            return None, 0.0
        probabilidades = self.modelo.predict_proba(maos)
        classes = [str(c) for c in self.modelo.classes_]
        if NENHUM in classes:
            probabilidades[:, classes.index(NENHUM)] = 0.0
        mao, classe = np.unravel_index(np.argmax(probabilidades), probabilidades.shape)
        confianca = float(probabilidades[mao, classe])
        if confianca < config.CONFIANCA_PRE_SINAL:
            return None, confianca
        return classes[classe], confianca


def treinar_pre_sinais(
    dados_pre: ConjuntoDados,
    dados_sinais: ConjuntoDados,
    sinal_para_pre: dict[str, str],
    caminho: Path,
) -> ModeloPreSinais | None:
    """Treina e salva a IA de pré-sinais (ou apaga o arquivo, se não houver).

    Args:
        dados_pre: Quadros gravados como pré-sinal.
        dados_sinais: Quadros dos sinais; as mãos dos sinais SEM pré-sinal viram
            exemplos "de nenhum".
        sinal_para_pre: ``{sinal: pre_sinal}``.
        caminho: Onde salvar (veja :func:`caminho_modelo_pre`).

    Returns:
        O modelo treinado, ou ``None`` se não há pré-sinais gravados.
    """
    caminho = Path(caminho)
    X, y = [], []
    for nome in dados_pre.classes:
        maos = maos_dos_quadros(dados_pre.X[dados_pre.y == nome])
        X.append(maos)
        y += [nome] * len(maos)
    if not y:
        if caminho.exists():
            os.remove(caminho)
        return None

    sem_pre = ~np.isin(dados_sinais.y, list(sinal_para_pre))
    nenhum = maos_dos_quadros(dados_sinais.X[sem_pre])
    limite = _NENHUM_POR_PRE * len(y)
    if len(nenhum) > limite:
        escolhidos = np.random.default_rng(config.SEMENTE_ALEATORIA).choice(
            len(nenhum), limite, replace=False
        )
        nenhum = nenhum[np.sort(escolhidos)]
    X.append(nenhum)
    y += [NENHUM] * len(nenhum)

    modelo = RandomForestClassifier(
        n_estimators=config.ARVORES_PRE_SINAL,
        min_samples_leaf=config.MIN_AMOSTRAS_FOLHA,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.SEMENTE_ALEATORIA,
    )
    modelo.fit(np.vstack(X), np.asarray(y, dtype=str))
    resultado = ModeloPreSinais(modelo, dict(sinal_para_pre))

    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    joblib.dump(resultado, temporario, compress=3)
    os.replace(temporario, caminho)
    logger.info(
        "IA de pré-sinais salva: %s (%d mãos de pré-sinal, %d de nenhum).",
        resultado.pre_sinais,
        len(y) - len(nenhum),
        len(nenhum),
    )
    return resultado


#: {caminho: (data de modificação, nomes)} para a lista de sinais não
#: recarregar a IA de pré-sinais a cada atualização.
_nomes_em_cache: dict[Path, tuple[float, list[str]]] = {}


def nomes_dos_pre_sinais(caminho: Path) -> list[str]:
    """Pré-sinais que a IA salva conhece (guardado até o arquivo mudar)."""
    caminho = Path(caminho)
    try:
        momento = caminho.stat().st_mtime
    except FileNotFoundError:
        return []
    guardado = _nomes_em_cache.get(caminho)
    if guardado is None or guardado[0] != momento:
        modelo = carregar_pre_sinais(caminho)
        guardado = (momento, modelo.pre_sinais if modelo else [])
        _nomes_em_cache[caminho] = guardado
    return list(guardado[1])


def carregar_pre_sinais(caminho: Path) -> ModeloPreSinais | None:
    """Carrega a IA de pré-sinais, ou ``None`` se não houver (ou não servir)."""
    caminho = Path(caminho)
    if not caminho.exists():
        return None
    try:
        modelo = joblib.load(caminho)
    except Exception:
        logger.exception("Não foi possível carregar a IA de pré-sinais.")
        return None
    if not isinstance(modelo, ModeloPreSinais) or (
        getattr(modelo.modelo, "n_features_in_", None) != FEATURES_POR_MAO
    ):
        logger.warning("A IA de pré-sinais é de outra versão. Treine a IA de novo.")
        return None
    if hasattr(modelo.modelo, "n_jobs"):
        modelo.modelo.n_jobs = 1
    return modelo
