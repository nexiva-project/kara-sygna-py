"""Treinamento do classificador, com avaliação honesta por rodada."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold

from kara_sygna import config
from kara_sygna.dados.repositorio import ConjuntoDados
from kara_sygna.ia.aumento import expandir_treino, numero_de_variantes
from kara_sygna.ia.modelo_io import salvar_modelo
from kara_sygna.visao.caracteristicas import nomes_features

logger = logging.getLogger(__name__)

#: Função que recebe uma mensagem de progresso.
Progresso = Callable[[str], None]


class TreinoImpossivelError(ValueError):
    """Os dados não bastam para treinar (poucos sinais, CSV vazio...)."""


@dataclass(frozen=True)
class ResultadoTreino:
    """Resumo do treinamento.

    Attributes:
        amostras: Quantidade de linhas usadas.
        classes: Sinais aprendidos.
        rodadas: Quantidade de rodadas de gravação distintas.
        acuracia_validacao: Acurácia na validação por rodada, ou ``None`` se
            não havia rodadas suficientes para avaliar.
        relatorio: Texto com precisão/revocação por sinal (ou um aviso).
    """

    amostras: int
    classes: list[str]
    rodadas: int
    acuracia_validacao: float | None
    relatorio: str


def _novo_modelo() -> RandomForestClassifier:
    """Cria o RandomForest com os hiperparâmetros do projeto."""
    return RandomForestClassifier(
        n_estimators=config.ARVORES,
        min_samples_leaf=config.MIN_AMOSTRAS_FOLHA,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=config.SEMENTE_ALEATORIA,
    )


def _ajustar(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Aumenta os dados de treino e ajusta um modelo novo."""
    X_exp, y_exp = expandir_treino(X, y, numero_de_variantes(len(X)))
    modelo = _novo_modelo()
    modelo.fit(pd.DataFrame(X_exp, columns=nomes_features()), y_exp)
    return modelo


def avaliar(
    dados: ConjuntoDados, progresso: Progresso = lambda _m: None
) -> tuple[float | None, str]:
    """Validação cruzada por rodada.

    Quadros da mesma rodada nunca aparecem no treino e no teste ao mesmo tempo,
    então o número reflete melhor o uso real. Sinais com uma única rodada não
    podem ser avaliados e ficam de fora **só da avaliação** (o modelo final os
    aprende normalmente).

    Args:
        dados: Todos os exemplos.
        progresso: Recebe mensagens de andamento.

    Returns:
        ``(acuracia, relatorio)``. Se não houver sinais com rodadas suficientes,
        ``(None, aviso)``.
    """
    rodadas = {c: len(np.unique(dados.grupos[dados.y == c])) for c in dados.classes}
    avaliaveis = [c for c, n in rodadas.items() if n >= 2]
    if len(avaliaveis) < 2:
        return None, (
            "Avaliação pulada: é preciso que pelo menos 2 sinais tenham 2 ou mais rodadas de gravação "
            "(clique em COMEÇAR A ENSINAR mais de uma vez por sinal, variando posição, distância e luz)."
        )

    ficam = np.isin(dados.y, avaliaveis)
    X, y, grupos = dados.X[ficam], dados.y[ficam], dados.grupos[ficam]
    dobras = min(config.MAX_DOBRAS_VALIDACAO, min(rodadas[c] for c in avaliaveis))

    validador = StratifiedGroupKFold(
        n_splits=dobras, shuffle=True, random_state=config.SEMENTE_ALEATORIA
    )
    previstos = np.empty(len(y), dtype=object)
    for numero, (idx_treino, idx_teste) in enumerate(validador.split(X, y, grupos), 1):
        progresso(f"Validando {numero}/{dobras}...")
        modelo = _ajustar(X[idx_treino], y[idx_treino])
        previstos[idx_teste] = modelo.predict(
            pd.DataFrame(X[idx_teste], columns=nomes_features())
        )

    previstos = previstos.astype(str)
    relatorio = classification_report(y, previstos, zero_division=0)
    fora = [c for c in rodadas if c not in avaliaveis]
    if fora:
        relatorio += f"\nFora da avaliação (só 1 rodada): {', '.join(fora)}"
    return float(accuracy_score(y, previstos)), relatorio


def treinar(
    dados: ConjuntoDados,
    caminho_modelo: Path = config.ARQUIVO_MODELO,
    progresso: Progresso = lambda _m: None,
) -> ResultadoTreino:
    """Avalia, treina o modelo final com todos os dados e o salva.

    Roda por vários segundos: na interface, chame em segundo plano.

    Args:
        dados: Exemplos carregados do CSV.
        caminho_modelo: Onde salvar o ``.pkl``. Um resumo ``.json`` é salvo ao lado.
        progresso: Recebe mensagens de andamento.

    Returns:
        :class:`ResultadoTreino`.

    Raises:
        TreinoImpossivelError: Se o CSV estiver vazio ou tiver só 1 sinal.
    """
    if len(dados) == 0:
        raise TreinoImpossivelError("O CSV está vazio. Ensine pelo menos dois sinais.")
    classes = dados.classes
    if len(classes) < 2:
        raise TreinoImpossivelError(
            "Só há 1 sinal gravado. Grave pelo menos 2 sinais — de preferência também um "
            "sinal 'neutro' (mão relaxada) para a IA saber quando NÃO há sinal."
        )

    rodadas = len(set(dados.grupos.tolist()))
    progresso(f"{len(dados)} exemplos, {len(classes)} sinais, {rodadas} rodadas.")
    acuracia, relatorio = avaliar(dados, progresso)

    progresso("Treinando o modelo final...")
    modelo = _ajustar(dados.X, dados.y)
    salvar_modelo(modelo, caminho_modelo)

    resultado = ResultadoTreino(len(dados), classes, rodadas, acuracia, relatorio)
    _salvar_resumo(resultado, Path(caminho_modelo).with_suffix(".json"))
    logger.info(
        "Treino concluído: %d amostras, %d sinais, acurácia=%s",
        len(dados),
        len(classes),
        acuracia,
    )
    return resultado


def _salvar_resumo(resultado: ResultadoTreino, caminho: Path) -> None:
    """Grava um JSON legível com o resumo do treino."""
    resumo = {
        "amostras": resultado.amostras,
        "classes": resultado.classes,
        "rodadas": resultado.rodadas,
        "acuracia_validacao": None
        if resultado.acuracia_validacao is None
        else round(resultado.acuracia_validacao, 4),
    }
    caminho.write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
