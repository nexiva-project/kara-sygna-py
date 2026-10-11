"""Salvar, carregar e validar o modelo treinado (arquivo ``.pkl``)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import joblib

from kara_sygna.visao.caracteristicas import TOTAL_FEATURES, nomes_features

logger = logging.getLogger(__name__)


class ModeloInvalidoError(ValueError):
    """O arquivo do modelo não serve para este programa."""


def salvar_modelo(modelo: Any, caminho: Path) -> None:
    """Grava o modelo de forma segura.

    Escreve num arquivo temporário e só então troca o definitivo, para que uma
    falha no meio do caminho nunca deixe um modelo corrompido.

    Args:
        modelo: Modelo treinado (scikit-learn).
        caminho: Destino (``.pkl``).
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_suffix(caminho.suffix + ".tmp")
    joblib.dump(modelo, temporario)
    os.replace(temporario, caminho)
    logger.info("Modelo salvo em %s", caminho)


def validar_modelo(modelo: Any) -> None:
    """Confere se o modelo é compatível com as características atuais.

    Raises:
        ModeloInvalidoError: Se não tiver ``predict_proba``, se esperar outro
            número de características ou se os nomes das colunas diferirem.
    """
    if not hasattr(modelo, "predict_proba"):
        raise ModeloInvalidoError("O modelo carregado não oferece predict_proba().")

    esperado = getattr(modelo, "n_features_in_", None)
    if esperado is not None and esperado != TOTAL_FEATURES:
        raise ModeloInvalidoError(
            f"O modelo espera {esperado} características, mas o programa gera {TOTAL_FEATURES}. "
            "Treine a IA de novo com o CSV atual."
        )

    nomes = getattr(modelo, "feature_names_in_", None)
    if nomes is not None and list(nomes) != nomes_features():
        raise ModeloInvalidoError(
            "O modelo foi treinado com outras colunas. Treine a IA de novo."
        )


def carregar_modelo(caminho: Path) -> Any:
    """Carrega e valida o modelo.

    Para o reconhecimento em tempo real o modelo usa um único processo
    (``n_jobs=1``): com poucas linhas por quadro, paralelizar só atrasa.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ModeloInvalidoError: Se o modelo for incompatível (ver :func:`validar_modelo`).
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {caminho}")
    modelo = joblib.load(caminho)
    validar_modelo(modelo)
    if hasattr(modelo, "n_jobs"):
        modelo.n_jobs = 1
    return modelo


def modelo_esta_atualizado(caminho_modelo: Path, caminho_csv: Path) -> bool:
    """Diz se o modelo é mais novo que o CSV (ou seja, não há dados novos sem treinar)."""
    caminho_modelo, caminho_csv = Path(caminho_modelo), Path(caminho_csv)
    if not caminho_modelo.exists():
        return False
    if not caminho_csv.exists():
        return True
    return caminho_modelo.stat().st_mtime >= caminho_csv.stat().st_mtime


def classes_do_modelo(caminho_modelo: Path) -> set[str]:
    """Lista os sinais que o modelo salvo conhece (vazio se não houver modelo válido)."""
    try:
        return {str(c) for c in carregar_modelo(caminho_modelo).classes_}
    except (FileNotFoundError, ModeloInvalidoError):
        return set()
    except Exception:
        logger.exception("Não foi possível consultar as classes do modelo.")
        return set()
