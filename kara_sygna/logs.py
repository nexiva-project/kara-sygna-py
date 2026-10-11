"""Configuração de logs do Kara Sygna.

Em vez de um módulo global com funções ``log()`` chamadas de todo lugar, cada
módulo usa o ``logging`` padrão do Python::

    logger = logging.getLogger(__name__)
    logger.info("mensagem")

O formatador abaixo descobre a *categoria* (CAMERA, MODEL, DATA...) a partir do
nome do módulo, e o mesmo formato é usado no terminal e na janela de log.
"""

from __future__ import annotations

import logging
import sys

NOME_RAIZ = "kara_sygna"

_CATEGORIAS = {
    "camera": "CAMERA",
    "visao": "VISION",
    "dados": "DATA",
    "ia": "MODEL",
    "aplicacao": "APP",
    "interface": "APP",
    "cli": "APP",
}


def categoria_do_logger(nome: str) -> str:
    """Descobre a categoria de um logger pelo nome do módulo.

    Args:
        nome: Nome do logger, por exemplo ``"kara_sygna.camera.captura"``.

    Returns:
        Categoria em maiúsculas (``"CAMERA"``, ``"MODEL"``...). Para nomes
        desconhecidos devolve ``"APP"``.
    """
    partes = nome.split(".")
    if len(partes) > 1 and partes[0] == NOME_RAIZ:
        return _CATEGORIAS.get(partes[1], "APP")
    return "APP"


class FormatadorKara(logging.Formatter):
    """Formata registros como ``HH:MM:SS [CATEGORIA] mensagem``.

    Se houver uma exceção anexada ao registro, o traceback completo é
    acrescentado depois da mensagem.
    """

    def __init__(self) -> None:
        """Define o formato ``HH:MM:SS [CATEGORIA] mensagem``."""
        super().__init__(
            fmt="%(asctime)s [%(categoria)s] %(message)s", datefmt="%H:%M:%S"
        )

    def format(self, record: logging.LogRecord) -> str:
        """Acrescenta a categoria ao registro e devolve o texto formatado."""
        record.categoria = categoria_do_logger(record.name)
        return super().format(record)


def configurar_logging(nivel: int = logging.INFO) -> None:
    """Liga os logs do projeto no terminal (idempotente).

    Args:
        nivel: Nível mínimo exibido (``logging.INFO`` por padrão).
    """
    raiz = logging.getLogger(NOME_RAIZ)
    raiz.setLevel(nivel)
    if any(getattr(h, "_kara_console", False) for h in raiz.handlers):
        return
    console = logging.StreamHandler(sys.stderr)
    console._kara_console = True  # marca para não duplicar em chamadas repetidas
    console.setFormatter(FormatadorKara())
    raiz.addHandler(console)
