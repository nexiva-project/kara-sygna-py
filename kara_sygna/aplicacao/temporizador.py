"""Contagem antes do ensino e parada automática depois do tempo escolhido.

Fluxo: o usuário aperta Enter (ou o botão), espera
``config.SEGUNDOS_CONTAGEM_ENSINO`` segundos sem gravar e então grava pelo
tempo que escreveu. Sem tempo escrito, grava até o usuário parar.
"""

from __future__ import annotations

import enum
import math
import time
from collections.abc import Callable

from kara_sygna import config
from kara_sygna.aplicacao.erros import DuracaoInvalidaError

#: Maior tempo de gravação aceito, em segundos.
DURACAO_MAXIMA_S = 600


def ler_duracao(texto: str) -> float | None:
    """Converte o texto do campo de segundos.

    Args:
        texto: O que o usuário escreveu (ex.: ``"10"``, ``"7,5"``).

    Returns:
        Os segundos, ou ``None`` se o campo estiver vazio (grava até parar).

    Raises:
        DuracaoInvalidaError: Se não for um número entre 0 e ``DURACAO_MAXIMA_S``.
    """
    texto = texto.strip().lower().removesuffix("s").strip().replace(",", ".")
    if not texto:
        return None
    try:
        segundos = float(texto)
    except ValueError:
        raise DuracaoInvalidaError(
            f"Tempo de gravação inválido: “{texto}”. Escreva só os segundos, como 10."
        ) from None
    if not math.isfinite(segundos) or not 0 < segundos <= DURACAO_MAXIMA_S:
        raise DuracaoInvalidaError(
            f"O tempo de gravação precisa ficar entre 0 e {DURACAO_MAXIMA_S} segundos."
        )
    return segundos


class Fase(enum.Enum):
    """Em que ponto o temporizador está."""

    PARADO = enum.auto()
    CONTAGEM = enum.auto()
    GRAVANDO = enum.auto()


class Evento(enum.Enum):
    """O que a interface deve fazer neste quadro."""

    NADA = enum.auto()
    COMECAR = enum.auto()  # a contagem acabou: começar a gravar
    PARAR = enum.auto()  # o tempo de gravação acabou


class TemporizadorEnsino:
    """Controla a contagem e o tempo de gravação (sem nenhuma interface).

    Args:
        contagem_s: Segundos de espera antes de gravar. Padrão:
            ``config.SEGUNDOS_CONTAGEM_ENSINO``.
        relogio: Função que devolve o tempo em segundos (trocada nos testes).
    """

    def __init__(
        self,
        contagem_s: float | None = None,
        relogio: Callable[[], float] = time.monotonic,
    ) -> None:
        """Cria o temporizador parado (veja a classe para os argumentos)."""
        self.contagem_s = float(
            config.SEGUNDOS_CONTAGEM_ENSINO if contagem_s is None else contagem_s
        )
        self._relogio = relogio
        self.fase = Fase.PARADO
        self.duracao_s: float | None = None
        self._inicio = 0.0

    @property
    def ativo(self) -> bool:
        """Diz se está contando ou gravando."""
        return self.fase is not Fase.PARADO

    def iniciar(self, duracao_s: float | None) -> None:
        """Começa a contagem.

        Args:
            duracao_s: Quanto tempo gravar depois da contagem, ou ``None``
                para gravar até o usuário parar.
        """
        self.duracao_s = duracao_s
        self.fase = Fase.CONTAGEM
        self._inicio = self._relogio()

    def cancelar(self) -> None:
        """Para tudo (Enter ou PARAR durante a contagem ou a gravação)."""
        self.fase = Fase.PARADO

    def atualizar(self) -> Evento:
        """Avança o tempo; chame a cada quadro da câmera."""
        decorrido = self._relogio() - self._inicio
        if self.fase is Fase.CONTAGEM and decorrido >= self.contagem_s:
            self.fase = Fase.GRAVANDO
            self._inicio += self.contagem_s
            return Evento.COMECAR
        if (
            self.fase is Fase.GRAVANDO
            and self.duracao_s is not None
            and decorrido >= self.duracao_s
        ):
            self.fase = Fase.PARADO
            return Evento.PARAR
        return Evento.NADA

    def segundos_restantes(self) -> int | None:
        """Segundos inteiros que faltam para a fase atual acabar (arredonda para cima).

        Returns:
            ``None`` se estiver parado ou gravando sem tempo definido.
        """
        if self.fase is Fase.CONTAGEM:
            limite = self.contagem_s
        elif self.fase is Fase.GRAVANDO and self.duracao_s is not None:
            limite = self.duracao_s
        else:
            return None
        return max(0, math.ceil(limite - (self._relogio() - self._inicio)))
