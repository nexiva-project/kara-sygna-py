"""Janelas de quadros: o que a IA usa para reconhecer sinais com movimento.

Um quadro sozinho só mostra a pose. A IA olha os últimos
``config.QUADROS_POR_JANELA`` quadros e recebe:

* o **último quadro** inteiro (forma das mãos, rosto, posição);
* para cada mão, o **movimento** ao longo da janela: de onde o pulso veio
  e quanto ele andou (pelo braço, no MediaPipe Pose, em larguras de ombro e
  relativo ao meio dos ombros), quanto a forma da mão mudou e em quantos
  quadros a mão apareceu.

Sinais parados também funcionam: o movimento simplesmente fica perto de zero.
A forma das mãos do último quadro fica nas primeiras 126 colunas, então o
aumento de dados de :mod:`kara_sygna.ia.aumento` continua valendo.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np

from kara_sygna import config
from kara_sygna.dados.repositorio import ConjuntoDados
from kara_sygna.visao.quadro import (
    COLUNA_CORPO_PRESENTE,
    COLUNAS_PULSO_BRACO,
    IDX_FORMA,
    TOTAL_FEATURES_QUADRO,
    colunas_de_lugar,
    espelhar_quadros,
    maos_presentes,
    nomes_features_quadro,
)

#: Quantos pontos da trajetória do pulso entram no vetor (além do último).
PONTOS_TRAJETORIA = 3
_LADOS = ("esq", "dir")
FEATURES_MOVIMENTO_POR_MAO = 2 * PONTOS_TRAJETORIA + 3


@lru_cache(maxsize=2)
def _colunas_quadro(usar_lugar: bool) -> np.ndarray:
    if usar_lugar:
        return np.arange(TOTAL_FEATURES_QUADRO)
    return np.setdiff1d(np.arange(TOTAL_FEATURES_QUADRO), colunas_de_lugar())


def colunas_quadro_na_janela() -> np.ndarray:
    """Colunas do último quadro que entram na janela.

    Sem ``config.USAR_LUGAR``, as colunas de lugar (onde as mãos estão) ficam
    de fora: a IA olha a forma e o movimento. O movimento é calculado antes,
    com todas as colunas.
    """
    return _colunas_quadro(bool(config.USAR_LUGAR))


def total_features_janela() -> int:
    """Quantos valores tem o vetor de uma janela (depende de ``USAR_LUGAR``)."""
    return len(colunas_quadro_na_janela()) + 2 * FEATURES_MOVIMENTO_POR_MAO


def nomes_features_janela() -> list[str]:
    """Nomes das colunas do vetor de uma janela, na ordem."""
    todos = nomes_features_quadro()
    nomes = [todos[i] for i in colunas_quadro_na_janela()]
    for lado in _LADOS:
        for k in range(PONTOS_TRAJETORIA):
            nomes += [f"mov_{lado}_dx{k}", f"mov_{lado}_dy{k}"]
        nomes += [f"mov_{lado}_caminho", f"mov_{lado}_forma", f"mov_{lado}_presenca"]
    return nomes


def _preencher(posicoes: np.ndarray, validos: np.ndarray) -> np.ndarray | None:
    """Repete a posição válida mais próxima nos quadros sem mão ou sem rosto."""
    if not validos.any():
        return None
    indices = np.flatnonzero(validos)
    proximo = np.clip(
        np.searchsorted(indices, np.arange(len(posicoes))), 0, len(indices) - 1
    )
    anterior = np.clip(proximo - 1, 0, len(indices) - 1)
    escolha = np.where(
        np.abs(indices[anterior] - np.arange(len(posicoes)))
        <= np.abs(indices[proximo] - np.arange(len(posicoes))),
        indices[anterior],
        indices[proximo],
    )
    return posicoes[escolha]


def _movimento_da_mao(
    janela: np.ndarray, presentes: np.ndarray, lado: str
) -> np.ndarray:
    """Calcula as características de movimento de uma mão na janela."""
    saida = np.zeros(FEATURES_MOVIMENTO_POR_MAO)
    presente = presentes[:, _LADOS.index(lado)]
    saida[-1] = presente.mean()

    # Forma: quanto a mão mudou desde o primeiro quadro em que apareceu.
    if presente[-1]:
        forma = janela[:, IDX_FORMA[lado]]
        primeiro = int(np.argmax(presente))
        saida[-2] = float(np.mean(np.abs(forma[-1] - forma[primeiro])))

    # Trajetória do pulso pelo BRAÇO (MediaPipe Pose), em larguras de ombro.
    # O braço continua visível quando a mão borra e some, então o caminho do
    # movimento não tem buracos. Sem corpo na janela, a trajetória fica zero.
    corpo = janela[:, COLUNA_CORPO_PRESENTE] > 0.5
    posicoes = _preencher(janela[:, list(COLUNAS_PULSO_BRACO[lado])], corpo)
    if posicoes is None:
        return saida

    amostras = (
        np.linspace(0, len(janela) - 1, PONTOS_TRAJETORIA + 1).round().astype(int)
    )
    deltas = posicoes[amostras[:-1]] - posicoes[-1]
    saida[: 2 * PONTOS_TRAJETORIA] = deltas.reshape(-1)
    saida[-3] = float(np.linalg.norm(np.diff(posicoes, axis=0), axis=1).sum())
    return saida


def features_janela(janela: np.ndarray) -> np.ndarray:
    """Transforma uma janela de quadros (n, 223) no vetor que a IA recebe.

    Args:
        janela: Quadros em ordem de tempo; o último é o "agora".

    Returns:
        Vetor de ``total_features_janela()`` valores.
    """
    janela = np.asarray(janela, dtype=np.float64)
    if janela.ndim != 2 or janela.shape[1] != TOTAL_FEATURES_QUADRO or len(janela) == 0:
        raise ValueError(
            f"A janela precisa ter formato (n, {TOTAL_FEATURES_QUADRO}); "
            f"recebido {janela.shape}."
        )
    presentes = maos_presentes(janela)
    movimento = [_movimento_da_mao(janela, presentes, lado) for lado in _LADOS]
    return np.concatenate([janela[-1][colunas_quadro_na_janela()], *movimento])


def janelas_de_uma_gravacao(
    quadros: np.ndarray,
    tamanho: int = config.QUADROS_POR_JANELA,
    passo: int = config.PASSO_JANELA_TREINO,
) -> np.ndarray:
    """Corta uma gravação contínua em janelas que terminam em quadros com mão.

    Args:
        quadros: Matriz (n, 223), na ordem em que foram gravados.
        tamanho: Quadros por janela.
        passo: De quantos em quantos quadros começa uma nova janela.

    Returns:
        Matriz (m, ``total_features_janela()``). Pode ter 0 linhas se a gravação
        for curta demais ou não tiver mãos.
    """
    quadros = np.asarray(quadros, dtype=np.float64).reshape(-1, TOTAL_FEATURES_QUADRO)
    com_mao = maos_presentes(quadros).any(axis=1)
    fins = [fim for fim in range(tamanho - 1, len(quadros), passo) if com_mao[fim]]
    if not fins:
        return np.empty((0, total_features_janela()))
    return np.array(
        [features_janela(quadros[fim - tamanho + 1 : fim + 1]) for fim in fins]
    )


def mudar_velocidade(quadros: np.ndarray, fator: float) -> np.ndarray:
    """Refaz uma gravação como se o sinal fosse feito mais rápido ou devagar.

    Args:
        quadros: Matriz (n, 223), em ordem de gravação.
        fator: ``> 1`` = mais rápido (pula quadros); ``< 1`` = mais devagar
            (repete quadros); ``1`` = igual.

    Returns:
        Nova matriz com cerca de ``n / fator`` quadros.
    """
    quadros = np.asarray(quadros)
    if fator <= 0:
        raise ValueError("O fator de velocidade precisa ser positivo.")
    if fator == 1 or len(quadros) < 2:
        return quadros
    indices = np.round(np.arange(0, len(quadros) - 0.5, fator)).astype(int)
    return quadros[np.clip(indices, 0, len(quadros) - 1)]


def montar_amostras(
    dados: ConjuntoDados,
    tamanho: int = config.QUADROS_POR_JANELA,
    passo: int = config.PASSO_JANELA_TREINO,
    velocidades: tuple[float, ...] = config.VELOCIDADES_TREINO,
) -> ConjuntoDados:
    """Transforma os quadros gravados nas amostras (janelas) do treino.

    Cada gravação (rodada) é cortada em janelas. Cada gravação também entra
    espelhada, para que um sinal feito com uma mão valha para a outra, e em
    outras ``velocidades`` (mais rápida e mais devagar), para o sinal ser
    reconhecido no ritmo de cada um. As cópias ficam no mesmo grupo da
    original, então nunca vazam para o teste.

    Args:
        dados: Quadros do CSV, com a rodada de cada um, em ordem de gravação.

    Returns:
        ``ConjuntoDados`` com ``X`` de ``total_features_janela()`` colunas.
    """
    X: list[np.ndarray] = []
    y: list[str] = []
    grupos: list = []
    for rodada in dict.fromkeys(dados.grupos.tolist()):
        da_rodada = dados.grupos == rodada
        rotulos = dados.y[da_rodada]
        quadros = dados.X[da_rodada]
        # Uma rodada é de um sinal só; se não for, cada sinal vira sua sequência.
        for rotulo in dict.fromkeys(rotulos.tolist()):
            sequencia = quadros[rotulos == rotulo]
            # Gravação curta demais não vira janela, nem mais devagar.
            fatores = (1.0, *velocidades) if len(sequencia) >= tamanho else (1.0,)
            for espelho in (sequencia, espelhar_quadros(sequencia)):
                for fator in dict.fromkeys(fatores):
                    versao = mudar_velocidade(espelho, fator)
                    janelas = janelas_de_uma_gravacao(versao, tamanho, passo)
                    X.append(janelas)
                    y += [rotulo] * len(janelas)
                    grupos += [rodada] * len(janelas)

    matriz = np.vstack(X) if X else np.empty((0, total_features_janela()))
    return ConjuntoDados(matriz, np.asarray(y, dtype=str), np.asarray(grupos))


class BufferQuadros:
    """Guarda os últimos quadros da câmera para montar a janela ao vivo."""

    def __init__(self, tamanho: int = config.QUADROS_POR_JANELA) -> None:
        """Cria o buffer vazio.

        Args:
            tamanho: Quantos quadros formam uma janela.
        """
        self.tamanho = tamanho
        self._quadros: deque[np.ndarray] = deque(maxlen=tamanho)

    def limpar(self) -> None:
        """Esquece todos os quadros (ao começar a reconhecer)."""
        self._quadros.clear()

    def adicionar(self, vetor_quadro: np.ndarray) -> None:
        """Acrescenta o quadro mais recente."""
        vetor = np.asarray(vetor_quadro, dtype=np.float64).reshape(-1)
        if vetor.size != TOTAL_FEATURES_QUADRO:
            raise ValueError(
                f"Esperados {TOTAL_FEATURES_QUADRO} valores; recebidos {vetor.size}."
            )
        self._quadros.append(vetor)

    @property
    def cheio(self) -> bool:
        """Diz se já há quadros suficientes para uma janela."""
        return len(self._quadros) == self.tamanho

    def features(self) -> np.ndarray:
        """Vetor da janela atual (veja :func:`features_janela`)."""
        return features_janela(np.array(self._quadros))
