"""Preenche os quadros em que uma mão sumiu por pouco tempo (na gravação).

Por que existe
--------------
Com movimento rápido, a mão borra e o MediaPipe a perde por alguns quadros.
Na gravação dá para saber onde a mão estava ANTES e DEPOIS do buraco, então
os quadros do meio são preenchidos com o caminho entre os dois. A IA aprende
o movimento completo, sem buracos e sem a mão "congelada".

* Estudos com MediaPipe fazem o mesmo (cada mão separada): +6% de acurácia
  no WLASL e de +4 a +15 pontos num estudo de Libras, com curvas suaves.
* No vídeo do sinal "carro", preencher entre antes e depois errou 5,0% da
  imagem; congelar a mão, 7,5%; continuar na mesma direção, 9,8%.
* A maioria dos buracos é curta: até 0,5 s cobriu de 80% a 92% do tempo
  em que as mãos sumiram.

Só é usado na gravação: ao vivo, preencher exigiria esperar a mão voltar.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from kara_sygna import config
from kara_sygna.visao.quadro import (
    TOTAL_FEATURES_QUADRO,
    colunas_da_mao,
    maos_presentes,
)

#: Quantos quadros vistos de cada lado do buraco guiam a curva.
_VIZINHOS = 3
_LADOS = ("esq", "dir")


def preencher_buracos(
    quadros: np.ndarray,
    tempos: np.ndarray,
    maximo_s: float = config.PREENCHER_BURACO_S,
) -> tuple[np.ndarray, int]:
    """Preenche, mão por mão, os buracos curtos entre dois quadros com a mão.

    A curva usa até ``_VIZINHOS`` quadros com a mão antes e depois do buraco
    (PCHIP: suave, sem passar do ponto). Com menos de 4 pontos, vira uma reta.
    Buracos no começo ou no fim da gravação, ou mais longos que ``maximo_s``,
    ficam como estão (mão ausente).

    Args:
        quadros: Matriz (n, 223), em ordem de gravação.
        tempos: Momento de cada quadro, em segundos (crescente).
        maximo_s: Maior buraco preenchido, em segundos (``0`` = nenhum).

    Returns:
        ``(quadros preenchidos, quantos quadros-mão foram preenchidos)``. A
        matriz de entrada não é alterada.
    """
    quadros = np.array(quadros, dtype=np.float64).reshape(-1, TOTAL_FEATURES_QUADRO)
    tempos = np.asarray(tempos, dtype=np.float64).reshape(-1)
    if len(tempos) != len(quadros):
        raise ValueError("É preciso um tempo para cada quadro.")
    if maximo_s <= 0 or len(quadros) < 3:
        return quadros, 0
    # Relógios de baixa resolução repetem o mesmo valor: garante a ordem.
    tempos = tempos + np.arange(len(tempos)) * 1e-6

    preenchidos = 0
    presentes = maos_presentes(quadros)
    for coluna, lado in enumerate(_LADOS):
        vistos = np.flatnonzero(presentes[:, coluna])
        colunas = colunas_da_mao(lado)
        for posicao in range(len(vistos) - 1):
            antes, depois = vistos[posicao], vistos[posicao + 1]
            if depois - antes < 2 or tempos[depois] - tempos[antes] > maximo_s:
                continue
            buraco = np.arange(antes + 1, depois)
            guias = np.concatenate(
                [
                    vistos[max(0, posicao - _VIZINHOS + 1) : posicao + 1],
                    vistos[posicao + 1 : posicao + 1 + _VIZINHOS],
                ]
            )
            valores = quadros[np.ix_(guias, colunas)]
            if len(guias) >= 4:
                curva = PchipInterpolator(tempos[guias], valores, axis=0)
                quadros[np.ix_(buraco, colunas)] = curva(tempos[buraco])
            else:
                fracao = (tempos[buraco] - tempos[antes]) / (
                    tempos[depois] - tempos[antes]
                )
                quadros[np.ix_(buraco, colunas)] = quadros[antes, colunas] + fracao[
                    :, None
                ] * (quadros[depois, colunas] - quadros[antes, colunas])
            preenchidos += len(buraco)
    return quadros, preenchidos
