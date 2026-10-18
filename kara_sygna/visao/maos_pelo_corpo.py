"""Confere as mãos detectadas pelos braços do corpo (MediaPipe Pose).

Por que existe
--------------
Com a exigência de detecção baixa e o realce de pele, o MediaPipe às vezes
acha uma "segunda mão" que não existe. Num teste com o sinal ``neutro``, só
com a mão esquerda, apareceu uma mão "direita" em 27 de 69 quadros, sempre
colada na mão esquerda.

Regras (o Pose acerta o lado do braço, mas erra a posição exata do pulso,
principalmente com as mãos embaixo, perto da borda da imagem: no sinal
"carro" ele chegava a pôr o pulso fora da imagem, longe da mão de verdade):

* **mão duplicada**: duas mãos quase no mesmo lugar (até
  ``config.DISTANCIA_MAOS_DUPLICADAS`` larguras de ombro) são a mesma mão;
  fica a mais perto de um pulso do braço;
* **mão perdida**: só é descartada a mão muito longe dos dois pulsos (mais
  de ``config.DISTANCIA_MAO_PULSO`` larguras de ombro);
* **lado**: cada mão recebe o lado do braço pela melhor combinação
  mão-braço (menor distância somada), e não pelo palpite do MediaPipe: nos
  vídeos testados, quando os dois discordavam, o braço acertou 82 de 83.

Sem corpo na imagem, as mãos passam sem mudança.
"""

from __future__ import annotations

from itertools import permutations
from types import SimpleNamespace
from typing import Any

import numpy as np

from kara_sygna import config
from kara_sygna.visao.quadro import POSE_DOS_SLOTS

_VISIBILIDADE_OMBRO = 0.5
_LADOS = ("esq", "dir")


def conferir_maos(
    resultado_maos: Any,
    resultado_corpo: Any,
    largura: int,
    altura: int,
    distancia_maxima: float | None = None,
) -> Any:
    """Tira as mãos fantasmas e dá a cada mão o lado do braço dela.

    Args:
        resultado_maos: Resultado bruto de ``DetectorMaos.detectar``.
        resultado_corpo: Resultado de ``DetectorCorpo.detectar`` (ou ``None``).
        largura: Largura do quadro, em pixels.
        altura: Altura do quadro, em pixels.
        distancia_maxima: Distância máxima pulso da mão - pulso do braço mais
            perto, em larguras de ombro. Padrão: ``config.DISTANCIA_MAO_PULSO``.

    Returns:
        O próprio ``resultado_maos`` se não houver corpo; senão, um objeto com
        as mãos que sobraram (``multi_hand_landmarks``, ``multi_handedness``,
        ``multi_hand_world_landmarks``), o lado de cada uma em ``lados`` e
        quantas foram descartadas em ``descartadas``.
    """
    listas = list(getattr(resultado_maos, "multi_hand_landmarks", None) or [])
    pontos = getattr(getattr(resultado_corpo, "pose_landmarks", None), "landmark", None)
    if not listas or pontos is None or len(pontos) < 17:
        return resultado_maos

    escala = np.array([largura, altura], dtype=np.float64)

    def px(ponto) -> np.ndarray:
        return np.array([ponto.x, ponto.y]) * escala

    ombros = [pontos[POSE_DOS_SLOTS[lado][0]] for lado in _LADOS]
    if any(getattr(o, "visibility", 1.0) < _VISIBILIDADE_OMBRO for o in ombros):
        return resultado_maos
    largura_ombros = float(np.linalg.norm(px(ombros[0]) - px(ombros[1])))
    if not np.isfinite(largura_ombros) or largura_ombros < 1.0:
        return resultado_maos

    limite = (
        config.DISTANCIA_MAO_PULSO if distancia_maxima is None else distancia_maxima
    )
    pulsos = np.array([px(pontos[POSE_DOS_SLOTS[lado][2]]) for lado in _LADOS])
    validas = [
        i for i, lm in enumerate(listas) if len(getattr(lm, "landmark", [])) >= 21
    ]
    maos = {i: px(listas[i].landmark[0]) for i in validas}

    def ao_braco(i: int) -> float:
        """Distância da mão ``i`` ao pulso de braço mais perto (em ombros)."""
        return float(np.min(np.linalg.norm(pulsos - maos[i], axis=1))) / largura_ombros

    # 1. Mão muito longe dos dois braços: perdida.
    ficam = [i for i in validas if ao_braco(i) <= limite]
    # 2. Duas mãos quase no mesmo lugar: a mesma mão (fica a mais perto do braço).
    ficam.sort(key=ao_braco)
    unicas: list[int] = []
    for i in ficam:
        if all(
            np.linalg.norm(maos[i] - maos[j]) / largura_ombros
            >= config.DISTANCIA_MAOS_DUPLICADAS
            for j in unicas
        ):
            unicas.append(i)
    unicas = unicas[: len(_LADOS)]

    # 3. Lado de cada mão: a combinação mão-braço com a menor distância somada.
    # Em 83 quadros de vídeo em que o lado do MediaPipe e o do braço
    # discordavam, o braço acertou 82 (pela posição da mão em relação ao nariz).
    custos = sorted(
        (
            sum(
                float(np.linalg.norm(maos[i] - pulsos[lado]))
                for i, lado in zip(unicas, lados, strict=True)
            ),
            lados,
        )
        for lados in permutations(range(len(_LADOS)), len(unicas))
    )
    ordem = sorted(unicas)
    melhor = dict(zip(unicas, custos[0][1], strict=True)) if custos else {}

    lateralidades = list(getattr(resultado_maos, "multi_handedness", None) or [])
    mundos = list(getattr(resultado_maos, "multi_hand_world_landmarks", None) or [])
    return SimpleNamespace(
        multi_hand_landmarks=[listas[i] for i in ordem],
        multi_handedness=[
            lateralidades[i] if i < len(lateralidades) else None for i in ordem
        ],
        multi_hand_world_landmarks=[
            mundos[i] if i < len(mundos) else None for i in ordem
        ],
        lados=[_LADOS[melhor[i]] for i in ordem],
        descartadas=len(listas) - len(ordem),
    )
