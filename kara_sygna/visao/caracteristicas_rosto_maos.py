"""Kara Sygna — características combinadas de expressão facial e mãos.

Este módulo é independente dos extratores antigos:
- não altera os 126 valores das mãos;
- usa os 468 pontos brutos do FaceMesh (resultado_rosto.landmarks), não o
  vetor facial leve do DetectorRosto;
- gera um vetor adicional de 82 características.

Layout:
    [rosto presente (1) |
     mãos presentes (2) |
     expressão facial (9) |
     distâncias dedos-regiões (70)]

As distâncias são relativas à largura do rosto.
A proximidade não comprova contato físico.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

PONTAS_DEDOS = (4, 8, 12, 16, 20)

NOMES_REGIOES = (
    "testa",
    "olhos",
    "nariz",
    "bochecha_esquerda",
    "bochecha_direita",
    "boca",
    "queixo",
)

# Índices do modelo FaceMesh de 468 pontos.
REGIOES: dict[str, tuple[int, ...]] = {
    "testa": (
        10,
        67,
        69,
        104,
        108,
        109,
        151,
        297,
        299,
        333,
        337,
        338,
    ),
    "olhos": (
        33,
        133,
        159,
        145,
        160,
        144,
        158,
        153,
        362,
        263,
        386,
        374,
        387,
        373,
        385,
        380,
    ),
    "nariz": (
        1,
        2,
        4,
        5,
        6,
        19,
        94,
        97,
        98,
        168,
        195,
        197,
        326,
        327,
        328,
        358,
        457,
    ),
    "bochecha_esquerda": (
        50,
        101,
        118,
        119,
        120,
        121,
        126,
        205,
        206,
        207,
        187,
        123,
        116,
    ),
    "bochecha_direita": (
        280,
        330,
        347,
        348,
        349,
        350,
        355,
        425,
        426,
        427,
        411,
        352,
        345,
    ),
    "boca": (
        0,
        13,
        14,
        17,
        37,
        39,
        40,
        61,
        78,
        80,
        81,
        82,
        84,
        87,
        88,
        91,
        146,
        178,
        181,
        185,
        191,
        267,
        269,
        270,
        291,
        308,
        310,
        311,
        312,
        314,
        317,
        318,
        321,
        375,
        402,
        405,
        409,
    ),
    "queixo": (
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        377,
        400,
        378,
        379,
        365,
    ),
}

INDICES_EXPRESSAO = (
    "olho_esquerdo_abertura",
    "olho_direito_abertura",
    "boca_abertura",
    "boca_largura",
    "sobrancelha_esquerda",
    "sobrancelha_direita",
    "assimetria_boca",
    "nariz_boca",
    "inclinacao_olhos",
)

# Valor de distância para dados indisponíveis.
# É um valor sentinela, não uma distância física real.
DISTANCIA_INDISPONIVEL = 2.0

TOTAL_FEATURES_ROSTO_MAOS = 82


def nomes_features_rosto_maos() -> list[str]:
    """Devolve os nomes das 82 características, na ordem do vetor."""
    nomes = [
        "rosto_presente",
        "mao_esq_presente",
        "mao_dir_presente",
    ]

    nomes.extend(INDICES_EXPRESSAO)

    for lado in ("esq", "dir"):
        for dedo in PONTAS_DEDOS:
            for regiao in NOMES_REGIOES:
                nomes.append(f"{lado}_dedo_{dedo}_dist_{regiao}")

    return nomes


def _ponto(
    landmarks: Any,
    indice: int,
    largura: int,
    altura: int,
) -> np.ndarray:
    """Converte um landmark normalizado para pixels (x, y)."""
    p = landmarks.landmark[indice]
    return np.array(
        [p.x * largura, p.y * altura],
        dtype=np.float64,
    )


def _distancia(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _razao(
    numerador: float,
    denominador: float,
) -> float:
    if not np.isfinite(denominador) or denominador < 1e-6:
        return 0.0

    valor = numerador / denominador

    if not np.isfinite(valor):
        return 0.0

    return float(np.clip(valor, 0.0, 2.0))


def _extrair_expressao(
    rosto: Any,
    largura: int,
    altura: int,
    largura_rosto: float,
) -> list[float]:
    """Calcula nove medidas geométricas faciais."""
    p = lambda i: _ponto(rosto, i, largura, altura)

    olho_esq_largura = _distancia(p(33), p(133))
    olho_esq_abertura = _distancia(p(159), p(145))

    olho_dir_largura = _distancia(p(362), p(263))
    olho_dir_abertura = _distancia(p(386), p(374))

    boca_largura = _distancia(p(61), p(291))
    boca_abertura = _distancia(p(13), p(14))

    # Distância vertical entre sobrancelha e região superior do olho.
    sobrancelha_esq = abs(float(p(105)[1] - p(159)[1]))
    sobrancelha_dir = abs(float(p(334)[1] - p(386)[1]))

    # Inclinação dos olhos, útil para acompanhar a rotação da cabeça.
    centro_esq = (p(33) + p(133)) / 2.0
    centro_dir = (p(362) + p(263)) / 2.0
    delta = centro_dir - centro_esq
    inclinacao = float(np.arctan2(delta[1], delta[0]) / np.pi)

    valores = [
        _razao(olho_esq_abertura, olho_esq_largura),
        _razao(olho_dir_abertura, olho_dir_largura),
        _razao(boca_abertura, boca_largura),
        _razao(boca_largura, largura_rosto),
        _razao(sobrancelha_esq, largura_rosto),
        _razao(sobrancelha_dir, largura_rosto),
        _razao(abs(float(p(61)[1] - p(291)[1])), largura_rosto),
        _razao(_distancia(p(1), p(13)), largura_rosto),
        float(np.clip(inclinacao, -1.0, 1.0)),
    ]

    return valores


def _obter_maos(resultado_maos: Any) -> list[np.ndarray | None]:
    """Obtém até duas mãos, ordenadas pela posição horizontal na imagem.

    Returns:
        Lista com 2 posições; cada uma é uma matriz (21, 3) de coordenadas
        normalizadas do MediaPipe, ou ``None`` se não houver mão.
    """
    listas = getattr(resultado_maos, "multi_hand_landmarks", None) or []

    maos_validas = []

    for landmarks in listas:
        if len(getattr(landmarks, "landmark", [])) < 21:
            continue

        pontos = np.array(
            [[p.x, p.y, p.z] for p in landmarks.landmark[:21]], dtype=np.float64
        )
        maos_validas.append((float(pontos[:, 0].mean()), pontos))

    maos_validas.sort(key=lambda item: item[0])

    maos: list[np.ndarray | None] = [item[1] for item in maos_validas[:2]]
    return maos + [None] * (2 - len(maos))


def extrair_features_rosto_maos(
    resultado_rosto: Any,
    resultado_maos: Any,
    largura: int,
    altura: int,
) -> np.ndarray:
    """Extrai 82 características faciais e de proximidade mão-rosto.

    Aqui as mãos são identificadas pela posição horizontal na imagem
    (esquerda da imagem primeiro). Para usar os mesmos lados do vetor das
    mãos, use :func:`extrair_features_rosto_maos_slots`.

    Args:
        resultado_rosto: ResultadoRosto produzido por DetectorRosto.detectar().
        resultado_maos: Resultado bruto produzido por DetectorMaos.detectar().
        largura: Largura do quadro usado pelos dois detectores.
        altura: Altura do quadro usado pelos dois detectores.

    Returns:
        Vetor float32 com exatamente 82 valores.
    """
    return extrair_features_rosto_maos_slots(
        resultado_rosto, _obter_maos(resultado_maos), largura, altura
    )


def extrair_features_rosto_maos_slots(
    resultado_rosto: Any,
    maos: Sequence[np.ndarray | None],
    largura: int,
    altura: int,
) -> np.ndarray:
    """Extrai 82 características faciais e de proximidade mão-rosto.

    Args:
        resultado_rosto: ResultadoRosto produzido por DetectorRosto.detectar().
        maos: Duas posições ``(esq, dir)``; cada uma é uma matriz (21, 3) de
            coordenadas normalizadas do MediaPipe, ou ``None``.
        largura: Largura do quadro usado pelos dois detectores.
        altura: Altura do quadro usado pelos dois detectores.

    Returns:
        Vetor float32 com exatamente 82 valores.
    """
    if largura <= 0 or altura <= 0:
        raise ValueError("Largura e altura precisam ser positivas.")
    if len(maos) != 2:
        raise ValueError("Informe exatamente duas posições de mão (esq, dir).")

    vetor = np.zeros(TOTAL_FEATURES_ROSTO_MAOS, dtype=np.float32)

    rosto_presente = bool(getattr(resultado_rosto, "presente", False))
    rosto = getattr(resultado_rosto, "landmarks", None)

    vetor[0] = float(
        rosto_presente
        and rosto is not None
        and len(getattr(rosto, "landmark", [])) == 468
    )
    vetor[1] = float(maos[0] is not None)
    vetor[2] = float(maos[1] is not None)

    # Inicializa todas as distâncias como indisponíveis.
    inicio_distancias = 3 + len(INDICES_EXPRESSAO)
    vetor[inicio_distancias:] = DISTANCIA_INDISPONIVEL

    if vetor[0] == 0:
        return vetor

    try:
        ponto_esq = _ponto(rosto, 234, largura, altura)
        ponto_dir = _ponto(rosto, 454, largura, altura)

        largura_rosto = _distancia(ponto_esq, ponto_dir)

        if not np.isfinite(largura_rosto) or largura_rosto < 1.0:
            vetor[0] = 0.0
            return vetor

        expressao = _extrair_expressao(
            rosto,
            largura,
            altura,
            largura_rosto,
        )
        vetor[3:12] = np.asarray(expressao, dtype=np.float32)

        # Converte cada região facial em coordenadas de pixel.
        regioes_pixels: dict[str, np.ndarray] = {}

        for nome, indices in REGIOES.items():
            pontos = np.asarray(
                [_ponto(rosto, i, largura, altura) for i in indices],
                dtype=np.float64,
            )
            regioes_pixels[nome] = pontos

        # Cada mão ocupa 5 dedos x 7 regiões = 35 distâncias.
        cursor = inicio_distancias

        for indice_mao in range(2):
            mao = maos[indice_mao]

            if mao is None:
                cursor += len(PONTAS_DEDOS) * len(NOMES_REGIOES)
                continue

            for indice_dedo in PONTAS_DEDOS:
                dedo = mao[indice_dedo]
                ponto_dedo = np.array(
                    [dedo[0] * largura, dedo[1] * altura],
                    dtype=np.float64,
                )

                for nome_regiao in NOMES_REGIOES:
                    pontos_regiao = regioes_pixels[nome_regiao]

                    distancias = np.linalg.norm(
                        pontos_regiao - ponto_dedo,
                        axis=1,
                    )
                    distancia_minima = float(np.min(distancias))
                    relativa = distancia_minima / largura_rosto

                    vetor[cursor] = float(
                        np.clip(relativa, 0.0, DISTANCIA_INDISPONIVEL)
                    )
                    cursor += 1

    except (IndexError, AttributeError, TypeError, ValueError):
        # Não propaga dados faciais incompletos para o modelo.
        return np.zeros(
            TOTAL_FEATURES_ROSTO_MAOS,
            dtype=np.float32,
        )

    # Segurança: substitui valores inválidos.
    vetor = np.nan_to_num(
        vetor,
        nan=0.0,
        posinf=DISTANCIA_INDISPONIVEL,
        neginf=0.0,
    )

    return vetor


__all__ = [
    "DISTANCIA_INDISPONIVEL",
    "NOMES_REGIOES",
    "REGIOES",
    "TOTAL_FEATURES_ROSTO_MAOS",
    "extrair_features_rosto_maos",
    "extrair_features_rosto_maos_slots",
    "nomes_features_rosto_maos",
]
