"""Testes do log e do desenho 3D."""

import logging

import numpy as np

from kara_sygna.logs import FormatadorKara, categoria_do_logger, configurar_logging
from kara_sygna.visao.dedos import dedos_levantados
from kara_sygna.visao.render_3d import desenhar_quadro_sinal
from testes.conftest import vetor_de_duas_maos, vetor_de_uma_mao


def test_categorias_pelo_nome_do_modulo():
    assert categoria_do_logger("kara_sygna.camera.captura") == "CAMERA"
    assert categoria_do_logger("kara_sygna.ia.treinamento") == "MODEL"
    assert categoria_do_logger("kara_sygna.dados.repositorio") == "DATA"
    assert categoria_do_logger("kara_sygna.visao.detector_maos") == "VISION"
    assert categoria_do_logger("kara_sygna.interface.janela_principal") == "APP"
    assert categoria_do_logger("outra.coisa") == "APP"


def test_formato_da_linha_de_log():
    registro = logging.LogRecord(
        "kara_sygna.ia.x", logging.INFO, "", 0, "treinando", None, None
    )
    linha = FormatadorKara().format(registro)
    assert linha.endswith("[MODEL] treinando") and linha[2] == ":"


def test_configurar_logging_nao_duplica_handler():
    configurar_logging()
    configurar_logging()
    manipuladores = [
        h
        for h in logging.getLogger("kara_sygna").handlers
        if getattr(h, "_kara_console", False)
    ]
    assert len(manipuladores) == 1


def test_render_devolve_imagem_com_as_maos(rng):
    vazio = desenhar_quadro_sinal(np.zeros(126), (600, 500), 0.0, "x")
    com_maos = desenhar_quadro_sinal(vetor_de_duas_maos(rng), (600, 500), 0.0, "x")
    assert vazio.shape == com_maos.shape == (500, 600, 3) and vazio.dtype == np.uint8
    assert not np.array_equal(vazio, com_maos)


def test_render_aceita_uma_mao_so(rng):
    assert desenhar_quadro_sinal(
        vetor_de_uma_mao(rng, "esq"), (300, 250), 1.0, "oi"
    ).shape == (250, 300, 3)


def test_dedos_levantados_mao_aberta_e_fechada():
    from types import SimpleNamespace as N

    def mao(pontas_longe: bool):
        pts = [N(x=0.5, y=0.9) for _ in range(21)]  # tudo perto do pulso
        for ponta, junta in ((8, 6), (12, 10), (16, 14), (20, 18)):
            pts[junta] = N(x=0.5, y=0.7)
            pts[ponta] = N(x=0.5, y=0.3 if pontas_longe else 0.75)
        pts[3], pts[4], pts[17] = N(x=0.45, y=0.8), N(x=0.45, y=0.8), N(x=0.6, y=0.85)
        return N(landmark=pts)

    assert set(dedos_levantados(mao(True))) >= {
        "indicador",
        "medio",
        "anelar",
        "mindinho",
    }
    assert dedos_levantados(mao(False)) == []
