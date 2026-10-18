"""Testes da conferência das mãos pelos braços (tira as mãos fantasmas)."""

from types import SimpleNamespace

import numpy as np

from kara_sygna.visao.caracteristicas import indices_dos_slots, separar_maos
from kara_sygna.visao.maos_pelo_corpo import conferir_maos
from kara_sygna.visao.memoria_maos import MemoriaMaos
from kara_sygna.visao.quadro import POSE_DOS_SLOTS
from testes.conftest import resultado_falso
from testes.test_filtro_quadros import Relogio

LARGURA, ALTURA = 640, 480


def _mao(x, y):
    """Mão com o pulso em (x, y) normalizado."""
    pontos = [SimpleNamespace(x=x, y=y - 0.01 * i, z=0.0) for i in range(21)]
    return SimpleNamespace(landmark=pontos)


def _corpo(pulso_esq, pulso_dir, visivel=1.0):
    """Corpo com os ombros em 0,4 e 0,6 e os pulsos dados (slots "esq"/"dir")."""
    marcos = [SimpleNamespace(x=0.5, y=0.5, visibility=visivel) for _ in range(33)]
    ombro_esq, _, p_esq = POSE_DOS_SLOTS["esq"]
    ombro_dir, _, p_dir = POSE_DOS_SLOTS["dir"]
    marcos[ombro_esq] = SimpleNamespace(x=0.4, y=0.4, visibility=visivel)
    marcos[ombro_dir] = SimpleNamespace(x=0.6, y=0.4, visibility=visivel)
    marcos[p_esq] = SimpleNamespace(x=pulso_esq[0], y=pulso_esq[1], visibility=visivel)
    marcos[p_dir] = SimpleNamespace(x=pulso_dir[0], y=pulso_dir[1], visibility=visivel)
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=marcos))


def test_mao_fantasma_longe_dos_pulsos_e_descartada():
    real, fantasma = _mao(0.30, 0.70), _mao(0.36, 0.72)
    resultado = resultado_falso((real, "Left"), (fantasma, "Right"))
    # Braço direito parado lá embaixo, do outro lado.
    conferido = conferir_maos(
        resultado, _corpo((0.31, 0.70), (0.75, 0.95)), LARGURA, ALTURA
    )
    assert conferido.multi_hand_landmarks == [real]
    assert conferido.lados == ["esq"] and conferido.descartadas == 1
    assert separar_maos(conferido)["dir"] is None


def test_o_lado_vem_do_braco_mesmo_com_rotulo_errado():
    esquerda, direita = _mao(0.30, 0.70), _mao(0.70, 0.70)
    # O MediaPipe errou os dois rótulos ("Right" na esquerda e vice-versa).
    resultado = resultado_falso((esquerda, "Right"), (direita, "Left"))
    conferido = conferir_maos(
        resultado, _corpo((0.31, 0.71), (0.69, 0.71)), LARGURA, ALTURA
    )
    slots = indices_dos_slots(conferido)
    assert conferido.multi_hand_landmarks[slots["esq"]] is esquerda
    assert conferido.multi_hand_landmarks[slots["dir"]] is direita


def test_maos_cruzadas_seguem_os_bracos():
    # A mão "esq" foi para o lado direito da imagem (braços cruzados).
    esquerda, direita = _mao(0.65, 0.60), _mao(0.35, 0.60)
    resultado = resultado_falso((direita, "Left"), (esquerda, "Right"))
    conferido = conferir_maos(
        resultado, _corpo((0.64, 0.61), (0.36, 0.61)), LARGURA, ALTURA
    )
    slots = indices_dos_slots(conferido)
    assert conferido.multi_hand_landmarks[slots["esq"]] is esquerda


def test_sem_corpo_ou_ombros_escondidos_nada_muda():
    resultado = resultado_falso((_mao(0.3, 0.7), "Left"))
    assert conferir_maos(resultado, None, LARGURA, ALTURA) is resultado
    escondido = _corpo((0.3, 0.7), (0.7, 0.7), visivel=0.1)
    assert conferir_maos(resultado, escondido, LARGURA, ALTURA) is resultado


def test_memoria_respeita_o_lado_dado_pelos_bracos():
    relogio = Relogio()
    memoria = MemoriaMaos(memoria_s=0.3, relogio=relogio)
    esquerda, direita = _mao(0.30, 0.70), _mao(0.70, 0.70)
    corpo = _corpo((0.31, 0.71), (0.69, 0.71))
    memoria.aplicar(
        conferir_maos(
            resultado_falso((esquerda, "Left"), (direita, "Right")),
            corpo,
            LARGURA,
            ALTURA,
        )
    )
    relogio.agora = 0.1
    so_esq = conferir_maos(resultado_falso((esquerda, "Left")), corpo, LARGURA, ALTURA)
    resultado = memoria.aplicar(so_esq)
    slots = indices_dos_slots(resultado)
    assert resultado.lados == ["esq", "dir"]
    assert resultado.multi_hand_landmarks[slots["dir"]] is direita
    assert np.isclose(separar_maos(resultado)["esq"][0, 0], 0.30)


def test_mao_real_com_pulso_do_pose_errado_nao_e_descartada():
    """No "carro", o Pose põe o pulso longe (até fora da imagem): a mão fica."""
    esquerda, direita = _mao(0.30, 0.80), _mao(0.70, 0.80)
    resultado = resultado_falso((esquerda, "Left"), (direita, "Right"))
    # Pulsos do Pose a ~1 largura de ombro abaixo das mãos (fora da imagem).
    corpo = _corpo((0.30, 1.05), (0.70, 1.05))
    conferido = conferir_maos(resultado, corpo, LARGURA, ALTURA)
    assert conferido.descartadas == 0 and len(conferido.multi_hand_landmarks) == 2


def test_duas_deteccoes_da_mesma_mao_viram_uma():
    mao, quase_igual = _mao(0.40, 0.60), _mao(0.41, 0.61)
    resultado = resultado_falso((mao, "Left"), (quase_igual, "Right"))
    conferido = conferir_maos(
        resultado, _corpo((0.40, 0.61), (0.75, 0.9)), LARGURA, ALTURA
    )
    assert conferido.descartadas == 1 and conferido.lados == ["esq"]
