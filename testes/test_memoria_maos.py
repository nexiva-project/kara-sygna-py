"""Testes da memória curta das mãos."""

from kara_sygna.visao.memoria_maos import MemoriaMaos
from testes.conftest import mao_falsa, resultado_falso
from testes.test_filtro_quadros import Relogio


def _memoria(relogio):
    return MemoriaMaos(memoria_s=0.3, max_maos=2, relogio=relogio)


def _n(resultado):
    return len(resultado.multi_hand_landmarks or [])


def test_mao_que_pisca_continua_valendo_por_pouco_tempo(rng):
    relogio = Relogio()
    memoria = _memoria(relogio)
    esq, dir_ = mao_falsa(rng, -0.3), mao_falsa(rng, 0.3)
    assert _n(memoria.aplicar(resultado_falso((esq, "Left"), (dir_, "Right")))) == 2

    relogio.agora = 0.1  # a direita sumiu por um quadro
    resultado = memoria.aplicar(resultado_falso((esq, "Left")))
    assert _n(resultado) == 2 and memoria.lembradas == 1
    assert dir_ in resultado.multi_hand_landmarks

    relogio.agora = 0.5  # sumiu há mais de 0,3 s: não vale mais
    assert _n(memoria.aplicar(resultado_falso((esq, "Left")))) == 1
    assert memoria.lembradas == 0


def test_lembra_a_mao_certa_mesmo_com_rotulo_trocado(rng):
    relogio = Relogio()
    memoria = _memoria(relogio)
    esq, dir_ = mao_falsa(rng, -0.3), mao_falsa(rng, 0.3)
    memoria.aplicar(resultado_falso((esq, "Left"), (dir_, "Right")))
    relogio.agora = 0.05
    # Só a esquerda aparece, mas com o rótulo errado: a lembrada é a direita.
    resultado = memoria.aplicar(resultado_falso((esq, "Right")))
    assert resultado.multi_hand_landmarks == [esq, dir_]


def test_nao_duplica_a_mao_que_esta_no_mesmo_lugar(rng):
    relogio = Relogio()
    memoria = _memoria(relogio)
    a, b = mao_falsa(rng, 0.0), mao_falsa(rng, 0.02)
    memoria.aplicar(resultado_falso((a, "Left"), (b, "Right")))
    relogio.agora = 0.05
    resultado = memoria.aplicar(resultado_falso((a, "Left")))
    assert _n(resultado) == 1


def test_sem_nenhuma_mao_lembra_as_duas_e_zero_desliga(rng):
    relogio = Relogio()
    memoria = _memoria(relogio)
    esq, dir_ = mao_falsa(rng, -0.3), mao_falsa(rng, 0.3)
    memoria.aplicar(resultado_falso((esq, "Left"), (dir_, "Right")))
    relogio.agora = 0.1
    assert _n(memoria.aplicar(resultado_falso())) == 2

    desligada = MemoriaMaos(memoria_s=0, relogio=relogio)
    desligada.aplicar(resultado_falso((esq, "Left")))
    relogio.agora = 0.15
    assert _n(desligada.aplicar(resultado_falso())) == 0


def test_mao_lembrada_mantem_os_world_landmarks(rng):
    relogio = Relogio()
    memoria = _memoria(relogio)
    esq, dir_ = mao_falsa(rng, -0.3), mao_falsa(rng, 0.3)
    mundo_esq, mundo_dir = mao_falsa(rng), mao_falsa(rng)
    primeiro = resultado_falso((esq, "Left"), (dir_, "Right"))
    primeiro.multi_hand_world_landmarks = [mundo_esq, mundo_dir]
    memoria.aplicar(primeiro)
    relogio.agora = 0.1
    segundo = resultado_falso((esq, "Left"))
    segundo.multi_hand_world_landmarks = [mundo_esq]
    resultado = memoria.aplicar(segundo)
    assert resultado.multi_hand_world_landmarks == [mundo_esq, mundo_dir]
