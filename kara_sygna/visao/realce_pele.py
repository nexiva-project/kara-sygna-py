"""Realce de pele: a imagem que vai para o detector de mãos fica com a pele
clara e todo o resto escuro.

Por que existe
--------------
Em sinais com as mãos fechadas e em movimento (ex.: "carro"), os punhos
borram e se misturam com a roupa, e o MediaPipe perde as mãos. Escurecendo
tudo que não é pele, as mãos se destacam: no teste com o vídeo do "carro",
os quadros sem nenhuma mão caíram de 56% para 31%. Escurecer só o fundo da
sala não ajudou, porque o problema é a roupa atrás das mãos.

Como a pele é reconhecida
-------------------------
Pela cor (Cr, Cb do espaço YCrCb), que muda pouco com o brilho. A cor é
medida no próprio rosto (testa e maçãs do rosto, pelo FaceMesh), e vira uma
elipse estreita em volta dela: assim funciona para qualquer tom de pele e
qualquer luz. A elipse precisa ser estreita; com folga, uma roupa cinza
(Cr, Cb perto de 128) já entra como pele. Sem rosto, vale a última cor
medida ou, no começo, uma faixa padrão.

Só o detector de mãos recebe a imagem realçada: a tela e o rosto usam a
imagem original.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from kara_sygna import config

#: Faixa padrão de pele (Cr, Cb), usada até medir o primeiro rosto.
FAIXA_PADRAO = ((135, 180), (85, 135))
#: Pontos do FaceMesh com pele à mostra mesmo com barba ou óculos: testa e
#: maçãs do rosto, logo abaixo dos olhos.
PONTOS_PELE = (151, 108, 337, 117, 118, 346, 347)
_RAIO_AMOSTRA = 4
#: Tamanho da elipse, em desvios-padrão da cor do rosto, e desvio mínimo
#: (em níveis de cor). Valores testados no vídeo do "carro": com k maior ou
#: piso maior, a roupa entrava como pele e o realce deixava de ajudar.
_K = 2.5
_PISO = 2.0
#: Quanto da nova medida entra a cada quadro (o resto é a medida anterior).
_SUAVIZACAO = 0.1
#: Brilho mínimo (Y) para contar como pele: sombras muito escuras ficam de fora.
_BRILHO_MINIMO = 30
#: A tabela de cores é refeita a cada tantas medidas (refazer custa ~1,5 ms).
_REFAZER_TABELA_A_CADA = 15


def _tabela_faixa(faixa) -> np.ndarray:
    """Tabela 256x256 (Cr, Cb) -> 255 se a cor está na faixa retangular."""
    (cr_min, cr_max), (cb_min, cb_max) = faixa
    cr, cb = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    dentro = (cr >= cr_min) & (cr <= cr_max) & (cb >= cb_min) & (cb <= cb_max)
    return dentro.astype(np.uint8) * 255


def _tabela_elipse(media: np.ndarray, covariancia: np.ndarray) -> np.ndarray:
    """Tabela 256x256 (Cr, Cb) -> 255 se a cor está dentro da elipse."""
    inversa = np.linalg.inv(covariancia)
    cr, cb = np.meshgrid(np.arange(256.0), np.arange(256.0), indexing="ij")
    d_cr, d_cb = cr - media[0], cb - media[1]
    distancia2 = (
        inversa[0, 0] * d_cr**2
        + 2 * inversa[0, 1] * d_cr * d_cb
        + inversa[1, 1] * d_cb**2
    )
    return (distancia2 < _K**2).astype(np.uint8) * 255


class RealcePele:
    """Escurece tudo que não é pele, com a cor de pele medida no rosto.

    Args:
        ativo: Liga o realce. Padrão: ``config.REALCE_PELE``.
        escuro: Quanto sobra do que não é pele (0 = preto, 1 = sem mudança).
            Padrão: ``config.REALCE_PELE_ESCURO``.
    """

    def __init__(self, ativo: bool | None = None, escuro: float | None = None) -> None:
        """Cria o realce com a faixa padrão (veja a classe para os argumentos)."""
        self.ativo = config.REALCE_PELE if ativo is None else bool(ativo)
        self.escuro = float(config.REALCE_PELE_ESCURO if escuro is None else escuro)
        if not 0.0 <= self.escuro <= 1.0:
            raise ValueError("escuro precisa ficar entre 0 e 1.")
        self._tabela = _tabela_faixa(FAIXA_PADRAO).ravel()
        self.media: np.ndarray | None = None
        self.covariancia: np.ndarray | None = None
        self._medidas = 0

    @property
    def calibrado(self) -> bool:
        """True depois que a cor foi medida em algum rosto."""
        return self.media is not None

    def calibrar(self, quadro_bgr: np.ndarray, resultado_rosto: Any) -> None:
        """Mede a cor da pele no rosto e ajusta a elipse aos poucos.

        Args:
            quadro_bgr: Quadro original (o mesmo enviado ao FaceMesh).
            resultado_rosto: ``ResultadoRosto``; sem rosto, nada muda.
        """
        if not self.ativo or not getattr(resultado_rosto, "presente", False):
            return
        pontos = getattr(getattr(resultado_rosto, "landmarks", None), "landmark", None)
        if pontos is None or len(pontos) <= max(PONTOS_PELE):
            return

        altura, largura = quadro_bgr.shape[:2]
        r = _RAIO_AMOSTRA
        amostras = []
        for indice in PONTOS_PELE:
            x = round(pontos[indice].x * largura)
            y = round(pontos[indice].y * altura)
            if r <= x < largura - r and r <= y < altura - r:
                amostras.append(
                    quadro_bgr[y - r : y + r + 1, x - r : x + r + 1].reshape(-1, 3)
                )
        if not amostras:
            return

        pixels = np.concatenate(amostras)[None].astype(np.uint8)
        ycrcb = cv2.cvtColor(pixels, cv2.COLOR_BGR2YCrCb)[0].astype(np.float64)
        cores = ycrcb[ycrcb[:, 0] >= _BRILHO_MINIMO][:, 1:]
        if len(cores) < 20:
            return

        media = cores.mean(axis=0)
        covariancia = np.cov(cores.T) + np.eye(2) * _PISO**2
        if self.media is None:
            self.media, self.covariancia = media, covariancia
        else:
            self.media = (1 - _SUAVIZACAO) * self.media + _SUAVIZACAO * media
            self.covariancia = (
                1 - _SUAVIZACAO
            ) * self.covariancia + _SUAVIZACAO * covariancia
        if self._medidas % _REFAZER_TABELA_A_CADA == 0:
            self._tabela = _tabela_elipse(self.media, self.covariancia).ravel()
        self._medidas += 1

    def mascara(self, quadro_bgr: np.ndarray) -> np.ndarray:
        """Máscara suave de onde há pele: ``uint8`` de 0 a 255, (altura, largura)."""
        ycrcb = cv2.cvtColor(quadro_bgr, cv2.COLOR_BGR2YCrCb)
        indices = (ycrcb[..., 1].astype(np.uint16) << 8) | ycrcb[..., 2]
        mascara = np.take(self._tabela, indices)
        mascara[ycrcb[..., 0] < _BRILHO_MINIMO] = 0
        # Engorda e suaviza: as bordas borradas da mão também ficam claras.
        mascara = cv2.dilate(mascara, np.ones((9, 9), np.uint8))
        return cv2.GaussianBlur(mascara, (21, 21), 0)

    def aplicar(self, quadro_bgr: np.ndarray) -> np.ndarray:
        """Devolve uma cópia com a pele clara e o resto escuro (ou o original)."""
        if not self.ativo:
            return quadro_bgr
        mascara = self.mascara(quadro_bgr)
        # fator = escuro + (1 - escuro) * mascara, em 0..255
        piso = round(self.escuro * 255)
        fator = cv2.addWeighted(mascara, 1.0 - self.escuro, mascara, 0.0, piso)
        return cv2.multiply(quadro_bgr, cv2.merge([fator, fator, fator]), scale=1 / 255)
