"""Abertura e leitura da webcam (Windows, Linux e macOS).

Por que este módulo existe
--------------------------
O código antigo abria a câmera sempre com ``cv2.CAP_DSHOW``. Esse *backend*
(DirectShow) só existe no Windows, então no Ubuntu a câmera nunca abria. Aqui
cada sistema operacional tem a sua lista de backends, e cada combinação
(backend, índice) só é aceita se entregar um quadro de verdade.
"""

from __future__ import annotations

import logging
import platform
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Self

import cv2
import numpy as np

from kara_sygna import config

logger = logging.getLogger(__name__)

#: Função que cria uma captura a partir de ``(indice, codigo_do_backend)``.
FabricaCaptura = Callable[[int, int], "cv2.VideoCapture"]


class CameraIndisponivelError(RuntimeError):
    """Nenhuma câmera pôde ser aberta (ou nenhuma entregou imagem)."""


@dataclass(frozen=True)
class Backend:
    """Um backend de captura do OpenCV.

    Attributes:
        nome: Nome legível, usado nas mensagens e no log.
        codigo: Constante ``cv2.CAP_*`` correspondente.
    """

    nome: str
    codigo: int


def backends_da_plataforma(sistema: str | None = None) -> list[Backend]:
    """Lista os backends de câmera adequados ao sistema operacional.

    Args:
        sistema: Resultado de ``platform.system()`` ("Windows", "Linux",
            "Darwin"). Se omitido, é detectado automaticamente.

    Returns:
        Backends em ordem de preferência. O último é sempre o automático do
        OpenCV, usado como último recurso.
    """
    sistema = sistema or platform.system()
    automatico = Backend("automático", cv2.CAP_ANY)
    if sistema == "Windows":
        # MediaFoundation primeiro: em webcams comuns ele entrega 30 quadros
        # por segundo, contra ~16 do DirectShow. Com mais quadros a mão anda
        # menos entre um e outro e o MediaPipe a acompanha melhor.
        return [
            Backend("MediaFoundation", cv2.CAP_MSMF),
            Backend("DirectShow", cv2.CAP_DSHOW),
            automatico,
        ]
    if sistema == "Linux":
        return [Backend("V4L2", cv2.CAP_V4L2), automatico]
    if sistema == "Darwin":
        return [Backend("AVFoundation", cv2.CAP_AVFOUNDATION), automatico]
    return [automatico]


def recortar_proporcao(quadro: np.ndarray, proporcao: float) -> np.ndarray:
    """Recorta o centro do quadro para a proporção ``largura / altura`` pedida.

    As características das mãos usam coordenadas relativas à largura e à
    altura da imagem. Um quadro 16:9 "achata" a mão em relação aos dados
    gravados em 4:3, então todo quadro precisa ter a mesma proporção.

    Args:
        quadro: Imagem BGR.
        proporcao: Proporção desejada (ex.: ``640 / 480``).

    Returns:
        Uma *view* recortada do quadro (ou o próprio quadro, se já estiver
        na proporção certa).
    """
    altura, largura = quadro.shape[:2]
    if abs(largura / altura - proporcao) < 0.01:
        return quadro
    if largura / altura > proporcao:
        nova_largura = round(altura * proporcao)
        inicio = (largura - nova_largura) // 2
        return quadro[:, inicio : inicio + nova_largura]
    nova_altura = round(largura / proporcao)
    inicio = (altura - nova_altura) // 2
    return quadro[inicio : inicio + nova_altura]


class Camera:
    """Webcam já aberta e validada.

    Pode ser usada como gerenciador de contexto (``with``), que libera a
    câmera automaticamente ao sair do bloco.
    """

    def __init__(
        self,
        captura: cv2.VideoCapture,
        indice: int,
        backend: Backend,
        largura: int,
        altura: int,
        proporcao: float | None = None,
    ) -> None:
        """Cria o invólucro.

        Args:
            captura: Objeto ``cv2.VideoCapture`` já aberto.
            indice: Índice da câmera no sistema (0, 1, ...).
            backend: Backend com o qual a câmera foi aberta.
            largura: Largura real dos quadros entregues pela câmera, em pixels.
            altura: Altura real dos quadros entregues pela câmera, em pixels.
            proporcao: Se informada, cada quadro é recortado no centro para
                esta proporção (``largura / altura``). ``None`` não recorta.
        """
        self._captura = captura
        self.indice = indice
        self.backend = backend
        self.proporcao = proporcao
        self.largura_original = largura
        self.altura_original = altura
        # Tamanho dos quadros que ``ler`` devolve (depois do recorte).
        if proporcao:
            recortado = recortar_proporcao(np.empty((altura, largura, 0)), proporcao)
            self.altura, self.largura = recortado.shape[:2]
        else:
            self.largura, self.altura = largura, altura

    @property
    def recortada(self) -> bool:
        """Informa se os quadros são recortados para manter a proporção."""
        return (self.largura, self.altura) != (
            self.largura_original,
            self.altura_original,
        )

    @property
    def descricao(self) -> str:
        """Texto curto como ``"câmera 0 via V4L2 (640x480)"``."""
        tamanho = f"{self.largura_original}x{self.altura_original}"
        if self.recortada:
            tamanho += f", recortada para {self.largura}x{self.altura}"
        return f"câmera {self.indice} via {self.backend.nome} ({tamanho})"

    def ler(self) -> np.ndarray | None:
        """Lê o próximo quadro.

        Returns:
            Imagem BGR (``numpy.ndarray``) ou ``None`` se a leitura falhou.
        """
        sucesso, quadro = self._captura.read()
        if not sucesso or quadro is None:
            return None
        if self.proporcao:
            # Cópia contínua na memória: o MediaPipe e o Tk não aceitam
            # bem uma "fatia" com saltos entre as linhas.
            quadro = np.ascontiguousarray(recortar_proporcao(quadro, self.proporcao))
        return quadro

    def esta_aberta(self) -> bool:
        """Informa se a câmera continua aberta."""
        return bool(self._captura.isOpened())

    def liberar(self) -> None:
        """Libera a câmera. Pode ser chamado mais de uma vez."""
        self._captura.release()

    def __enter__(self) -> Self:
        """Entra no bloco ``with`` devolvendo a própria câmera."""
        return self

    def __exit__(self, *_excecao) -> None:
        """Libera a câmera ao sair do bloco ``with``, mesmo se houve erro."""
        self.liberar()


def _fabrica_padrao(indice: int, codigo_backend: int) -> cv2.VideoCapture:
    """Cria uma ``cv2.VideoCapture`` real (a fábrica usada fora dos testes)."""
    return cv2.VideoCapture(indice, codigo_backend)


def _aplicar_resolucao(
    captura: cv2.VideoCapture, largura: int | None, altura: int | None
) -> None:
    """Pede à câmera a resolução desejada (o driver pode escolher outra)."""
    if largura and altura:
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)


def _ler_com_tentativas(
    captura: cv2.VideoCapture, tentativas: int, pausa_s: float
) -> np.ndarray | None:
    """Tenta ler um quadro algumas vezes.

    Em muitas câmeras Linux os primeiros quadros chegam vazios enquanto o
    sensor "esquenta", então uma única leitura não basta para validar.
    """
    for _ in range(max(1, tentativas)):
        sucesso, quadro = captura.read()
        if sucesso and quadro is not None:
            return quadro
        time.sleep(pausa_s)
    return None


def _tentar_abrir(
    fabrica: FabricaCaptura,
    indice: int,
    backend: Backend,
    largura: int | None,
    altura: int | None,
    tentativas_leitura: int,
    pausa_s: float,
    falhas: list[str],
) -> Camera | None:
    """Tenta abrir uma combinação (backend, índice).

    Args:
        fabrica: Cria a captura.
        indice: Índice da câmera.
        backend: Backend a usar.
        largura: Largura pedida (ou ``None``).
        altura: Altura pedida (ou ``None``).
        tentativas_leitura: Quantas leituras tentar antes de desistir.
        pausa_s: Pausa entre leituras, em segundos.
        falhas: Lista que recebe uma linha explicando cada falha.

    Returns:
        ``Camera`` pronta, ou ``None`` se esta combinação não funcionou.
    """
    rotulo = f"câmera {indice} ({backend.nome})"
    try:
        captura = fabrica(indice, backend.codigo)
    except Exception as erro:  # noqa: BLE001 - drivers nativos podem lançar qualquer coisa
        falhas.append(f"{rotulo}: erro ao abrir ({erro})")
        return None

    if not captura.isOpened():
        captura.release()
        falhas.append(f"{rotulo}: não abriu")
        return None

    _aplicar_resolucao(captura, largura, altura)
    quadro = _ler_com_tentativas(captura, tentativas_leitura, pausa_s)
    if quadro is None:
        captura.release()
        falhas.append(f"{rotulo}: abriu, mas não entregou imagem")
        return None

    altura_real, largura_real = quadro.shape[:2]
    # A câmera pode ignorar a resolução pedida (ex.: só entrega 16:9).
    # Os quadros são recortados para a proporção pedida, a mesma dos dados.
    proporcao = largura / altura if largura and altura else None
    return Camera(captura, indice, backend, largura_real, altura_real, proporcao)


def _mensagem_de_falha(falhas: Sequence[str], sistema: str) -> str:
    """Monta a mensagem de erro final, com dicas específicas do sistema."""
    detalhes = (
        "\n".join(f"  - {linha}" for linha in falhas[:12])
        or "  (nenhuma tentativa foi feita)"
    )
    dica_linux = (
        "\nNo Linux, confira:\n"
        "  - 'ls /dev/video*' mostra algum dispositivo?\n"
        "  - 'v4l2-ctl --list-devices' (pacote v4l-utils) lista a webcam?\n"
        "  - seu usuário tem permissão? ('groups' deve incluir 'video'; "
        "senão: sudo usermod -aG video $USER e entre de novo na sessão)"
        if sistema == "Linux"
        else ""
    )
    return (
        "Não foi possível acessar nenhuma câmera.\n"
        f"Tentativas:\n{detalhes}\n"
        "Verifique se a câmera está conectada e se outro programa "
        f"(Zoom, Teams, navegador) não está usando ela.{dica_linux}"
    )


def abrir_camera(
    indices: Iterable[int] | None = None,
    backends: Sequence[Backend] | None = None,
    largura: int | None = config.LARGURA_CAPTURA,
    altura: int | None = config.ALTURA_CAPTURA,
    fabrica: FabricaCaptura | None = None,
    tentativas_leitura: int = config.LEITURAS_DE_TESTE,
    pausa_s: float = config.PAUSA_ENTRE_LEITURAS_S,
) -> Camera:
    """Abre a primeira câmera que realmente entregar imagem.

    Percorre cada backend da plataforma e, dentro dele, cada índice de câmera.
    Uma combinação só é aceita se conseguir ler um quadro.

    Args:
        indices: Índices a testar. Padrão: ``config.INDICES_CAMERA``.
        backends: Backends a testar. Padrão: os da plataforma atual.
        largura: Largura pedida à câmera (``None`` para não forçar).
        altura: Altura pedida à câmera (``None`` para não forçar).
        fabrica: Cria capturas. Só é trocada nos testes automatizados.
        tentativas_leitura: Leituras tentadas por combinação.
        pausa_s: Pausa entre as leituras, em segundos.

    Returns:
        Uma ``Camera`` pronta para uso.

    Raises:
        CameraIndisponivelError: Se nenhuma combinação funcionou. A mensagem
            lista cada tentativa e dá dicas de correção.
    """
    indices = tuple(config.INDICES_CAMERA if indices is None else indices)
    backends = list(backends) if backends is not None else backends_da_plataforma()
    fabrica = fabrica or _fabrica_padrao
    falhas: list[str] = []

    for backend in backends:
        for indice in indices:
            camera = _tentar_abrir(
                fabrica,
                indice,
                backend,
                largura,
                altura,
                tentativas_leitura,
                pausa_s,
                falhas,
            )
            if camera is not None:
                logger.info("Câmera conectada: %s", camera.descricao)
                if camera.recortada:
                    logger.warning(
                        "A câmera não aceitou %dx%d; a imagem será recortada no "
                        "centro para manter a proporção dos sinais gravados.",
                        largura,
                        altura,
                    )
                return camera
            logger.debug("%s", falhas[-1])

    raise CameraIndisponivelError(_mensagem_de_falha(falhas, platform.system()))
