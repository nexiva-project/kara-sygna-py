"""Regras do programa, sem nenhuma interface gráfica.

A janela (Tk) só mostra o que o controlador decide e repassa os cliques. Assim
dá para testar tudo isto sem câmera, sem MediaPipe e sem tela.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kara_sygna import config
from kara_sygna.aplicacao.erros import (
    ModeloAusenteError,
    ModeloDesatualizadoError,
    PalavraInvalidaError,
    SemDadosError,
)
from kara_sygna.aplicacao.previsualizacao import Reproducao, combinar_espelhos
from kara_sygna.dados.repositorio import (
    RepositorioSinais,
    novo_id_rodada,
    remover_arquivo_se_existir,
)
from kara_sygna.ia import treinamento
from kara_sygna.ia.modelo_io import (
    carregar_modelo,
    classes_do_modelo,
    modelo_esta_atualizado,
)
from kara_sygna.ia.reconhecimento import Estabilizador, prever
from kara_sygna.ia.rotacao import ROTACOES

logger = logging.getLogger(__name__)


class Modo(enum.Enum):
    """O que o programa está fazendo com cada quadro da câmera."""

    PARADO = "parado"
    ENSINANDO = "ensinando"
    RECONHECENDO = "reconhecendo"


class Nivel(enum.Enum):
    """Tom do texto mostrado sobre o vídeo (a interface escolhe a cor)."""

    NEUTRO = "neutro"
    ENSINO = "ensino"
    OK = "ok"
    INCERTO = "incerto"


@dataclass(frozen=True)
class ResultadoQuadro:
    """O que a interface deve mostrar depois de processar um quadro.

    Attributes:
        texto_video: Texto para escrever sobre o vídeo ("" = nada). Só ASCII: o
            OpenCV não desenha acentos.
        nivel: Tom do texto (define a cor).
        status: Nova mensagem para a caixa de status, ou ``None`` para manter.
        sinal_confirmado: Palavra confirmada neste quadro (vai para o texto), ou ``None``.
    """

    texto_video: str = ""
    nivel: Nivel = Nivel.NEUTRO
    status: str | None = None
    sinal_confirmado: str | None = None


@dataclass(frozen=True)
class InfoSinal:
    """Uma linha da lista de sinais aprendidos."""

    palavra: str
    exemplos: int
    treinado: bool


class ControladorSinais:
    """Coordena ensino, treinamento e reconhecimento.

    Args:
        repositorio: Onde ficam os exemplos (CSV).
        caminho_modelo: Arquivo ``.pkl`` do modelo.
        carregador_modelo: Função que carrega o modelo (trocada nos testes).
    """

    def __init__(
        self,
        repositorio: RepositorioSinais | None = None,
        caminho_modelo: Path = config.ARQUIVO_MODELO,
        carregador_modelo: Callable[[Path], Any] = carregar_modelo,
    ) -> None:
        """Guarda as dependências e começa no modo parado (veja a classe para os argumentos)."""
        self.repositorio = repositorio or RepositorioSinais()
        self.caminho_modelo = Path(caminho_modelo)
        self._carregador_modelo = carregador_modelo
        self.estabilizador = Estabilizador()
        self.modo = Modo.PARADO
        self.rotulo_atual = ""
        self.total_gravado = 0
        self._rodada = 0
        self._modelo: Any = None

    # -- ensino ----------------------------------------------------------------

    def iniciar_ensino(self, palavra: str) -> str:
        """Começa a gravar exemplos de um sinal.

        Args:
            palavra: Nome do sinal (é normalizado para minúsculas).

        Returns:
            Mensagem para o status.

        Raises:
            PalavraInvalidaError: Se a palavra estiver vazia.
        """
        palavra = palavra.strip().lower()
        if not palavra:
            raise PalavraInvalidaError("Escreva a palavra do sinal antes de começar.")
        self.repositorio.garantir_arquivo()
        self.rotulo_atual = palavra
        self.total_gravado = 0
        self._rodada = novo_id_rodada()
        self.modo = Modo.ENSINANDO
        logger.info("Ensino iniciado: %s", palavra)
        return f"Ensinando “{palavra}”. Faça o sinal diante da câmera."

    def parar(self) -> str:
        """Para o que estiver em andamento.

        Returns:
            Mensagem para o status.
        """
        anterior, self.modo = self.modo, Modo.PARADO
        if anterior is Modo.ENSINANDO:
            logger.info("Ensino parado: %d linhas gravadas.", self.total_gravado)
            return f"Ensino de “{self.rotulo_atual}” parado: {self.total_gravado} exemplos gravados."
        if anterior is Modo.RECONHECENDO:
            logger.info("Reconhecimento parado.")
            return "Reconhecimento parado."
        return "Programa parado."

    # -- treinamento -------------------------------------------------------------

    def treinar(
        self, progresso: Callable[[str], None] = lambda _m: None
    ) -> treinamento.ResultadoTreino:
        """Treina a IA com o CSV atual (demora; chame em segundo plano).

        Raises:
            SemDadosError: Se não houver exemplos gravados.
            TreinoImpossivelError: Se os dados não bastarem (ex.: só 1 sinal).
        """
        if not self.repositorio.existe():
            raise SemDadosError("Ensine pelo menos dois sinais antes de treinar a IA.")
        dados = self.repositorio.carregar()
        resultado = treinamento.treinar(dados, self.caminho_modelo, progresso)
        self._modelo = None  # o modelo em memória ficou velho
        return resultado

    # -- reconhecimento ----------------------------------------------------------

    def iniciar_reconhecimento(self) -> str:
        """Carrega o modelo e começa a reconhecer.

        Returns:
            Mensagem para o status.

        Raises:
            SemDadosError: Se não houver CSV.
            ModeloAusenteError: Se a IA ainda não foi treinada.
            ModeloDesatualizadoError: Se os sinais mudaram depois do treino.
            ModeloInvalidoError: Se o arquivo do modelo for incompatível.
        """
        if not self.repositorio.existe():
            raise SemDadosError("Ensine pelo menos um sinal primeiro.")
        if not self.caminho_modelo.exists():
            raise ModeloAusenteError("Ensine sinais e clique em TREINAR IA primeiro.")
        if not modelo_esta_atualizado(self.caminho_modelo, self.repositorio.caminho):
            raise ModeloDesatualizadoError(
                "Os sinais mudaram.\n\nClique em TREINAR IA antes de reconhecer."
            )

        self._modelo = self._carregador_modelo(self.caminho_modelo)
        self.estabilizador.reiniciar()
        self.modo = Modo.RECONHECENDO
        logger.info(
            "Reconhecimento iniciado. Sinais do modelo: %s", list(self._modelo.classes_)
        )
        return "Reconhecendo sinais ao vivo..."

    def limpar_reconhecimento(self) -> None:
        """Esquece o histórico de votos (use junto com "limpar texto")."""
        self.estabilizador.reiniciar()

    # -- quadro a quadro -----------------------------------------------------------

    def processar_quadro(self, vetor: np.ndarray | None) -> ResultadoQuadro:
        """Trata um quadro da câmera de acordo com o modo atual.

        Args:
            vetor: As 126 características do quadro, ou ``None`` se não há mão.

        Returns:
            O que a interface deve mostrar.
        """
        if self.modo is Modo.ENSINANDO:
            return self._quadro_de_ensino(vetor)
        if self.modo is Modo.RECONHECENDO:
            return self._quadro_de_reconhecimento(vetor)
        return ResultadoQuadro()

    def _quadro_de_ensino(self, vetor: np.ndarray | None) -> ResultadoQuadro:
        """Grava o quadro, se houver mão."""
        if vetor is None:
            return ResultadoQuadro()
        self.total_gravado += self.repositorio.anexar_quadro(
            self.rotulo_atual, self._rodada, vetor
        )
        return ResultadoQuadro(
            texto_video=f"ENSINANDO: {self.rotulo_atual} ({self.total_gravado})",
            nivel=Nivel.ENSINO,
            status=f"Ensinando “{self.rotulo_atual}”: {self.total_gravado} exemplos.",
        )

    def _quadro_de_reconhecimento(self, vetor: np.ndarray | None) -> ResultadoQuadro:
        """Prevê o sinal e atualiza a votação."""
        if vetor is None:
            self.estabilizador.atualizar(None)
            return ResultadoQuadro("MOSTRE UM SINAL")

        rotacoes = ROTACOES if config.CORRECAO_ROTACAO_ATIVA else None
        predicao = prever(self._modelo, vetor, rotacoes)
        confirmado = self.estabilizador.atualizar(
            predicao.sinal if predicao.aceito else None
        )
        if confirmado in config.ROTULOS_IGNORADOS:
            confirmado = None

        if not predicao.aceito:
            return ResultadoQuadro(
                f"INCERTO ({predicao.confianca:.0%})",
                Nivel.INCERTO,
                sinal_confirmado=confirmado,
            )

        texto = f"SINAL: {predicao.sinal} ({predicao.confianca:.0%})"
        if any(predicao.rotacao):
            rx, ry, rz = predicao.rotacao
            texto += f" [ajuste {rx:+d},{ry:+d},{rz:+d} graus]"
        return ResultadoQuadro(
            texto,
            Nivel.OK,
            f"Sinal reconhecido: {predicao.sinal} ({predicao.confianca:.0%})",
            confirmado,
        )

    # -- lista de sinais -----------------------------------------------------------

    def listar_sinais(self) -> list[InfoSinal]:
        """Sinais gravados, com a quantidade de exemplos e se a IA já os conhece."""
        contagem = self.repositorio.contagem_por_sinal()
        if not contagem:
            return []
        atualizado = modelo_esta_atualizado(
            self.caminho_modelo, self.repositorio.caminho
        )
        conhecidos = classes_do_modelo(self.caminho_modelo) if atualizado else set()
        return [
            InfoSinal(palavra, qtd, palavra in conhecidos)
            for palavra, qtd in contagem.items()
        ]

    def apagar_sinal(self, palavra: str) -> str:
        """Apaga um sinal e o modelo (que ficou desatualizado).

        Returns:
            Mensagem para o status.
        """
        self.repositorio.apagar_sinal(palavra)
        remover_arquivo_se_existir(self.caminho_modelo)
        self._modelo = None
        self.modo = Modo.PARADO
        return f"Sinal “{palavra}” apagado. Treine a IA novamente."

    def criar_reproducao(self, palavra: str) -> Reproducao:
        """Prepara a pré-visualização de um sinal gravado.

        Raises:
            SemDadosError: Se não houver exemplos desse sinal.
        """
        exemplos = self.repositorio.exemplos_do_sinal(palavra)
        if len(exemplos) == 0:
            raise SemDadosError(f"Não há exemplos gravados para “{palavra}”.")
        return Reproducao(palavra, combinar_espelhos(exemplos))
