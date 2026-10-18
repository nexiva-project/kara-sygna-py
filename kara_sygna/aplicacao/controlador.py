"""Regras do programa, sem nenhuma interface gráfica."""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from kara_sygna import config
from kara_sygna.aplicacao.diagnostico import DiagnosticoReconhecimento
from kara_sygna.aplicacao.erros import (
    ModeloAusenteError,
    ModeloDesatualizadoError,
    PalavraInvalidaError,
    SemDadosError,
)
from kara_sygna.aplicacao.filtro_quadros import FiltroQuadros
from kara_sygna.aplicacao.previsualizacao import Reproducao, completar_com_espelho
from kara_sygna.aplicacao.qualidade import QualidadeGravacao
from kara_sygna.dados.repositorio import (
    TIPO_PRE,
    TIPO_SINAL,
    RepositorioSinais,
    nome_pre_sinal,
    novo_id_rodada,
    remover_arquivo_se_existir,
)
from kara_sygna.ia import treinamento
from kara_sygna.ia.modelo_io import (
    carregar_modelo,
    classes_do_modelo,
    modelo_esta_atualizado,
)
from kara_sygna.ia.pre_sinais import (
    ModeloPreSinais,
    caminho_modelo_pre,
    carregar_pre_sinais,
    nomes_dos_pre_sinais,
    treinar_pre_sinais,
)
from kara_sygna.ia.reconhecimento import Estabilizador, prever
from kara_sygna.ia.sequencia import BufferQuadros
from kara_sygna.visao.preenchimento import preencher_buracos
from kara_sygna.visao.quadro import (
    COLUNA_CORPO_PRESENTE,
    COLUNA_ROSTO_PRESENTE,
    maos_presentes,
    tem_mao,
)

logger = logging.getLogger(__name__)


class Modo(enum.Enum):
    """O que o programa está fazendo com cada quadro da câmera."""

    PARADO = "parado"
    ENSINANDO = "ensinando"
    RECONHECENDO = "reconhecendo"


class Nivel(enum.Enum):
    """Tom do texto mostrado sobre o vídeo."""

    NEUTRO = "neutro"
    ENSINO = "ensino"
    OK = "ok"
    INCERTO = "incerto"


@dataclass(frozen=True)
class ResultadoQuadro:
    """Resultado do processamento de um quadro."""

    texto_video: str = ""
    nivel: Nivel = Nivel.NEUTRO
    status: str | None = None
    sinal_confirmado: str | None = None


#: Prefixo da chave de um pré-sinal na lista de sinais (ex.: ``"pre:S"``).
PREFIXO_PRE = "pre:"


@dataclass(frozen=True)
class InfoSinal:
    """Uma linha da lista de sinais aprendidos.

    Attributes:
        palavra: Nome do sinal ou do pré-sinal.
        exemplos: Quantidade de quadros gravados.
        treinado: Se a IA atual já conhece este sinal.
        pre_sinal: Para um sinal, o pré-sinal que vem antes dele ("" se nenhum).
        eh_pre_sinal: ``True`` se a linha é um pré-sinal.
    """

    palavra: str
    exemplos: int
    treinado: bool
    pre_sinal: str = ""
    eh_pre_sinal: bool = False

    @property
    def chave(self) -> str:
        """Identifica a linha para ver ou apagar (pré-sinais têm ``PREFIXO_PRE``)."""
        return f"{PREFIXO_PRE}{self.palavra}" if self.eh_pre_sinal else self.palavra


def _separar_chave(chave: str) -> tuple[str, str]:
    """``"pre:S"`` -> ``("S", TIPO_PRE)``; ``"carro"`` -> ``("carro", TIPO_SINAL)``."""
    if chave.startswith(PREFIXO_PRE):
        return chave[len(PREFIXO_PRE) :], TIPO_PRE
    return chave, TIPO_SINAL


class ControladorSinais:
    """Coordena ensino, treinamento e reconhecimento."""

    def __init__(
        self,
        repositorio: RepositorioSinais | None = None,
        caminho_modelo: Path = config.ARQUIVO_MODELO,
        carregador_modelo: Callable[[Path], Any] = carregar_modelo,
        relogio: Callable[[], float] = time.monotonic,
    ) -> None:
        self.repositorio = repositorio or RepositorioSinais()
        self.caminho_modelo = Path(caminho_modelo)
        self.caminho_modelo_pre = caminho_modelo_pre(self.caminho_modelo)
        self._carregador_modelo = carregador_modelo
        self._relogio = relogio
        self.estabilizador = Estabilizador()
        self.buffer = BufferQuadros()
        self.modo = Modo.PARADO
        self.rotulo_atual = ""
        self.tipo_atual = TIPO_SINAL
        self.pre_sinal_atual = ""
        self.total_gravado = 0
        self._rodada = 0
        # Quadros do ensino ainda na memória: (número, vetor, momento).
        self._pendentes: list[tuple[int, np.ndarray, float]] = []
        # Quadros sem mão depois da última mão; só são gravados se a mão voltar.
        self._lacuna: list[tuple[np.ndarray, float]] = []
        self.filtro_quadros = FiltroQuadros()
        self.qualidade = QualidadeGravacao()
        self._ultima_resposta = ResultadoQuadro()
        self._ultimo_voto: str | None = None
        self._modelo: Any = None
        # Pré-sinal no reconhecimento: IA, votação e o pré-sinal visto agora.
        self._pre: ModeloPreSinais | None = None
        self.estabilizador_pre = Estabilizador(
            janela=config.JANELA_PRE_SINAL,
            votos_minimos=config.VOTOS_PRE_SINAL,
        )
        # Pré-sinais vistos há pouco: {nome: até quando ajudam (segundos)}.
        self._pre_recentes: dict[str, float] = {}
        self.espera_sinal_s = float(config.SEGUNDOS_ESPERA_SINAL_PADRAO)
        # Diagnóstico: um CSV por sessão de reconhecimento (veja diagnostico.py).
        self.diagnostico: DiagnosticoReconhecimento | None = None
        self._diag: dict = {}
        self._ultimo_diag: dict = {}

    # -- ensino ----------------------------------------------------------------

    def iniciar_ensino(
        self, palavra: str, tipo: str = TIPO_SINAL, pre_sinal: str = ""
    ) -> str:
        """Começa a gravar exemplos de um sinal ou de um pré-sinal.

        Args:
            palavra: Nome do sinal (ou do pré-sinal, se ``tipo`` for ``TIPO_PRE``).
            tipo: ``TIPO_SINAL`` ou ``TIPO_PRE``.
            pre_sinal: Para um sinal, o pré-sinal que vem antes dele.
        """
        # Sinais em minúsculas; pré-sinais em maiúsculas ("s" e "S" são o mesmo).
        palavra = (
            nome_pre_sinal(palavra) if tipo == TIPO_PRE else palavra.strip().lower()
        )
        pre_sinal = nome_pre_sinal(pre_sinal) if tipo == TIPO_SINAL else ""

        if not palavra:
            raise PalavraInvalidaError(
                "Escreva o pré-sinal antes de começar."
                if tipo == TIPO_PRE
                else "Escreva a palavra do sinal antes de começar."
            )

        if self.modo is Modo.RECONHECENDO:
            self.parar()  # fecha o diagnóstico do reconhecimento
        self.repositorio.garantir_arquivo()
        self.rotulo_atual = palavra
        self.tipo_atual = tipo
        self.pre_sinal_atual = pre_sinal
        self.total_gravado = 0
        self._rodada = novo_id_rodada()
        self._pendentes = []
        self._lacuna = []
        self.filtro_quadros = FiltroQuadros()  # lê o modo atual do config
        self.qualidade = QualidadeGravacao(tipo, palavra)
        self.modo = Modo.ENSINANDO

        if tipo == TIPO_PRE:
            logger.info("Ensino iniciado: pré-sinal %s", palavra)
            return (
                f"Pré-sinal “{palavra}”: mostre a mão nessa forma, girando para "
                "todos os lados (frente, trás, lados)."
            )
        logger.info(
            "Ensino iniciado: %s%s",
            palavra,
            f" (pré-sinal {pre_sinal})" * bool(pre_sinal),
        )
        return f"Ensinando “{palavra}”. Faça o sinal diante da câmera."

    def parar(self) -> str:
        """Para o que estiver em andamento."""
        anterior, self.modo = self.modo, Modo.PARADO

        if anterior is Modo.ENSINANDO:
            self._lacuna = []  # o fim sem mão não é gravado
            self._gravar_pendentes()
            qualidade = self.qualidade.resumo(self.total_gravado)
            logger.info(
                "Ensino parado: %d linhas gravadas. %s",
                self.total_gravado,
                qualidade,
            )
            nome = (
                f"Pré-sinal “{self.rotulo_atual}”"
                if self.tipo_atual == TIPO_PRE
                else f"“{self.rotulo_atual}”"
            )
            return f"{nome}: {self.total_gravado} quadros gravados; {qualidade}"

        if anterior is Modo.RECONHECENDO:
            logger.info("Reconhecimento parado.")
            if self.diagnostico is not None:
                logger.info("%s", self.diagnostico.fechar())
                self.diagnostico = None
            return "Reconhecimento parado."

        return "Programa parado."

    # -- treinamento -----------------------------------------------------------

    def treinar(
        self,
        progresso: Callable[[str], None] = lambda _m: None,
    ) -> treinamento.ResultadoTreino:
        """Treina a IA com o CSV atual."""
        if not self.repositorio.existe():
            raise SemDadosError("Ensine pelo menos um sinal antes de treinar a IA.")

        dados = self.repositorio.carregar()
        resultado = treinamento.treinar(
            dados,
            self.caminho_modelo,
            progresso,
        )
        progresso("Treinando os pré-sinais...")
        treinar_pre_sinais(
            self.repositorio.carregar(TIPO_PRE),
            dados,
            self.repositorio.pre_sinais_dos_sinais(),
            self.caminho_modelo_pre,
        )
        self._modelo = None
        return resultado

    # -- reconhecimento --------------------------------------------------------

    def iniciar_reconhecimento(self, espera_sinal_s: float | None = None) -> str:
        """Carrega o modelo e começa a reconhecer.

        Args:
            espera_sinal_s: Depois de ver um pré-sinal, por quantos segundos
                esperar o sinal. Padrão: ``config.SEGUNDOS_ESPERA_SINAL_PADRAO``.
        """
        if not self.repositorio.existe():
            raise SemDadosError("Ensine pelo menos um sinal primeiro.")

        if not self.caminho_modelo.exists():
            raise ModeloAusenteError("Ensine sinais e clique em TREINAR IA primeiro.")

        if not modelo_esta_atualizado(
            self.caminho_modelo,
            self.repositorio.caminho,
        ):
            raise ModeloDesatualizadoError(
                "Os sinais mudaram.\n\nClique em TREINAR IA antes de reconhecer."
            )

        self._modelo = self._carregador_modelo(self.caminho_modelo)
        self._pre = carregar_pre_sinais(self.caminho_modelo_pre)
        self.espera_sinal_s = float(
            config.SEGUNDOS_ESPERA_SINAL_PADRAO
            if espera_sinal_s is None
            else espera_sinal_s
        )
        self.estabilizador_pre.reiniciar()
        self._pre_recentes = {}
        self.estabilizador.reiniciar()
        self.buffer.limpar()
        self.filtro_quadros = FiltroQuadros()  # lê o modo atual do config
        self._ultima_resposta = ResultadoQuadro("PREPARANDO...")
        self._ultimo_voto = None
        self._ultimo_diag = {}
        if config.DIAGNOSTICO_ATIVO and self.diagnostico is None:
            try:
                self.diagnostico = DiagnosticoReconhecimento(relogio=self._relogio)
            except OSError:
                logger.exception("Não foi possível criar o arquivo de diagnóstico.")
        self.modo = Modo.RECONHECENDO

        logger.info(
            "Reconhecimento iniciado. Sinais do modelo: %s. Pré-sinais: %s",
            list(self._modelo.classes_),
            self._pre.sinal_para_pre if self._pre else "nenhum",
        )
        return "Reconhecendo sinais ao vivo..."

    def limpar_reconhecimento(self) -> None:
        """Esquece o histórico de votos."""
        self.estabilizador.reiniciar()

    # -- quadro a quadro -------------------------------------------------------

    def processar_quadro(
        self,
        vetor: np.ndarray | None,
    ) -> ResultadoQuadro:
        """Processa um quadro de acordo com o modo atual.

        Args:
            vetor: Vetor de 223 valores do quadro
                (:func:`kara_sygna.visao.quadro.montar_vetor_quadro`), mesmo
                sem mãos. ``None`` só se o quadro não pôde ser analisado.
        """
        if self.modo is Modo.ENSINANDO:
            return self._quadro_de_ensino(vetor)

        if self.modo is Modo.RECONHECENDO:
            return self._quadro_de_reconhecimento(vetor)

        return ResultadoQuadro()

    def _quadro_de_ensino(
        self,
        vetor: np.ndarray | None,
    ) -> ResultadoQuadro:
        """Grava os quadros em sequência: a IA aprende o movimento.

        Para o CSV não crescer à toa, só são gravados os quadros escolhidos
        pelo :class:`FiltroQuadros` (``config.MODO_QUADROS``), e os quadros sem
        mão do começo e do fim da gravação ficam de fora.
        """
        if vetor is None:
            return ResultadoQuadro()

        self.qualidade.adicionar(vetor)
        if self.filtro_quadros.aceitar(vetor):
            vetor = np.asarray(vetor, dtype=np.float64)
            agora = self._relogio()
            if not tem_mao(vetor):
                if self.total_gravado:  # antes da primeira mão, nada é guardado
                    self._lacuna.append((vetor, agora))
            else:
                for quadro, momento in [*self._lacuna, (vetor, agora)]:
                    self._pendentes.append((self.total_gravado, quadro, momento))
                    self.total_gravado += 1
                self._lacuna = []
                if len(self._pendentes) >= config.QUADROS_POR_LOTE_GRAVACAO:
                    self._gravar_pendentes()

        dica = "" if tem_mao(vetor) else " Mostre as mãos para a câmera."
        return ResultadoQuadro(
            texto_video=f"{self._titulo_ensino()}: {self.rotulo_atual} ({self.total_gravado})",
            nivel=Nivel.ENSINO,
            status=(
                f"Ensinando “{self.rotulo_atual}”: {self.total_gravado} quadros.{dica}"
            ),
        )

    def _titulo_ensino(self) -> str:
        """Texto sobre a câmera (sem acentos: o OpenCV não os desenha)."""
        if self.tipo_atual == TIPO_PRE:
            return "PRE-SINAL"
        return "SINAL" if self.pre_sinal_atual else "ENSINANDO"

    def _gravar_pendentes(self) -> None:
        """Preenche os buracos curtos das mãos e grava os quadros no CSV."""
        pendentes, self._pendentes = self._pendentes, []
        if not pendentes:
            return
        numeros = [numero for numero, _, _ in pendentes]
        quadros, preenchidos = preencher_buracos(
            np.array([quadro for _, quadro, _ in pendentes]),
            np.array([momento for _, _, momento in pendentes]),
        )
        self.qualidade.adicionar_salvos(quadros)
        if preenchidos:
            logger.info("Mãos preenchidas em %d quadros (buracos curtos).", preenchidos)
        self.repositorio.anexar_quadros(
            self.rotulo_atual,
            self._rodada,
            zip(numeros, quadros, strict=True),
            tipo=self.tipo_atual,
            pre_sinal=self.pre_sinal_atual,
        )

    def _quadro_de_reconhecimento(
        self,
        vetor: np.ndarray | None,
    ) -> ResultadoQuadro:
        """Junta o quadro aos anteriores, prevê o sinal e atualiza a votação.

        Só os quadros escolhidos pelo :class:`FiltroQuadros` (o mesmo modo do
        ensino) entram na janela e geram uma nova previsão. Nos outros, a
        última previsão continua valendo: cada quadro da câmera conta um voto,
        então um sinal parado é confirmado tão rápido quanto um em movimento.
        """
        if vetor is None:
            self.estabilizador.atualizar(None)
            self.estabilizador_pre.atualizar(None)
            return ResultadoQuadro("MOSTRE UM SINAL")

        presentes = maos_presentes(vetor)[0]
        self._diag = {
            "mao_esq": int(presentes[0]),
            "mao_dir": int(presentes[1]),
            "corpo": int(vetor[COLUNA_CORPO_PRESENTE] > 0.5),
            "rosto": int(vetor[COLUNA_ROSTO_PRESENTE] > 0.5),
        }
        self._atualizar_pre_sinal(vetor)
        if self.filtro_quadros.aceitar(vetor):
            self._ultima_resposta, self._ultimo_voto = self._analisar_janela(vetor)
        else:
            self._diag.update(self._ultimo_diag, motivo="repete")

        confirmado = self.estabilizador.atualizar(self._ultimo_voto)
        if confirmado in config.ROTULOS_IGNORADOS:
            confirmado = None
        if confirmado:
            logger.info(
                "Sinal confirmado: %s (%s%%)",
                confirmado,
                self._diag.get("top1_pct", "?"),
            )
        if confirmado and self._pre is not None:
            # O sinal chegou: o pré-sinal dele já ajudou.
            self._pre_recentes.pop(self._pre.sinal_para_pre.get(confirmado, ""), None)
        if self.diagnostico is not None:
            ativo = self.pre_ativo
            restante = (
                round(max(0.0, self._pre_recentes[ativo] - self._relogio()), 2)
                if ativo
                else ""
            )
            self.diagnostico.registrar(
                {
                    **self._diag,
                    "pre_ativo": ativo or "",
                    "pre_restante_s": restante,
                    "voto": self._ultimo_voto or "",
                    "escrito": confirmado or "",
                }
            )
        # O pré-sinal visto não aparece na tela (fica no log e no diagnóstico).
        return replace(self._ultima_resposta, sinal_confirmado=confirmado)

    @property
    def pre_ativo(self) -> str | None:
        """O pré-sinal visto mais recentemente que ainda ajuda (ou ``None``)."""
        agora = self._relogio()
        validos = {n: t for n, t in self._pre_recentes.items() if t >= agora}
        return max(validos, key=validos.get) if validos else None

    def _atualizar_pre_sinal(self, vetor: np.ndarray) -> None:
        """Procura um pré-sinal nas mãos e lembra dele por alguns segundos.

        O pré-sinal é só uma AJUDA, nunca uma trava: por
        ``espera_sinal_s`` segundos, os sinais dele são aceitos com a
        confiança normal; os outros sinais continuam valendo (veja
        :meth:`_limite_do_sinal`). A votação (:class:`Estabilizador`) evita
        que as mãos fechadas do próprio "carro" disparem o "S" de novo.
        """
        if self._pre is None:
            return
        agora = self._relogio()
        for nome in [n for n, t in self._pre_recentes.items() if t < agora]:
            logger.info("Pré-sinal %s expirou sem sinal.", nome)
            del self._pre_recentes[nome]

        nome, confianca = self._pre.classificar(vetor)
        self._diag.update(pre_visto=nome or "", pre_pct=round(100 * confianca))
        visto = self.estabilizador_pre.atualizar(nome)
        if visto and self._pre.sinais_do_pre(visto):
            logger.info("Pré-sinal visto: %s (%.0f%%)", visto, 100 * confianca)
            self._diag["pre_novo"] = visto
            self._pre_recentes[visto] = agora + self.espera_sinal_s

    def _limite_do_sinal(self, sinal: str) -> float:
        """Confiança mínima para aceitar um sinal agora.

        * Sem pré-sinal (ou sem IA de pré-sinais): ``CONFIANCA_MINIMA``.
        * Com pré-sinal visto há pouco: ``CONFIANCA_MINIMA`` (o pré-sinal ajuda).
        * Com pré-sinal não visto: ``CONFIANCA_SEM_PRE_SINAL`` (um pouco mais
          exigente, mas nunca bloqueado).
        """
        if self._pre is None or sinal not in self._pre.sinal_para_pre:
            return config.CONFIANCA_MINIMA
        agora = self._relogio()
        if self._pre_recentes.get(self._pre.sinal_para_pre[sinal], -1.0) >= agora:
            return config.CONFIANCA_MINIMA
        return max(config.CONFIANCA_MINIMA, config.CONFIANCA_SEM_PRE_SINAL)

    def _analisar_janela(self, vetor: np.ndarray) -> tuple[ResultadoQuadro, str | None]:
        """Acrescenta o quadro à janela e prevê o sinal.

        Returns:
            O que mostrar na tela e o voto deste quadro (o sinal aceito, ou
            ``None`` se não há mão, a janela não encheu ou a IA ficou incerta).
        """
        self.buffer.adicionar(vetor)
        self._ultimo_diag = {}
        if not tem_mao(vetor):
            self._diag["motivo"] = "sem_mao"
            return ResultadoQuadro("MOSTRE UM SINAL"), None

        if not self.buffer.cheio:
            self._diag["motivo"] = "preparando"
            return ResultadoQuadro("PREPARANDO..."), None

        predicao = prever(self._modelo, self.buffer.features())
        # Limite de cada sinal: o pré-sinal visto deixa os sinais dele mais
        # fáceis; sem ele, o sinal precisa de um pouco mais de certeza.
        limite = self._limite_do_sinal(predicao.sinal)
        aceito = (
            predicao.confianca >= limite and predicao.margem >= config.MARGEM_MINIMA
        )
        predicao = replace(predicao, aceito=aceito)
        self._registrar_previsao(predicao, limite)

        if not predicao.aceito:
            return ResultadoQuadro(
                f"INCERTO ({predicao.confianca:.0%})", Nivel.INCERTO
            ), None

        return ResultadoQuadro(
            f"SINAL: {predicao.sinal} ({predicao.confianca:.0%})",
            Nivel.OK,
            f"Sinal reconhecido: {predicao.sinal} ({predicao.confianca:.0%})",
        ), predicao.sinal

    def _registrar_previsao(self, predicao, limite: float) -> None:
        """Guarda no diagnóstico os 3 sinais mais prováveis e o motivo."""
        ordem = sorted(predicao.probabilidades.items(), key=lambda kv: -kv[1])[:3]
        campos: dict = {
            "margem_pct": round(100 * predicao.margem),
            "limite_pct": round(100 * limite),
        }
        for posicao, (sinal, prob) in enumerate(ordem, 1):
            campos[f"top{posicao}"] = sinal
            campos[f"top{posicao}_pct"] = round(100 * prob)
        exigente = limite > config.CONFIANCA_MINIMA  # pré-sinal não visto
        if predicao.aceito:
            campos["motivo"] = "ok_sem_pre" if exigente else "ok"
        elif (
            exigente
            and config.CONFIANCA_MINIMA <= predicao.confianca < limite
            and predicao.margem >= config.MARGEM_MINIMA
        ):
            campos["motivo"] = "falta_pre"
        else:
            campos["motivo"] = "incerto"
        self._diag.update(campos)
        self._ultimo_diag = dict(campos)

    # -- lista de sinais -------------------------------------------------------

    def listar_sinais(self) -> list[InfoSinal]:
        """Lista pré-sinais e sinais, exemplos e estado do treinamento."""
        contagem = self.repositorio.contagem_por_sinal()
        contagem_pre = self.repositorio.contagem_por_sinal(TIPO_PRE)

        if not contagem and not contagem_pre:
            return []

        atualizado = modelo_esta_atualizado(
            self.caminho_modelo,
            self.repositorio.caminho,
        )
        conhecidos = classes_do_modelo(self.caminho_modelo) if atualizado else set()
        pre_conhecidos = (
            set(nomes_dos_pre_sinais(self.caminho_modelo_pre)) if atualizado else set()
        )
        pre_dos_sinais = self.repositorio.pre_sinais_dos_sinais()

        return [
            InfoSinal(nome, qtd, nome in pre_conhecidos, eh_pre_sinal=True)
            for nome, qtd in contagem_pre.items()
        ] + [
            InfoSinal(
                palavra, qtd, palavra in conhecidos, pre_dos_sinais.get(palavra, "")
            )
            for palavra, qtd in contagem.items()
        ]

    def apagar_sinal(self, chave: str) -> str:
        """Apaga um sinal (ou um pré-sinal, ``"pre:S"``) e remove a IA treinada."""
        palavra, tipo = _separar_chave(chave)
        if self.modo is not Modo.PARADO:
            self.parar()  # grava o ensino ou fecha o diagnóstico do reconhecimento
        self.repositorio.apagar_sinal(palavra, tipo)
        remover_arquivo_se_existir(self.caminho_modelo)
        remover_arquivo_se_existir(self.caminho_modelo_pre)
        self._modelo = None
        self.modo = Modo.PARADO

        nome = "Pré-sinal" if tipo == TIPO_PRE else "Sinal"
        return f"{nome} “{palavra}” apagado. Treine a IA novamente."

    def criar_reproducao(self, chave: str) -> Reproducao:
        """Prepara a pré-visualização de um sinal (ou pré-sinal) gravado."""
        palavra, tipo = _separar_chave(chave)
        exemplos = self.repositorio.exemplos_do_sinal(palavra, tipo)

        if len(exemplos) == 0:
            raise SemDadosError(f"Não há exemplos gravados para “{palavra}”.")

        return Reproducao(palavra, completar_com_espelho(exemplos))
