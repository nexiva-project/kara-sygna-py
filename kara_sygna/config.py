"""Configurações centrais do Kara Sygna.

Todo número "mágico", caminho ou limiar fica aqui, em um só lugar. Os demais
módulos importam estas constantes em vez de repetir valores espalhados.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
# Ancorados na raiz do projeto: não dependem da pasta de onde o programa foi
# executado (o que antes fazia o CSV "cair" dentro de src/).
RAIZ_PROJETO = Path(__file__).resolve().parent.parent
PASTA_ARMAZENAMENTO = RAIZ_PROJETO / "armazenamento"
ARQUIVO_CSV = PASTA_ARMAZENAMENTO / "dados_sinais.csv"
ARQUIVO_MODELO = PASTA_ARMAZENAMENTO / "modelo_sinais.pkl"

# ---------------------------------------------------------------------------
# Câmera
# ---------------------------------------------------------------------------
# Índices testados em ordem. Em notebooks Linux é comum /dev/video0 ser a
# imagem e /dev/video1 apenas metadados; por isso cada índice é validado lendo
# um quadro de verdade.
INDICES_CAMERA = (0, 1, 2, 3, 4)
# Resolução pedida à câmera. As características das mãos usam coordenadas
# normalizadas pela imagem, então manter a mesma proporção (4:3) em todos os
# computadores evita que o modelo "veja" mãos esticadas ou achatadas.
# Use None para não forçar nenhuma resolução.
LARGURA_CAPTURA = 640
ALTURA_CAPTURA = 480
LEITURAS_DE_TESTE = 10
PAUSA_ENTRE_LEITURAS_S = 0.05

# ---------------------------------------------------------------------------
# MediaPipe (detecção das mãos)
# ---------------------------------------------------------------------------
MAX_MAOS = 2
CONFIANCA_DETECCAO = 0.7
CONFIANCA_RASTREIO = 0.5

# O frame é espelhado antes de ser enviado ao MediaPipe; com isso os rótulos
# "Left"/"Right" saem trocados nesta configuração. Mudar este valor altera o
# significado dos dados já gravados: se mudar, grave os sinais de novo.
INVERTER_LATERALIDADE = True

# Cores (BGR, como o OpenCV espera) usadas para desenhar cada mão.
CORES_MAOS_BGR = {"esq": (70, 190, 255), "dir": (150, 105, 255)}

# ---------------------------------------------------------------------------
# Reconhecimento
# ---------------------------------------------------------------------------
CONFIANCA_MINIMA = 0.70  # probabilidade mínima do melhor sinal
MARGEM_MINIMA = 0.15  # melhor - segundo melhor (evita empates)
JANELA_ESTABILIZADOR = 10  # quadros considerados na votação
VOTOS_MINIMOS = 6  # votos necessários para confirmar um sinal
QUADROS_PARA_LIBERAR = 8  # quadros sem o sinal para poder repeti-lo
# Rótulos aprendidos pela IA que NÃO viram texto (ex.: mão relaxada).
ROTULOS_IGNORADOS = frozenset({"neutro", "nada"})
# Testa várias inclinações da mão a cada quadro e fica com a mais confiante.
CORRECAO_ROTACAO_ATIVA = True

# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------
SEMENTE_ALEATORIA = 42
ARVORES = 300
MIN_AMOSTRAS_FOLHA = 2
MAX_DOBRAS_VALIDACAO = 5
MIN_VARIANTES_AUMENTO = 2
MAX_VARIANTES_AUMENTO = 8
ORCAMENTO_LINHAS_AUMENTO = 40_000
# Quando o CSV não tem a coluna "rodada", cada sinal é dividido em, no
# mínimo, este número de blocos contíguos para permitir a validação.
RODADAS_MINIMAS_INFERIDAS = 5

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
INTERVALO_QUADRO_MS = 15
LARGURA_AREA_CAMERA = 900
ALTURA_AREA_CAMERA = 510
