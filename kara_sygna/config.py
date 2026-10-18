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
# Um CSV por sessão de reconhecimento, com o que a IA viu e decidiu em cada
# quadro (para estudar erros). DIAGNOSTICO_ATIVO = False desliga.
PASTA_DIAGNOSTICO = PASTA_ARMAZENAMENTO / "diagnostico"
DIAGNOSTICO_ATIVO = True

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
# 0.5: com 0.7, mãos borradas pelo movimento eram descartadas (no teste com
# o sinal "carro", 42% dos quadros ficavam sem nenhuma mão; com 0.5, 24%).
CONFIANCA_DETECCAO = 0.5
# 0.3: no vídeo do sinal "carro", foi o valor que mais manteve as duas mãos.
CONFIANCA_RASTREIO = 0.3
# O detector de mãos recebe uma cópia da imagem com a pele clara e o resto
# escuro (a cor da pele é medida no rosto), para as mãos se destacarem da
# roupa. Só o detector vê essa cópia; a tela mostra a imagem normal.
REALCE_PELE = True
# Quanto sobra do que não é pele (0 = preto, 1 = sem realce).
REALCE_PELE_ESCURO = 0.1
# Se uma mão some por até estes segundos (quadro borrado), continua valendo a
# última posição dela, no desenho e na IA. 0 = desliga.
MEMORIA_MAO_S = 0.3

# MediaPipe Pose (braços): 0 = modelo leve (~13 ms por quadro), 1 = completo.
COMPLEXIDADE_POSE = 0
# Conferência das mãos pelos braços (tira as "mãos fantasmas"), em larguras
# de ombro. Só é descartada a mão mais longe que isto dos dois pulsos do Pose
# (o Pose erra o pulso com as mãos embaixo; com 0,6 ele descartava a mão
# esquerda de verdade no sinal "carro").
DISTANCIA_MAO_PULSO = 1.5
# Duas mãos mais perto que isto uma da outra são a mesma mão (fantasma).
DISTANCIA_MAOS_DUPLICADAS = 0.35

# O frame é espelhado (como um espelho) antes de ir para o MediaPipe, e o
# MediaPipe já espera a imagem espelhada: o rótulo "Right" é a mão direita de
# verdade. False = sem inverter (a mão direita é "dir"). Até a versão com
# True, a mão direita aparecia como "esq". Mudar este valor troca o
# significado dos dados gravados: grave os sinais de novo.
INVERTER_LATERALIDADE = False

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

# ---------------------------------------------------------------------------
# Pré-sinal (forma de mão, como o "S", que vem antes de alguns sinais)
# ---------------------------------------------------------------------------
# Probabilidade mínima para a IA dizer que uma mão está num pré-sinal. 0.50:
# num teste ao vivo, o "S" mostrado ficou entre 33% e 60% e nunca chegava a 70%.
CONFIANCA_PRE_SINAL = 0.50
# Um sinal que tem pré-sinal também é escrito SEM o pré-sinal ter sido visto,
# se a IA tiver pelo menos esta certeza. O pré-sinal vira uma ajuda (quando
# aparece, só os sinais dele valem), não uma trava. 1.0 = exige o pré-sinal.
# 0.78: num teste ao vivo, os quadros certos que esperavam o pré-sinal tinham
# 78% em média (com 0.85 eles não eram escritos).
CONFIANCA_SEM_PRE_SINAL = 0.78
# Votação do pré-sinal: VOTOS_PRE_SINAL entre os últimos JANELA_PRE_SINAL
# quadros (menos que nos sinais: a forma da mão é fácil de ver e precisa sair
# logo, para o sinal não atrasar).
JANELA_PRE_SINAL = 5
VOTOS_PRE_SINAL = 3
# Árvores da IA de pré-sinais (é menor que a dos sinais: só olha uma mão).
ARVORES_PRE_SINAL = 100
# No ensino com várias etapas e o campo de segundos vazio, cada etapa dura
# isto (para a sequência andar sozinha).
SEGUNDOS_PRE_SINAL_PADRAO = 8
# Um Enter grava o sinal uma vez em cada ritmo (cada uma vira uma rodada):
# "carro 1/3: devagar", "carro 2/3: médio", "carro 3/3: rápido".
# () = uma vez só, como antes.
RITMOS_ENSINO = ("devagar", "médio", "rápido")
# No reconhecimento, por quantos segundos o pré-sinal visto ajuda os sinais
# dele (eles são aceitos com a confiança normal).
SEGUNDOS_ESPERA_SINAL_PADRAO = 3

# ---------------------------------------------------------------------------
# Quais quadros da câmera são usados (ensino E reconhecimento)
# ---------------------------------------------------------------------------
# "mudanca": só usa um quadro quando a mão muda (forma ou lugar) ou quando uma
#            mão aparece/some. Mão parada não enche o CSV. (Filtro original.)
# "fixo":    usa 1 a cada USAR_1_A_CADA_QUADROS quadros, mude ou não.
# Se trocar o modo, grave os sinais de novo e treine a IA.
MODO_QUADROS = "mudanca"

# Modo "mudanca":
# Mudança mínima na forma das mãos (média da diferença dos 126 valores).
LIMIAR_MUDANCA_FORMA = 0.015
# Mudança mínima no lugar das mãos, em larguras de rosto (0.05 = 5% do rosto).
LIMIAR_MUDANCA_POSICAO = 0.05
# Com a mão parada, ainda usa 1 quadro a cada tantos segundos, para a IA
# aprender o sinal parado. 0 = nunca (exatamente como o filtro original).
PARADO_USAR_A_CADA_S = 0.5

# Modo "fixo":
USAR_1_A_CADA_QUADROS = 2

# False: a IA NÃO usa o lugar exato das mãos (distância dos dedos até o rosto,
# posição das mãos em relação ao nariz e dos braços em relação aos ombros);
# olha a forma das mãos e o movimento. Num teste com o sinal "casa" feito em
# 7 lugares, com o lugar ela só reconhecia no lugar de sempre; sem o lugar,
# reconheceu em 6 dos 7, com a mesma acurácia (99%). Ligue (True) para sinais
# que só mudam pelo lugar (ex.: mesma mão na testa ou no peito). Ao mudar,
# treine a IA de novo (não precisa regravar).
USAR_LUGAR = False

# Quantos quadros usados a IA olha para reconhecer um sinal (inclusive o
# movimento).
QUADROS_POR_JANELA = 10

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
# No treino, cada gravação é cortada em janelas que começam de N em N quadros.
PASSO_JANELA_TREINO = 2
# Velocidades extras de cada gravação no treino (0.75 = mais devagar, 1.33 =
# mais rápido), para o sinal valer feito em ritmos diferentes. () = desliga.
VELOCIDADES_TREINO = (0.75, 1.33)
# Durante o ensino, os quadros ficam na memória e vão para o disco ao parar
# (ou a cada tantos quadros, em gravações longas). Antes de gravar, as mãos
# que sumiram por pouco tempo são preenchidas entre antes e depois.
QUADROS_POR_LOTE_GRAVACAO = 900
# Maior buraco de uma mão preenchido na gravação, em segundos (0 = nenhum).
PREENCHER_BURACO_S = 0.5
# Ao começar a ensinar, espera estes segundos antes de gravar (tempo para se
# posicionar e já começar o sinal).
SEGUNDOS_CONTAGEM_ENSINO = 3

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
INTERVALO_QUADRO_MS = 15
# Tamanho da janela principal ao abrir (menor, se a tela for menor).
LARGURA_JANELA_PADRAO = 1024
ALTURA_JANELA_PADRAO = 768
LARGURA_AREA_CAMERA = 900
ALTURA_AREA_CAMERA = 510
