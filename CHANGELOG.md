# Changelog

## [0.6.0] - 2026-10-18

### Adicionado
- Um Enter grava o sinal 3 vezes seguidas, em ritmos diferentes (`config.RITMOS_ENSINO`): "carro 1/3: devagar", "carro 2/3: médio", "carro 3/3: rápido", cada uma como uma rodada (o pré-sinal, se houver, é gravado uma vez antes). O `neutro` grava 1/3, 2/3, 3/3. Enter ou PARAR cancelam o resto; o que já foi gravado fica salvo.
- Braços (MediaPipe Pose, modelo leve, rodando em paralelo): ombros, cotovelos e pulsos entram no vetor de cada quadro (9 colunas `b_…`), e o MOVIMENTO dos sinais passa a ser medido pelos pulsos dos braços. O braço continua visível quando a mão borra: no vídeo do sinal "carro", o Pose achou os dois pulsos em 100% dos quadros, contra ~20% das duas mãos.
- Na gravação, a mão que some por até `config.PREENCHER_BURACO_S` (0,5 s) é preenchida entre antes e depois, com curva suave (PCHIP). No vídeo do "carro": quadros com as duas mãos de 20% para 85%, sem nenhuma mão de 32% para 1%. Ao vivo continua a memória (mão congelada por um instante), para não atrasar o reconhecimento.
- No treino, cada gravação também entra mais devagar e mais rápida (`config.VELOCIDADES_TREINO`), para o sinal valer em ritmos diferentes.
- Pré-sinal: botão **Pré-sinal** com um campo (ex.: `S`). No ensino, a gravação tem duas etapas automáticas: primeiro o pré-sinal (uma mão na forma, girando para todos os lados), depois o sinal. Uma IA própria aprende a forma de cada pré-sinal (vale para qualquer mão). No reconhecimento, o pré-sinal é uma ajuda (veja abaixo). O CSV ganhou as colunas `tipo` e `pre_sinal` (o anterior vai para o backup).
- Realce de pele (`config.REALCE_PELE`): o detector de mãos recebe uma cópia da imagem com a pele clara e o resto escuro, com a cor da pele medida no rosto. As mãos se destacam da roupa: no vídeo do sinal "carro", os quadros sem nenhuma mão caíram de 56% para 32%. A tela continua mostrando a imagem normal.
- Medidor de qualidade da gravação: ao parar o ensino, a barra de status mostra em quantos quadros apareceram as mãos, as duas mãos e o rosto, e avisa quando vale gravar de novo (mãos fora da câmera, segunda mão sumindo, rosto ausente, gravação curta).
- Memória da mão: se uma mão some por até `config.MEMORIA_MAO_S` (0,3 s), a última posição dela continua valendo, no desenho e na IA. Tapa os quadros borrados no meio de sinais com movimento.
- Contador `MAOS: n` sobre a imagem da câmera (verde com 2 mãos; avisa quando uma mão veio da memória).
- A IA passa a olhar o rosto: distância de cada ponta de dedo até testa, olhos, nariz, bochechas, boca e queixo, a expressão facial e a posição das mãos em relação ao nariz. Em Libras, o lugar do sinal faz parte dele.
- Enter (ou COMEÇAR A ENSINAR) faz uma contagem de 3 s (`config.SEGUNDOS_CONTAGEM_ENSINO`) e só então começa a gravar. No campo **TEMPO DE GRAVAÇÃO**, embaixo da câmera, escolhe-se por quantos segundos gravar; no fim, a gravação para sozinha. Vazio: grava até parar. Enter na contagem ou na gravação para na hora.
- Sinais com movimento: a IA olha os últimos `QUADROS_POR_JANELA` quadros (de onde o pulso veio, quanto andou, quanto a mão mudou), não só um quadro.

### Alterado
- A janela principal abre em 1024x768 (`config.LARGURA_JANELA_PADRAO` / `ALTURA_JANELA_PADRAO`; menor se a tela for menor). O painel de controle agora rola (barra e roda do mouse) quando não cabe na altura, em vez de esconder os botões de baixo.
- A IA não usa mais o lugar exato das mãos (`config.USAR_LUGAR = False`): distância dos dedos até o rosto, posição das mãos e dos braços ficam fora da janela; ela olha a forma e o movimento. Com o lugar, o sinal "casa" só era reconhecido no lugar de sempre; sem ele, foi escrito em 6 de 7 lugares testados (esquerda, direita, alto, baixo, longe), sem nenhum sinal errado. Os dados continuam no CSV: basta treinar de novo; religue para sinais que só mudam pelo lugar.
- Pré-sinal é só uma ajuda, nunca uma trava: qualquer sinal pode ser reconhecido a qualquer momento (antes, depois do "S", a casa ficava bloqueada por alguns segundos). Um sinal com pré-sinal precisa de 70% de certeza se o pré-sinal dele foi visto há pouco, e de 78% se não foi (`CONFIANCA_SEM_PRE_SINAL`). A exigência para ver o pré-sinal caiu de 70% para 50% (`CONFIANCA_PRE_SINAL`). O pré-sinal não aparece mais na tela (só no log e no diagnóstico: motivos `falta_pre` e `ok_sem_pre`).
- A IA pode ser treinada com um sinal só (antes exigia 2). Com um sinal e sem `neutro`, toda mão na câmera é reconhecida como esse sinal.
- Rastreio das mãos com 30% de confiança (antes 50%), o valor que mais manteve as duas mãos no vídeo do "carro".
- A forma das mãos agora vem dos *world landmarks* do MediaPipe (pontos 3D em metros), com profundidade melhor que a da imagem. Num teste com 5 formas de mão girando o pulso, acertaram 58% contra 54%. As colunas ganharam o prefixo `w_`: o CSV anterior vai para o backup e os sinais precisam ser gravados de novo.
- No Windows, a câmera abre primeiro pelo MediaFoundation (30 quadros por segundo; o DirectShow entregava ~16). Abrir demora ~1 s a mais.
- A detecção das mãos exige 50% de certeza (antes 70%): mãos borradas pelo movimento eram descartadas.
- O ensino grava os quadros em sequência, em lotes, e o CSV ganhou a coluna `quadro`. Quais quadros entram é escolhido em `config.MODO_QUADROS`: `"mudanca"` (padrão; o filtro original, que só grava quando a forma ou o lugar da mão muda, mais 1 quadro a cada `PARADO_USAR_A_CADA_S` com a mão parada) ou `"fixo"` (1 a cada `USAR_1_A_CADA_QUADROS`). O reconhecimento usa o mesmo modo. O começo e o fim da gravação sem mão não são gravados. Dados do formato antigo vão para um backup e precisam ser gravados de novo.
- O modelo é salvo comprimido (cerca de 3 vezes menor) e o CSV usa 4 casas decimais.
- O espelhamento (sinal da mão direita valer para a esquerda) é feito no treino, na gravação inteira, em vez de duplicar linhas no CSV.
- O reconhecimento avalia cada janela uma vez. As 31 rotações testadas a cada quadro foram removidas: escolher a rotação mais confiante aumentava os falsos positivos. As inclinações continuam sendo aprendidas pelo aumento de dados no treino.

### Corrigido
- No Ubuntu a janela abria em branco e só aparecia depois de minimizar. No Linux a tela de carregamento deixa de usar `overrideredirect` (o XWayland não desenha essa janela) e só é mostrada com o conteúdo pronto; em seguida ela é remapeada uma vez, o mesmo efeito de minimizar e restaurar.
- No Linux a janela principal não abria ("bad cursor spec \"size_nw_se\""): esse cursor da alça de redimensionar só existe no Windows. Fora do Windows agora usa "bottom_right_corner".
- A conferência pelos braços descartava a mão esquerda de verdade no sinal "carro" (o Pose erra o pulso com as mãos embaixo). Agora só descarta mão muito longe dos dois braços (`DISTANCIA_MAO_PULSO = 1.5`) ou duplicada (`DISTANCIA_MAOS_DUPLICADAS`), e o lado da mão vem sempre do braço (nos vídeos, quando o MediaPipe e o braço discordavam, o braço acertou 82 de 83). No vídeo do carro: 0 mãos reais descartadas (antes 16).
- O app não abria se o download do modelo Pose falhasse (sem internet ou sem permissão): agora segue sem os braços e avisa no log.
- Apagar um sinal (ou começar a ensinar) durante o reconhecimento deixava o arquivo de diagnóstico aberto; duas sessões no mesmo segundo sobrescreviam o mesmo arquivo.
- Regravar um sinal sem pré-sinal não tirava o pré-sinal antigo dele.
- O medidor de qualidade avisava "segunda mão sumiu" no pré-sinal (uma mão) e "mãos fora" no `neutro`; agora conta as duas mãos depois do preenchimento e não reclama desses casos.
- O diagnóstico marcava `falta_pre` quando o problema era a dúvida entre dois sinais (margem pequena).
- O campo de segundos da gravação também mudava a ajuda do pré-sinal no reconhecimento; agora a ajuda usa `SEGUNDOS_ESPERA_SINAL_PADRAO`.
- A lista de sinais recarregava os modelos do disco a cada atualização.
- O pré-sinal "s" e o "S" viravam dois pré-sinais diferentes (os exemplos se dividiam). Agora o nome do pré-sinal não diferencia maiúscula de minúscula, e os já gravados se juntam sozinhos.
- "Mão fantasma": aparecia uma segunda mão perto da mão de verdade (no `neutro` só com a esquerda, uma "direita" em 27 de 69 quadros). Agora cada mão é conferida pelos braços do Pose: só vale a mão perto do pulso de um braço (`config.DISTANCIA_MAO_PULSO`), cada braço fica com uma mão e o lado da mão é o lado do braço. No vídeo do "carro", com as duas mãos reais, só 8 de 396 mãos foram descartadas.
- A mão direita aparecia como "esquerda" (desenho, dados e diagnóstico): `INVERTER_LATERALIDADE` agora é `False`, porque o MediaPipe já espera a imagem espelhada. O CSV ganhou a coluna `formato_v2` e o anterior vai para o backup sozinho (grave os sinais de novo).
- As duas mãos eram desenhadas com a cor da "direita"; agora cada uma usa a sua cor, com o mesmo lado que a IA usa.
- A tela de carregamento abria escondida atrás de outro programa quando o app era iniciado por um terminal dentro dele (VS Code, Claude); agora ela e a janela principal abrem na frente.
- A câmera que entregava 16:9 achatava as mãos para a IA; os quadros agora são recortados para 4:3.
- Duas rodadas iniciadas no mesmo segundo recebiam o mesmo identificador.

### Dependências
- `scipy` (preenchimento suave da mão que some no meio da gravação).

## [0.5.0] - 2026-10-11

### Corrigido
- Webcam no Linux/Ubuntu: o código abria a câmera sempre com `cv2.CAP_DSHOW`, que só existe no Windows. Agora cada sistema usa os seus backends (Windows: DirectShow/MediaFoundation; Linux: V4L2; macOS: AVFoundation), com o automático do OpenCV como último recurso.
- Em notebooks Linux, `/dev/video1` costuma ser só metadados. Cada câmera só é aceita se entregar um quadro de verdade (com algumas tentativas de aquecimento).
- Digitar a letra "q" no campo da palavra fechava o programa. O atalho foi removido.
- Um sinal repetido ("um um") não podia ser escrito duas vezes; agora basta tirar a mão (ou trocar de sinal) entre as repetições.
- O CSV antigo (sem a coluna `rodada`) agora é convertido automaticamente, com backup, em vez de ser movido para o lado.
- A pré-visualização do sinal abria uma janela do OpenCV (instável junto com o Tk no Linux); agora é desenhada dentro de uma janela do Tk.
- O treinamento rodava na thread da interface e congelava a câmera; agora roda em segundo plano.
- Mensagens de erro da câmera agora listam cada tentativa e dão dicas (`ls /dev/video*`, grupo `video`).

### Refatorado
- O `interface_sinais.py` (≈1600 linhas, uma classe fazendo tudo) foi dividido em pacotes por responsabilidade: `camera`, `visao`, `dados`, `ia`, `aplicacao`, `interface` e `cli`.
- Regras do programa (`aplicacao/controlador.py`) separadas da interface Tk, permitindo testes sem câmera nem tela.
- Caminhos ancorados na raiz do projeto; CSV e modelo ficam em `armazenamento/`.
- Log feito com o módulo `logging` do Python (mesmo formato no terminal e na janela de log) em vez de um módulo global.
- Constantes e limiares reunidos em `config.py`.
- Todas as funções, classes e métodos têm docstring (estilo Google, em português).

### Adicionado
- Suíte de testes automatizados (`pytest`) e testes manuais de câmera em `testes/manuais/`.
- Validação do modelo por rodada de gravação, com aumento de dados só no treino.
- Votação em janela de quadros para confirmar sinais, com margem mínima entre o 1º e o 2º mais provável.
- Botão "ABRIR LOG" (a janela de log começa oculta).
- Scripts de terminal: `python -m kara_sygna.cli.coletar_dados | treinar_modelo | reconhecer_sinais`.

### Removido
- Código morto da mão 3D ao vivo e atalhos de teclado que não faziam nada.
- Scripts antigos com imports quebrados (`dados.py`, `aumentar_dados.py` etc. foram reescritos dentro dos pacotes).

### Dependências
- `opencv-python` substituído por `opencv-contrib-python` (não instale os dois juntos).
- Adicionado `pillow`; `numpy<2` fixado.

## [0.4.1] - 2026-10-02

### Corrigido
- Bug no Linux/Ubuntu em que `cv2.imshow` abria múltiplas janelas de câmera, causado por conflito entre os backends gráficos do OpenCV (GTK) e do Tkinter.

### Refatorado
- Vídeo da câmera agora é renderizado dentro de um widget `Label` do Tkinter, embutido na própria janela do painel, em vez de uma janela separada do OpenCV.

### Removido
- Uso de `cv2.imshow` e `cv2.waitKey` para exibir a câmera (substituídos pela renderização via Tkinter).

### Dependências
- Adicionado `pillow` ao `requirements.txt` (necessário para converter frames da câmera em imagens exibíveis pelo Tkinter).

## [0.4.0] - 2026-09-18

### Adicionado
- Painel de controle único para a aplicação.
- Modo "Ensinar" para coletar novos sinais diretamente pelo painel.
- Treinamento da IA diretamente pelo painel.
- Modo "Reconhecimento" para identificar sinais em tempo real.
- Lista de sinais aprendidos.
- Controle para iniciar e interromper o ensino e o reconhecimento.

### Refatorado
- Centralização do fluxo da aplicação no `app.py`.
- Integração entre coleta de dados, treinamento e reconhecimento.
- Atualização da câmera pelo loop do Tkinter, sem utilização de threads.
- Remove o módulo `pipeline.py`

### Interface
- Adicionados controles para:
  - Começar a ensinar.
  - Parar.
  - Treinar IA.
  - Iniciar reconhecimento.

## [0.3.0] - 2026-09-18

### Adicionado
- Painel de controle único desenvolvido com Tkinter.
- Botões para executar as principais funcionalidades do projeto.
- A câmera agora é iniciada pelo painel de controle.
- Interface gráfica separada da lógica de processamento.

### Refatorado
- Criação do módulo `pipeline.py` para centralizar a lógica do projeto.
- Criação do módulo `app.py` para controlar a interface gráfica.
- Reaproveitamento da lógica existente no novo fluxo.

### Estrutura
- `app.py`: interface gráfica e controles da aplicação.
- `pipeline.py`: lógica de processamento e execução do pipeline.

## [0.2.5] - 2026-09-11

### Adicionado
- Suporte à detecção de até duas mãos.
- Suavização temporal das previsões.

### Corrigido
- Reconhecimento entre mão esquerda e direita.

## [0.1.0] - 2026-09-08

### Adicionado
- Primeiro modelo de reconhecimento.
- Coleta de landmarks.
- Treinamento do modelo.
