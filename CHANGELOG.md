# Changelog

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