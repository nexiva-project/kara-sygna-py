# kara-sygna-py

Reconhece sinais feitos com as mãos pela webcam (OpenCV + MediaPipe + scikit-learn).
Você ensina palavras pela própria janela, treina a IA com um clique e ela escreve o que reconhece.

## Instalação

Use **Python 3.12** (o `mediapipe==0.10.21` não tem pacote para 3.13).

**Ubuntu/Debian**
```bash
sudo apt install python3.12-venv python3-tk v4l-utils   # python3-tk é obrigatório para a janela
python3.12 -m venv .venv && source .venv/bin/activate
pip uninstall -y opencv-python opencv-python-headless   # só se já instalou antes
pip install -r requirements.txt
```
**Windows (PowerShell)**
```bash
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
No PyCharm: *Settings → Project → Python Interpreter* e aponte para a `.venv`.

**Instalar Lefthook e Ruff**
```bash
python -m pip install lefthook
lefthook install
pip install ruff
```

## Como usar

```bash
python app.py            # abre a interface (ou: python -m kara_sygna)
```
1. Escreva a palavra, clique em **COMEÇAR A ENSINAR**, faça o sinal (com o rosto aparecendo na câmera) durante alguns segundos e clique em **PARAR**. Atalho: escreva quantos segundos gravar no campo **TEMPO DE GRAVAÇÃO**, embaixo da câmera, e aperte **Enter**. A gravação começa depois de uma contagem de 3 s e para sozinha no fim do tempo (campo vazio: grava até apertar Enter de novo). Se o sinal tem movimento, repita o movimento algumas vezes.
2. Um Enter já grava o sinal **3 vezes**: `carro 1/3: devagar`, `2/3: médio`, `3/3: rápido` (cada uma é uma *rodada*, o que permite medir a acurácia de verdade). Varie também a posição, a distância e a luz entre as gravações.
   Ao parar, a barra de status mostra a qualidade da gravação; se aparecer "ATENÇÃO, grave de novo", repita aquela rodada.
   **Pré-sinal** (opcional): ligue o botão **Pré-sinal** e escreva a forma de mão que vem antes do sinal (ex.: `S` para `carro`). Ao apertar Enter, a gravação tem duas etapas: primeiro `PRE-SINAL: S` (mostre uma mão em S girando para todos os lados), depois `SINAL: carro`. No reconhecimento, o pré-sinal só ajuda: se a IA vê o S, o carro sai com menos certeza; sem o S, o carro também sai, desde que a IA tenha bastante certeza. Nenhum sinal é bloqueado.
3. Grave também um sinal `neutro` (mão relaxada) para a IA aprender quando *não* há sinal.
4. **TREINAR IA** (roda em segundo plano) e depois **INICIAR RECONHECIMENTO**.

Sem janela (terminal): `python -m kara_sygna.cli.coletar_dados`, `...cli.treinar_modelo`, `...cli.reconhecer_sinais`.

## Organização

```
app.py                    abre a interface
kara_sygna/
  config.py                todas as constantes e limiares
  logs.py                  formato dos logs
  camera/captura.py        abre a webcam (backends por sistema operacional)
  visao/                   mãos -> números
    detector_maos.py         MediaPipe + desenho do esqueleto
    detector_rosto.py        FaceMesh + desenho do rosto
    detector_corpo.py        MediaPipe Pose: ombros, cotovelos e pulsos (movimento)
    preenchimento.py         preenche a mão que some por pouco tempo (gravação)
    maos_pelo_corpo.py       confere as mãos pelos braços (tira mãos fantasmas)
    caracteristicas.py       forma das mãos, world landmarks 3D (126 valores)
    caracteristicas_rosto_maos.py  expressão e distância dedos-rosto (82)
    quadro.py                vetor completo de cada quadro (223)
    memoria_maos.py          mão que some por um instante continua valendo
    realce_pele.py           pele clara e resto escuro para o detector de mãos
    render_3d.py             desenho 3D da pré-visualização
    dedos.py                 contagem de dedos (demonstração)
  dados/                   CSV de exemplos
    repositorio.py           ler/gravar/apagar sinais
    migracao.py              converte arquivos de versões antigas
  ia/                      aprendizado
    sequencia.py             janelas de quadros: pose + lugar + movimento
    treinamento.py           RandomForest + validação por rodada
    aumento.py, geometria.py
    reconhecimento.py        previsão e votação (Estabilizador)
    pre_sinais.py            IA das formas de mão que vêm antes do sinal
    modelo_io.py             salvar/carregar/validar o .pkl
  aplicacao/               regras do programa, sem Tk
    controlador.py, previsualizacao.py, erros.py, inicializacao.py
  interface/               só Tkinter
    janela_principal.py, janela_sinais.py, janela_previsualizacao.py, janela_log.py
    tema.py, imagem_tk.py, tarefas.py, iniciar.py
  cli/                     versões de terminal
armazenamento/             CSV e modelo gerados (não vão para o git)
testes/                    pytest (automáticos) e manuais/ (precisam de câmera)
```
Regra de dependência: `interface` e `cli` usam `aplicacao`; `aplicacao` usa `ia`, `dados`, `visao`, `camera`; nada embaixo importa Tk.

## Comentários no código (docstrings)

Em Python o equivalente ao Javadoc é a **docstring**: um texto entre `"""` logo abaixo do `def`/`class` (PEP 257). Este projeto usa o estilo Google (`Args:`, `Returns:`, `Raises:`). Elas aparecem no PyCharm com `Ctrl+Q` e em `help(funcao)`.

## Testes

```bash
pip install -r requirements-dev.txt
pytest                                      # automáticos, sem câmera
python -m testes.manuais.teste_camera       # a webcam abre? (--foto f.jpg salva um quadro)
python -m testes.manuais.teste_deteccao_maos
python -m testes.manuais.teste_landmarks
python -m testes.manuais.teste_contar_dedos
```

## Problemas comuns (Linux)

- **"Não foi possível acessar nenhuma câmera"**: a mensagem lista cada tentativa. Confira `ls /dev/video*`, `v4l2-ctl --list-devices` e se `groups` inclui `video` (senão `sudo usermod -aG video $USER` e entre de novo na sessão). Feche Zoom/Teams/navegador, que podem estar usando a câmera.
- **`No module named 'tkinter'`**: `sudo apt install python3-tk` (e recrie a `.venv` com o Python do sistema).
- **`mp.solutions` não existe**: versão errada do MediaPipe. Use Python 3.12 e `mediapipe==0.10.21`.
- **Dados antigos**: `dados_sinais.csv` e `modelo_sinais.pkl` da raiz (ou de `src/`) são copiados para `armazenamento/` na primeira execução. Um CSV de formato antigo (só a forma das mãos, sem rosto nem sequência) é guardado como `dados_sinais_formato_antigo_<data>.csv`, junto com o modelo antigo; grave os sinais de novo.
- **Só uma mão aparece / mão some no movimento**: confira o contador `MAOS` sobre a câmera. Mantenha as duas mãos dentro da imagem, com luz na frente (não uma janela atrás) e roupa de cor diferente da pele; movimento rápido com pouca luz borra a mão e o MediaPipe a perde.
- Os números do modelo dependem da proporção da imagem (4:3). A câmera é aberta em 640×480; mude em `config.py` só se for gravar tudo de novo.
- `USAR_LUGAR` em `config.py` (desligado): com ele desligado, o sinal vale em qualquer lugar da imagem. Ligue para sinais que só mudam pelo lugar (ex.: testa x peito) e treine de novo.
- `INVERTER_LATERALIDADE` em `config.py` define qual mão é "esquerda" nos dados. Se mudar, grave os sinais de novo.
