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
1. Escreva a palavra, clique em **COMEÇAR A ENSINAR**, faça o sinal e clique em **PARAR**.
2. Repita **pelo menos 3 vezes por sinal**, mudando posição, distância, inclinação e luz. Cada clique é uma *rodada*, e é isso que permite medir a acurácia de verdade.
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
    caracteristicas.py       vetor de 126 valores por quadro
    render_3d.py             desenho 3D da pré-visualização
    dedos.py                 contagem de dedos (demonstração)
  dados/                   CSV de exemplos
    repositorio.py           ler/gravar/apagar sinais
    migracao.py              converte arquivos de versões antigas
  ia/                      aprendizado
    treinamento.py           RandomForest + validação por rodada
    aumento.py, rotacao.py, geometria.py
    reconhecimento.py        previsão e votação (Estabilizador)
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
- **Dados antigos**: `dados_sinais.csv` e `modelo_sinais.pkl` da raiz (ou de `src/`) são copiados para `armazenamento/` na primeira execução, e o CSV ganha a coluna `rodada` (com backup ao lado).
- Os números do modelo dependem da proporção da imagem (4:3). A câmera é aberta em 640×480; mude em `config.py` só se for gravar tudo de novo.
- `INVERTER_LATERALIDADE` em `config.py` define qual mão é "esquerda" nos dados. Se mudar, grave os sinais de novo.
