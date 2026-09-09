# kara-sygna-py

Projeto de reconhecimento de sinais de mão via câmera, usando Python.

Roteiro de aprendizado (8 passos):
1. **Preparar o ambiente Python** ✅ (feito abaixo)
2. **Instalar OpenCV e capturar vídeo** ✅ (este passo — `src/camera.py`)
3. **Detectar a mão com MediaPipe** ✅ (este passo — `src/hand_tracker.py`)
4. **Extrair e organizar os landmarks** ✅ (este passo — `src/landmarks.py`)
5. Reconhecer gestos simples
6. Coletar dados para sinais mais específicos
7. Treinar um classificador simples
8. Rodar em tempo real e organizar o projeto

## Como configurar o ambiente (rodar na SUA máquina)

```bash
# 1. Entre na pasta do projeto
cd kara-sygna-py

# 2. Crie um ambiente virtual
python -m venv venv

# 3. Ative o ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Instale as dependências
pip install -r requirements.txt
```

## Como rodar o Passo 1

```bash
python src/camera.py
```

Uma janela deve abrir mostrando sua webcam ao vivo (espelhada, como um espelho).
Pressione `q` com a janela em foco para fechar.

## Como rodar o Passo 3

```bash
python src/hand_tracker.py
```

Mostre a mão para a câmera. Você deve ver 21 pontinhos e linhas desenhados
sobre os dedos e a palma, seguindo o movimento da mão em tempo real.

## Como rodar o Passo 4

```bash
python src/landmarks.py
```

Além de desenhar a mão, o script agora imprime no terminal e mostra na
tela a posição (x, y, z) da ponta do dedo indicador em tempo real.

### Problemas comuns
- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`**: a partir da versão 0.10.31, o Google removeu a API antiga (`mediapipe.solutions`) do pacote. Use a versão fixada no `requirements.txt` (0.10.21), que ainda tem essa API. Se já instalou a versão errada, rode: `pip install mediapipe==0.10.21`
- **Janela não abre / trava**: confirme que nenhum outro programa (Zoom, Teams, etc.) está usando a câmera.
- **`cv2.VideoCapture(0)` não encontra a câmera**: tente trocar o `0` por `1` no arquivo `src/camera.py`.
- **Erro ao instalar opencv-python**: confirme que está usando Python 3.9–3.12 e que o `pip` está atualizado (`pip install --upgrade pip`).