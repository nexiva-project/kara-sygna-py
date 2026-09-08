# kara-sygna-py
 
Projeto de reconhecimento de sinais de mão via câmera, usando Python.
 
Roteiro de aprendizado (8 passos):
1. **Preparar o ambiente Python** ✅ (feito abaixo)
2. **Instalar OpenCV e capturar vídeo** ✅ (este passo — `src/camera.py`)
3. Detectar a mão com MediaPipe
4. Extrair e organizar os landmarks
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
 
### Problemas comuns
- **Janela não abre / trava**: confirme que nenhum outro programa (Zoom, Teams, etc.) está usando a câmera.
- **`cv2.VideoCapture(0)` não encontra a câmera**: tente trocar o `0` por `1` no arquivo `src/camera.py`.
- **Erro ao instalar opencv-python**: confirme que está usando Python 3.9–3.12 e que o `pip` está atualizado (`pip install --upgrade pip`).