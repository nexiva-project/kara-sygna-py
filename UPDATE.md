```
  _    _ _____  _____       _______ ______ 
 | |  | |  __ \|  __ \   /\|__   __|  ____|
 | |  | | |__) | |  | | /  \  | |  | |__   
 | |  | |  ___/| |  | |/ /\ \ | |  |  __|  
 | |__| | |    | |__| / ____ \| |  | |____ 
  \____/|_|    |_____/_/    \_\_|  |______|                                                                               
```
                                          
---

**Atualização importante (suporte a duas mãos)**: agora o projeto
detecta até 2 mãos ao mesmo tempo, para sinais como "carro" que usam
ambas. Cada amostra salva um vetor de tamanho fixo com a mão esquerda
e a direita (preenchendo com zeros a mão que não aparecer). Isso muda
o formato do CSV — **apague o `dados_sinais.csv` e o
`modelo_sinais.pkl` antigos e colete TODOS os sinais de novo**
(inclusive os de uma mão só, como os números), já que todas as linhas
precisam ter a mesma quantidade de colunas.

**Atualização (suavização temporal)**: para reduzir confusão entre
sinais parecidos (ex: "3" e "4"), o reconhecimento agora olha os
últimos 10 frames e só mostra um sinal quando pelo menos 6 deles
concordam. Isso estabiliza a previsão e evita "piscar" entre sinais
por causa de ruído momentâneo da câmera. Se quiser ajustar a
sensibilidade, mexa em `TAMANHO_JANELA` e `MINIMO_DE_VOTOS` no início
do `reconhecer_sinais.py`.
 
Se mesmo assim sinais parecidos continuarem se confundindo, o próximo
passo é coletar mais amostras desses sinais especificamente, variando
mais o ângulo e a distância da mão durante a gravação.
 
### Problemas comuns
 
- **Sinal reconhecido só quando feito com a mesma mão que você usou
  para gravar** (ex: gravou "O" com a esquerda, mas com a direita dá
  "Incerto"): isso acontecia porque os dados eram salvos em posições
  fixas (mão esquerda / mão direita) e o modelo não generalizava entre
  elas. Agora o `coletar_dados.py` já salva automaticamente uma versão
  espelhada de cada sinal de mão única. Se você coletou dados **antes**
  dessa correção, rode uma vez: `python src/aumentar_dados.py`, depois
  retreine com `python src/treinar_modelo.py`.


- **Dois sinais com o mesmo formato de mão nunca serão distinguidos**
  (ex: letra "O" e número "0" em Libras costumam ser o mesmo formato
  de mão). Isso não é um bug — é uma limitação real de qualquer
  classificador baseado só na forma da mão: se dois sinais são
  fisicamente idênticos, não há como diferenciá-los apenas pelos
  landmarks. A solução é decidir um contexto (por exemplo, um "modo"
  de letras vs números no seu app) ou unificar os dois em um único
  rótulo.
--- 
- **`AttributeError: module 'mediapipe' has no attribute 'solutions'`**: a partir da versão 0.10.31, o Google removeu a API antiga (`mediapipe.solutions`) do pacote. Use a versão fixada no `requirements.txt` (0.10.21), que ainda tem essa API. Se já instalou a versão errada, rode: `pip install mediapipe==0.10.21`

---

- **Janela não abre / trava**: confirme que nenhum outro programa (Zoom, Teams, etc.) está usando a câmera.

---

- **`cv2.VideoCapture(0)` não encontra a câmera**: tente trocar o `0` por `1` no arquivo `src/camera.py`.

---

- **Erro ao instalar opencv-python**: confirme que está usando Python 3.9–3.12 e que o `pip` está atualizado (`pip install --upgrade pip`).