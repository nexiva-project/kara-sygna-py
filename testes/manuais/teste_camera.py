"""Teste manual 1: a câmera abre e mostra imagem?

python -m testes.manuais.teste_camera
python -m testes.manuais.teste_camera --foto foto.jpg     # sem janela
"""

import sys

from testes.manuais._comum import executar

if __name__ == "__main__":
    sys.exit(executar("kara-sygna - Teste de camera"))
