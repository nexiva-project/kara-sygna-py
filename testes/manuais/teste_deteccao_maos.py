"""Teste manual 2: o MediaPipe enxerga as mãos? (desenha o esqueleto)

python -m testes.manuais.teste_deteccao_maos
"""

import sys

from kara_sygna.visao.detector_maos import DetectorMaos
from testes.manuais._comum import executar


def main() -> int:
    """Executa o teste com o detector de mãos."""
    with DetectorMaos() as detector:

        def processar(quadro):
            """Desenha o esqueleto das mãos sobre o quadro."""
            detector.desenhar(quadro, detector.detectar(quadro))
            return quadro

        return executar("kara-sygna - Deteccao de maos", processar)


if __name__ == "__main__":
    sys.exit(main())
