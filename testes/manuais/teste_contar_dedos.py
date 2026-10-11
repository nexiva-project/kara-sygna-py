"""Teste manual 4: conta os dedos levantados de cada mão.

python -m testes.manuais.teste_contar_dedos
"""

import sys

import cv2

from kara_sygna.visao.dedos import dedos_levantados
from kara_sygna.visao.detector_maos import DetectorMaos, tem_maos
from testes.manuais._comum import executar


def main() -> int:
    """Executa o teste de contagem de dedos."""
    with DetectorMaos() as detector:

        def processar(quadro):
            """Escreve a contagem de dedos de cada mão sobre o quadro."""
            resultado = detector.detectar(quadro)
            detector.desenhar(quadro, resultado)
            if tem_maos(resultado):
                for numero, mao in enumerate(resultado.multi_hand_landmarks):
                    levantados = dedos_levantados(mao)
                    cv2.putText(
                        quadro,
                        f"Mao {numero + 1}: {len(levantados)} {levantados}",
                        (10, 60 + 30 * numero),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
            return quadro

        return executar("kara-sygna - Contar dedos", processar)


if __name__ == "__main__":
    sys.exit(main())
