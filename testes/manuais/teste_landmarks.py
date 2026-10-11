"""Teste manual 3: mostra as coordenadas do indicador e do vetor de 126 valores.

python -m testes.manuais.teste_landmarks
"""

import sys

import cv2

from kara_sygna.visao.caracteristicas import montar_vetor_duas_maos, quantidade_de_maos
from kara_sygna.visao.detector_maos import DetectorMaos, tem_maos
from testes.manuais._comum import executar

PONTA_DO_INDICADOR = 8


def main() -> int:
    """Executa o teste de landmarks."""
    with DetectorMaos() as detector:

        def processar(quadro):
            """Escreve as coordenadas do indicador e a quantidade de mãos sobre o quadro."""
            resultado = detector.detectar(quadro)
            detector.desenhar(quadro, resultado)
            if tem_maos(resultado):
                ponto = resultado.multi_hand_landmarks[0].landmark[PONTA_DO_INDICADOR]
                n = quantidade_de_maos(montar_vetor_duas_maos(resultado))
                texto = f"Indicador x={ponto.x:.3f} y={ponto.y:.3f} z={ponto.z:.3f} | maos no vetor: {n}"
                cv2.putText(
                    quadro,
                    texto,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            return quadro

        return executar("kara-sygna - Landmarks", processar)


if __name__ == "__main__":
    sys.exit(main())
