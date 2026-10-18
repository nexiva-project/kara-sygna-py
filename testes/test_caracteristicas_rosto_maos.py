"""Teste integrado Kara Sygna — FaceMesh + mãos suavizadas."""

import copy

import cv2
import numpy as np

from kara_sygna.visao.caracteristicas_rosto_maos import (
    TOTAL_FEATURES_ROSTO_MAOS,
    extrair_features_rosto_maos,
    nomes_features_rosto_maos,
)
from kara_sygna.visao.detector_maos import DetectorMaos
from kara_sygna.visao.detector_rosto import DetectorRosto, ResultadoRosto
from kara_sygna.visao.suavizacao import SuavizadorPontos

# Equilíbrio inicial entre estabilidade e velocidade.
INTENSIDADE_ROSTO = 0.65
INTENSIDADE_MAOS = 0.75


def suavizar_landmarks(landmarks, suavizador, identificador):
    """Suaviza uma cópia dos pontos, sem modificar os pontos originais."""

    if landmarks is None:
        suavizador.limpar(identificador)
        return None

    copia = copy.deepcopy(landmarks)

    coordenadas = np.asarray(
        [(p.x, p.y, p.z) for p in copia.landmark],
        dtype=np.float32,
    )

    suavizadas = suavizador.suavizar(
        identificador,
        coordenadas,
    )

    for ponto, coordenada in zip(copia.landmark, suavizadas):
        ponto.x = float(coordenada[0])
        ponto.y = float(coordenada[1])
        ponto.z = float(coordenada[2])

    return copia


def main():
    detector_rosto = None
    detector_maos = None
    camera = None

    suavizador_rosto = SuavizadorPontos(
        intensidade=0.30,
        zona_morta=0.002,
        limiar_movimento=0.012,
        resposta_movimento=0.85,
    )

    suavizador_maos = SuavizadorPontos(
        intensidade=INTENSIDADE_MAOS,
    )

    try:
        detector_rosto = DetectorRosto()
        detector_maos = DetectorMaos()

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            print("ERRO: não foi possível abrir a câmera.")
            return

        print("Teste integrado iniciado.")
        print("Total de características:", TOTAL_FEATURES_ROSTO_MAOS)
        print("Total de nomes:", len(nomes_features_rosto_maos()))
        print("Suavização do rosto:", INTENSIDADE_ROSTO)
        print("Suavização das mãos:", INTENSIDADE_MAOS)
        print("Pressione Q para sair.")

        contador = 0

        while True:
            sucesso, quadro = camera.read()

            if not sucesso:
                print("ERRO: não foi possível capturar o quadro.")
                break

            quadro = cv2.flip(quadro, 1)
            altura, largura = quadro.shape[:2]

            # Detecção original: não aplicamos filtros aos dados do modelo.
            resultado_rosto = detector_rosto.detectar(quadro)
            resultado_maos = detector_maos.detectar(quadro)

            vetor = extrair_features_rosto_maos(
                resultado_rosto,
                resultado_maos,
                largura,
                altura,
            )

            # Suaviza uma cópia dos pontos faciais para desenhar.
            if resultado_rosto.presente and resultado_rosto.landmarks is not None:
                rosto_visual = suavizar_landmarks(
                    resultado_rosto.landmarks,
                    suavizador_rosto,
                    "rosto",
                )

                resultado_rosto_visual = ResultadoRosto(
                    landmarks=rosto_visual,
                    vetor=resultado_rosto.vetor,
                    presente=True,
                )
            else:
                suavizador_rosto.limpar("rosto")
                resultado_rosto_visual = resultado_rosto

            # Desenha a malha facial suavizada.
            tela = detector_rosto.desenhar(
                quadro,
                resultado_rosto_visual,
            )

            # Copia o resultado das mãos para preservar os dados originais.
            maos_visual = copy.deepcopy(resultado_maos)
            listas = getattr(maos_visual, "multi_hand_landmarks", None)
            lateralidades = getattr(maos_visual, "multi_handedness", None)

            maos_encontradas = set()

            if listas and lateralidades:
                for landmarks, info in zip(listas, lateralidades):
                    if not landmarks.landmark:
                        continue

                    rotulo = info.classification[0].label
                    identificador = f"mao_{rotulo.lower()}"
                    maos_encontradas.add(identificador)

                    pontos_suavizados = suavizar_landmarks(
                        landmarks,
                        suavizador_maos,
                        identificador,
                    )

                    if pontos_suavizados is not None:
                        for original, suavizado in zip(
                            landmarks.landmark,
                            pontos_suavizados.landmark,
                        ):
                            original.x = suavizado.x
                            original.y = suavizado.y
                            original.z = suavizado.z

            # Limpa o histórico das mãos que não foram detectadas.
            for identificador in ("mao_left", "mao_right"):
                if identificador not in maos_encontradas:
                    suavizador_maos.limpar(identificador)

            # O detector modifica a imagem recebida.
            detector_maos.desenhar(tela, maos_visual)

            contador += 1

            if contador == 1 or contador % 30 == 0:
                print(
                    f"Quadro {contador} | "
                    f"Vetor: {len(vetor)} | "
                    f"Rosto: {int(vetor[0])} | "
                    f"Mão à esquerda: {int(vetor[1])} | "
                    f"Mão à direita: {int(vetor[2])}"
                )

            cv2.imshow("Kara Sygna - Malhas Suavizadas", tela)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as erro:
        print(f"ERRO durante o teste: {erro}")
        raise

    finally:
        if camera is not None:
            camera.release()

        if detector_rosto is not None:
            detector_rosto.fechar()

        if detector_maos is not None:
            detector_maos.fechar()

        cv2.destroyAllWindows()
        print("Teste encerrado.")


if __name__ == "__main__":
    main()
