"""Compara 3 jeitos de descrever a forma da mão para a IA.

python -m testes.manuais.comparar_world_landmarks

* ``atual``: pontos da imagem (x, y, z do MediaPipe), como a IA usa hoje.
  x e y são frações da largura e da altura; numa imagem 4:3 a mão fica
  "esticada" na horizontal, e o z é uma estimativa fraca.
* ``atual_4x3``: os mesmos pontos com o x corrigido pela proporção da imagem.
* ``world``: os *world landmarks* do MediaPipe, em metros, em 3D.

Todos passam pela mesma normalização (centro no pulso, divididos pela palma).
O teste grava 5 formas de mão, duas vezes, com UMA mão (a direita). Treina na
1ª rodada e testa na 2ª (e vice-versa): quanto mais alta a acurácia, melhor
o jeito de descrever a mão.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from kara_sygna.camera import abrir_camera
from kara_sygna.visao.caracteristicas import normalizar_mao
from kara_sygna.visao.detector_maos import DetectorMaos

FORMAS = [
    "mao aberta",
    "punho fechado",
    "L (polegar e indicador)",
    "indicador para cima",
    "indicador apontando para a camera",
]
RODADAS = 2
CONTAGEM_S, GRAVACAO_S = 3, 8
REPRESENTACOES = ("atual", "atual_4x3", "world")


def _pontos(landmarks) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float64)


def _representacoes(resultado, largura: int, altura: int) -> dict | None:
    imagem = getattr(resultado, "multi_hand_landmarks", None)
    mundo = getattr(resultado, "multi_hand_world_landmarks", None)
    if not imagem or not mundo:
        return None
    pontos = _pontos(imagem[0])
    corrigidos = pontos * np.array([largura / altura, 1.0, largura / altura])
    saida = {
        "atual": normalizar_mao(pontos),
        "atual_4x3": normalizar_mao(corrigidos),
        "world": normalizar_mao(_pontos(mundo[0])),
    }
    if any(v is None for v in saida.values()):
        return None
    return {k: v.reshape(-1) for k, v in saida.items()}


def gravar(camera, detector) -> dict:
    dados = {k: [] for k in REPRESENTACOES}
    dados["forma"], dados["rodada"] = [], []
    for rodada in range(RODADAS):
        for indice, forma in enumerate(FORMAS):
            inicio = time.monotonic()
            while True:
                quadro = camera.ler()
                if quadro is None:
                    continue
                quadro = cv2.flip(quadro, 1)
                agora = time.monotonic() - inicio
                if agora > CONTAGEM_S + GRAVACAO_S:
                    break
                resultado = detector.detectar(quadro)
                detector.desenhar(quadro, resultado)
                titulo = f"Rodada {rodada + 1}/{RODADAS}: {forma.upper()}"
                if agora < CONTAGEM_S:
                    texto, cor = (
                        f"Prepare  {int(CONTAGEM_S - agora) + 1}",
                        (0, 200, 255),
                    )
                else:
                    texto = "Gire o pulso, chegue perto e longe"
                    cor = (0, 0, 255)
                    altura, largura = quadro.shape[:2]
                    linha = _representacoes(resultado, largura, altura)
                    if linha:
                        for k in REPRESENTACOES:
                            dados[k].append(linha[k])
                        dados["forma"].append(indice)
                        dados["rodada"].append(rodada)
                cv2.putText(
                    quadro, titulo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2
                )
                cv2.putText(
                    quadro, texto, (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2
                )
                cv2.imshow("Comparar world landmarks", quadro)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt
    cv2.destroyAllWindows()
    return {k: np.asarray(v) for k, v in dados.items()}


def separabilidade(X: np.ndarray, y: np.ndarray) -> float:
    """Distância média entre formas dividida pela variação dentro de cada forma."""
    centros = np.array([X[y == c].mean(axis=0) for c in np.unique(y)])
    dentro = np.mean(
        [
            np.linalg.norm(X[y == c] - centros[i], axis=1).mean()
            for i, c in enumerate(np.unique(y))
        ]
    )
    entre = np.mean(
        [np.linalg.norm(a - b) for i, a in enumerate(centros) for b in centros[i + 1 :]]
    )
    return entre / dentro


def avaliar(dados: dict) -> None:
    y, rodada = dados["forma"], dados["rodada"]
    print(f"\nQuadros com mão: {len(y)}  (por forma: {np.bincount(y).tolist()})")
    print(f"{'jeito':<11} {'acurácia':>9} {'separabilidade':>15}")
    for k in REPRESENTACOES:
        X = dados[k]
        acertos = []
        for treino in range(RODADAS):
            teste = 1 - treino
            modelo = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1)
            modelo.fit(X[rodada == treino], y[rodada == treino])
            acertos.append(
                (modelo.predict(X[rodada == teste]) == y[rodada == teste]).mean()
            )
        print(f"{k:<11} {np.mean(acertos):>8.1%} {separabilidade(X, y):>15.2f}")


def main() -> int:
    saida = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("comparacao_world.npz")
    if saida.exists() and saida.suffix == ".npz" and "--reavaliar" in sys.argv:
        avaliar(dict(np.load(saida)))
        return 0
    with abrir_camera() as camera, DetectorMaos() as detector:
        dados = gravar(camera, detector)
    np.savez(saida, **dados)
    print(f"Dados salvos em {saida}")
    avaliar(dados)
    return 0


if __name__ == "__main__":
    sys.exit(main())
