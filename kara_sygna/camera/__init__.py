"""Acesso à webcam."""

from kara_sygna.camera.captura import (
    Backend,
    Camera,
    CameraIndisponivelError,
    abrir_camera,
    backends_da_plataforma,
)

__all__ = [
    "Backend",
    "Camera",
    "CameraIndisponivelError",
    "abrir_camera",
    "backends_da_plataforma",
]
