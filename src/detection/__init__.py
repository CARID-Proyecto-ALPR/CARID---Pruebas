"""
Módulo de detección de placas para el sistema CARID.
Proporciona interfaces para diferentes algoritmos de detección.
"""
from src.detection.detector import Detector
from src.detection.yolo_detector import YoloDetector


def create_detector(detector_type="yolo"):
    """
    Fábrica para crear el detector especificado.

    Args:
        detector_type (str): Tipo de detector a crear.
            Opciones: "yolo" (por defecto)

    Returns:
        Detector: Instancia del detector solicitado.

    Raises:
        ValueError: Si el tipo de detector no es soportado.
    """
    if detector_type.lower() == "yolo":
        return YoloDetector()
    else:
        raise ValueError(f"Tipo de detector no soportado: {detector_type}")