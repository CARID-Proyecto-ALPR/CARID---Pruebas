"""
Implementación del detector de placas utilizando YOLOv8.
"""
import numpy as np
from ultralytics import YOLO

from src.config import settings
from src.detection.detector import Detector


class YoloDetector(Detector):
    """Detector de placas basado en YOLOv8."""

    def __init__(self):
        """Inicializa el detector YOLOv8."""
        super().__init__()
        try:
            # Cargar modelo YOLOv8
            self.model = YOLO(str(settings.MODEL_PATH))
            self._initialized = True
            print(f"Detector YOLOv8 inicializado: {settings.MODEL_PATH}")
        except Exception as e:
            print(f"Error inicializando detector YOLOv8: {e}")
            self._initialized = False

    def detect(self, frame):
        """
        Detecta placas en el frame proporcionado.

        Args:
            frame (numpy.ndarray): Frame de imagen a procesar.

        Returns:
            list: Lista de detecciones. Cada detección es un diccionario con:
                - box (tuple): Coordenadas (x1, y1, x2, y2) del recuadro.
                - confidence (float): Nivel de confianza de la detección.
        """
        if not self._initialized:
            return []

        try:
            # Realizar detección con YOLOv8
            results = self.model(
                frame,
                imgsz=640,
                conf=settings.CONFIDENCE_THRESHOLD,
                verbose=False
            )

            detections = []

            # Procesar resultados
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = map(int, box)
                    detections.append({
                        "box": (x1, y1, x2, y2),
                        "confidence": float(conf)
                    })

            return detections

        except Exception as e:
            print(f"Error en detección YOLOv8: {e}")
            return []

    def extract_region(self, frame, box):
        """
        Extrae la región de la placa del frame.

        Args:
            frame (numpy.ndarray): Frame completo.
            box (tuple): Coordenadas (x1, y1, x2, y2) de la placa.

        Returns:
            numpy.ndarray: Imagen recortada de la placa.
        """
        x1, y1, x2, y2 = box
        return frame[y1:y2, x1:x2]