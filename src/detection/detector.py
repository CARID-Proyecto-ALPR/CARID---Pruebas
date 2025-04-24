"""
Interfaz base para todos los detectores de placas.
Define el contrato que deben cumplir todas las implementaciones.
"""
from abc import ABC, abstractmethod


class Detector(ABC):
    """Clase base abstracta para detectores de placas."""

    def __init__(self):
        """Inicializa el detector base."""
        self._initialized = False

    @abstractmethod
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
        pass

    @abstractmethod
    def extract_region(self, frame, box):
        """
        Extrae la región de la placa del frame.

        Args:
            frame (numpy.ndarray): Frame completo.
            box (tuple): Coordenadas (x1, y1, x2, y2) de la placa.

        Returns:
            numpy.ndarray: Imagen recortada de la placa.
        """
        pass

    def is_initialized(self):
        """
        Verifica si el detector está inicializado correctamente.

        Returns:
            bool: True si está inicializado, False en caso contrario.
        """
        return self._initialized