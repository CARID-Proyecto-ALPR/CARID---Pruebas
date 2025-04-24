"""
Interfaz base para todos los motores de OCR.
Define el contrato que deben cumplir todas las implementaciones.
"""
from abc import ABC, abstractmethod


class OCREngine(ABC):
    """Clase base abstracta para motores de OCR."""

    def __init__(self):
        """Inicializa el motor OCR base."""
        self._initialized = False
        self._debug_dir = None

    @abstractmethod
    def recognize_text(self, image):
        """
        Reconoce texto en la imagen proporcionada.

        Args:
            image (numpy.ndarray): Imagen de la placa a procesar.

        Returns:
            str: Texto reconocido o None si no se detectó.
        """
        pass

    @abstractmethod
    def preprocess_image(self, image):
        """
        Preprocesa la imagen para mejorar el reconocimiento.

        Args:
            image (numpy.ndarray): Imagen original.

        Returns:
            numpy.ndarray: Imagen preprocesada.
        """
        pass

    def set_debug_mode(self, debug_dir=None):
        """
        Activa el modo de depuración y establece el directorio para guardar imágenes.

        Args:
            debug_dir (str): Ruta al directorio para guardar imágenes de depuración.
        """
        self._debug_dir = debug_dir

    def is_initialized(self):
        """
        Verifica si el motor OCR está inicializado correctamente.

        Returns:
            bool: True si está inicializado, False en caso contrario.
        """
        return self._initialized