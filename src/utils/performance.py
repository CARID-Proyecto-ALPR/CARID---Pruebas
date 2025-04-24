"""
Utilidades para medición de rendimiento del sistema CARID.
"""
import time
from collections import deque


class FPSCounter:
    """Clase para calcular y gestionar FPS (frames por segundo)."""

    def __init__(self, max_samples=10):
        """
        Inicializa el contador de FPS.

        Args:
            max_samples (int): Número máximo de muestras para calcular el promedio.
        """
        self.frame_times = deque(maxlen=max_samples)
        self.last_time = time.time()
        self.frame_count = 0

    def update(self):
        """
        Actualiza el contador con un nuevo frame.

        Returns:
            float: FPS actual basado en las últimas muestras.
        """
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time

        self.frame_times.append(frame_time)
        self.frame_count += 1

        return self.get_fps()

    def get_fps(self):
        """
        Obtiene el FPS actual.

        Returns:
            float: FPS actual basado en las últimas muestras.
        """
        if not self.frame_times:
            return 0

        # Calcular promedio de tiempo por frame
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)

        # Evitar división por cero
        if avg_frame_time == 0:
            return 0

        return 1.0 / avg_frame_time

    def get_total_fps(self):
        """
        Obtiene el FPS promedio desde el inicio.

        Returns:
            float: FPS promedio total.
        """
        total_time = time.time() - self.last_time + sum(self.frame_times)
        if total_time <= 0 or self.frame_count == 0:
            return 0

        return self.frame_count / total_time


class Timer:
    """Clase para medir tiempos de ejecución."""

    def __init__(self, name="Operation"):
        """
        Inicializa el temporizador.

        Args:
            name (str): Nombre de la operación a medir.
        """
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Inicia el temporizador al entrar en un bloque with."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finaliza el temporizador al salir de un bloque with."""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.name} completado en {elapsed:.4f} segundos")

    def elapsed(self):
        """
        Calcula el tiempo transcurrido.

        Returns:
            float: Tiempo transcurrido en segundos o None si no se ha detenido.
        """
        if self.start_time is None:
            return None

        if self.end_time is None:
            return time.time() - self.start_time

        return self.end_time - self.start_time


def profile_function(func):
    """
    Decorador para perfilar el tiempo de ejecución de una función.

    Args:
        func (callable): Función a perfilar.

    Returns:
        callable: Función envuelta que imprime su tiempo de ejecución.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"Función {func.__name__} ejecutada en {elapsed:.4f} segundos")
        return result

    return wrapper