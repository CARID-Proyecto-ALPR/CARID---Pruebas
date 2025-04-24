"""
Módulo de visualización para el sistema CARID.
Maneja la presentación de resultados en tiempo real.
"""
import cv2
import numpy as np
from src.config import settings


class DisplayManager:
    """Gestor de visualización en tiempo real."""

    def __init__(self, window_name="CARID - Detección de Placas"):
        """
        Inicializa el gestor de visualización.

        Args:
            window_name (str): Nombre de la ventana de visualización.
        """
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, settings.DISPLAY_WIDTH, settings.DISPLAY_HEIGHT)

    def draw_detection(self, frame, box, text=None, confidence=None):
        """
        Dibuja una detección de placa en el frame.

        Args:
            frame (numpy.ndarray): Frame a modificar.
            box (tuple): Coordenadas (x1, y1, x2, y2) de la placa.
            text (str, optional): Texto de la placa detectado por OCR.
            confidence (float, optional): Nivel de confianza de la detección.

        Returns:
            numpy.ndarray: Frame con la detección dibujada.
        """
        x1, y1, x2, y2 = box

        # Dibujar rectángulo
        cv2.rectangle(frame, (x1, y1), (x2, y2), settings.COLOR_DETECTION, 2)

        # Dibujar texto si está disponible
        if text:
            # Preparar texto con confianza si está disponible
            display_text = text
            if confidence is not None:
                display_text = f"{text} ({confidence:.2f})"

            # Determinar posición y fondo
            text_size = cv2.getTextSize(
                display_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                settings.FONT_SCALE,
                settings.LINE_THICKNESS
            )[0]

            # Calcular posición y tamaño del fondo
            text_x = x1
            text_y = y1 - 10

            # Dibujar fondo para el texto (rectángulo semi-transparente)
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (text_x, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (0, 0, 0),
                -1
            )
            # Aplicar transparencia
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Dibujar texto
            cv2.putText(
                frame,
                display_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                settings.FONT_SCALE,
                settings.COLOR_TEXT,
                settings.LINE_THICKNESS
            )

        return frame

    def add_info_overlay(self, frame, info):
        """
        Añade información de estado al frame.

        Args:
            frame (numpy.ndarray): Frame a modificar.
            info (dict): Información a mostrar.
                - mode (str): Modo actual (seguimiento/normal).
                - fps (float): FPS actuales.
                - frame (int): Número de frame.
                - plates (int): Total de placas detectadas.
                - unique_plates (int): Total de placas únicas.

        Returns:
            numpy.ndarray: Frame con la información añadida.
        """
        # Crear texto con la información
        info_text = f"Modo: {info.get('mode', 'N/A')} | "
        info_text += f"FPS: {info.get('fps', 0):.1f} | "
        info_text += f"Frame: {info.get('frame', 0)} | "
        info_text += f"Placas: {info.get('plates', 0)}"

        if 'unique_plates' in info:
            info_text += f" | Únicas: {info['unique_plates']}"

        # Dibujar fondo para el texto
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 50), (0, 0, 0), -1)

        # Aplicar transparencia
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Dibujar texto
        cv2.putText(
            frame,
            info_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        return frame

    def resize_frame(self, frame):
        """
        Redimensiona el frame para visualización manteniendo aspect ratio.

        Args:
            frame (numpy.ndarray): Frame original.

        Returns:
            numpy.ndarray: Frame redimensionado.
        """
        h, w = frame.shape[:2]

        # Calcular ratio de aspecto
        target_ratio = settings.DISPLAY_WIDTH / settings.DISPLAY_HEIGHT
        frame_ratio = w / h

        # Determinar dimensiones manteniendo aspect ratio
        if frame_ratio > target_ratio:
            # Ancho es el factor limitante
            new_w = settings.DISPLAY_WIDTH
            new_h = int(settings.DISPLAY_WIDTH / frame_ratio)
        else:
            # Alto es el factor limitante
            new_h = settings.DISPLAY_HEIGHT
            new_w = int(settings.DISPLAY_HEIGHT * frame_ratio)

        # Redimensionar
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Añadir bordes si es necesario
        if new_w != settings.DISPLAY_WIDTH or new_h != settings.DISPLAY_HEIGHT:
            # Calcular bordes
            delta_w = settings.DISPLAY_WIDTH - new_w
            delta_h = settings.DISPLAY_HEIGHT - new_h
            top = delta_h // 2
            bottom = delta_h - top
            left = delta_w // 2
            right = delta_w - left

            # Añadir bordes negros
            resized = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        return resized

    def show_frame(self, frame, info=None):
        """
        Muestra un frame con información opcional.

        Args:
            frame (numpy.ndarray): Frame a mostrar.
            info (dict, optional): Información a mostrar como overlay.
        """
        # Redimensionar frame
        display_frame = self.resize_frame(frame.copy())

        # Añadir información si se proporciona
        if info:
            display_frame = self.add_info_overlay(display_frame, info)

        # Mostrar frame
        cv2.imshow(self.window_name, display_frame)

    def cleanup(self):
        """Libera recursos de visualización."""
        cv2.destroyAllWindows()