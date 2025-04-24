"""
Manejador de datos de placas detectadas.
Gestiona el almacenamiento, procesamiento y exportación de información de placas.
"""
import os
import cv2
import time
import pandas as pd
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from src.config import settings


class PlateHandler:
    """Clase para manejar los datos de placas detectadas."""

    def __init__(self):
        """Inicializa el manejador de datos de placas."""
        self.plates_data = []  # Lista de todas las detecciones
        self.unique_plates = set()  # Conjunto de placas únicas
        self.last_seen = {}  # Diccionario para rastreo temporal
        self.similar_plates = {}  # Registro de variantes similares

        # Asegurar que exista el directorio de salida
        os.makedirs(settings.PLATES_DIR, exist_ok=True)

    def register_plate(self, frame_number, plate_text, confidence, box, plate_img):
        """
        Registra una nueva placa detectada.

        Args:
            frame_number (int): Número de frame donde se detectó.
            plate_text (str): Texto de la placa.
            confidence (float): Confianza de la detección.
            box (tuple): Coordenadas (x1, y1, x2, y2) de la placa.
            plate_img (numpy.ndarray): Imagen de la placa.

        Returns:
            dict: Datos de la placa registrada o None si se rechazó.
        """
        # Verificar formato de placa válido
        if not self._validate_plate_format(plate_text):
            return None

        # Registrar tiempo de detección
        current_time = time.time()

        # Crear registro de datos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Guardar imagen si está configurado
        image_path = None
        if settings.SAVE_IMAGES:
            image_path = self._save_plate_image(plate_img, frame_number, plate_text)

        # Crear registro de datos
        plate_data = {
            "frame": frame_number,
            "plate": plate_text,
            "confidence": round(float(confidence), 4),
            "timestamp": timestamp,
            "image_path": image_path,
            "box": f"{box[0]},{box[1]},{box[2]},{box[3]}"
        }

        # Almacenar datos
        self.plates_data.append(plate_data)
        self.unique_plates.add(plate_text)
        self.last_seen[plate_text] = current_time

        print(f"Nueva placa registrada: {plate_text} (Confianza: {confidence:.2f})")
        return plate_data

    def has_seen_recently(self, plate_text):
        """
        Verifica si una placa ha sido vista recientemente o es similar a otra vista.

        Args:
            plate_text (str): Texto de la placa a verificar.

        Returns:
            bool: True si se vio recientemente o es similar, False en caso contrario.
        """
        current_time = time.time()

        # Verificar modo de detección única
        if settings.SINGLE_DETECTION_MODE:
            # Verificar si la placa exacta ya está registrada
            if plate_text in self.unique_plates:
                return True

            # Verificar similitud con placas existentes
            if self._is_similar_to_existing(plate_text):
                return True

        # Verificar tiempo de enfriamiento
        if plate_text in self.last_seen:
            time_since_last = current_time - self.last_seen[plate_text]
            if time_since_last < settings.PLATE_COOLDOWN_TIME:
                return True

        return False

    def _is_similar_to_existing(self, plate_text):
        """
        Verifica si la placa es similar a alguna existente.

        Args:
            plate_text (str): Texto de la placa.

        Returns:
            bool: True si es similar a una existente, False en caso contrario.
        """
        for existing in self.unique_plates:
            similarity = SequenceMatcher(None, plate_text, existing).ratio()
            if similarity >= settings.SIMILARITY_THRESHOLD:
                # Registrar para análisis
                self.similar_plates[plate_text] = existing
                return True
        return False

    def _validate_plate_format(self, plate_text):
        """
        Valida el formato de la placa según configuración.

        Args:
            plate_text (str): Texto de la placa.

        Returns:
            bool: True si es válido, False en caso contrario.
        """
        if not plate_text:
            return False

        # Validar con expresión regular configurada
        import re
        return bool(re.match(settings.PLATE_REGEX, plate_text))

    def _save_plate_image(self, plate_img, frame_number, plate_text):
        """
        Guarda la imagen de una placa detectada.

        Args:
            plate_img (numpy.ndarray): Imagen de la placa.
            frame_number (int): Número de frame.
            plate_text (str): Texto de la placa para nombrar el archivo.

        Returns:
            str: Ruta del archivo guardado.
        """
        # Generar timestamp para nombre único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Crear nombre de archivo seguro
        safe_text = ''.join(c if c.isalnum() or c in '-_' else '_' for c in plate_text)
        filename = f"placa_{safe_text}_{frame_number}_{timestamp}.jpg"

        # Ruta completa
        file_path = Path(settings.PLATES_DIR) / filename

        # Guardar imagen
        cv2.imwrite(str(file_path), plate_img)

        return str(file_path)

    def get_plate_count(self):
        """
        Obtiene el número total de placas detectadas.

        Returns:
            int: Número de placas detectadas.
        """
        return len(self.plates_data)

    def get_unique_count(self):
        """
        Obtiene el número de placas únicas detectadas.

        Returns:
            int: Número de placas únicas.
        """
        return len(self.unique_plates)

    def get_unique_plates(self):
        """
        Obtiene la lista de textos de placas únicas.

        Returns:
            list: Lista de textos de placas únicas.
        """
        return list(self.unique_plates)

    def save_data(self, format_type=None):
        """
        Guarda los datos de placas en el formato especificado.

        Args:
            format_type (str, optional): Formato de exportación ('csv', 'json').
                Si es None, usa el configurado en settings.

        Returns:
            bool: True si se guardó correctamente, False en caso contrario.
        """
        if not self.plates_data:
            print("No hay datos de placas para guardar")
            return False

        # Determinar formato
        if format_type is None:
            format_type = settings.EXPORT_FORMAT

        try:
            # Crear DataFrame
            df = pd.DataFrame(self.plates_data)

            # Guardar según formato
            if format_type.lower() == 'csv':
                df.to_csv(settings.DATASET_PATH, index=False, encoding='utf-8-sig')
                print(f"Datos guardados en CSV: {settings.DATASET_PATH}")

            elif format_type.lower() == 'json':
                json_path = Path(settings.DATASET_PATH).with_suffix('.json')
                df.to_json(json_path, orient='records', indent=4)
                print(f"Datos guardados en JSON: {json_path}")

            else:
                print(f"Formato de exportación no soportado: {format_type}")
                return False

            # Guardar registro de similitudes si está configurado
            if settings.SAVE_SIMILARITIES and self.similar_plates:
                similarity_data = [{"variante": variant, "placa_principal": main}
                                   for variant, main in self.similar_plates.items()]

                if similarity_data:
                    sim_df = pd.DataFrame(similarity_data)
                    similarity_path = Path(settings.OUTPUT_DIR) / "placas_similares.csv"
                    sim_df.to_csv(similarity_path, index=False, encoding='utf-8-sig')
                    print(f"Registro de similitudes guardado en: {similarity_path}")

            return True

        except Exception as e:
            print(f"Error guardando datos: {e}")
            return False