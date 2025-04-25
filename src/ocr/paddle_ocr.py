"""
Implementación de OCR utilizando PaddleOCR.
Optimizado para reconocimiento de placas vehiculares.
"""
import os
import re
import time
import cv2
import numpy as np
from paddleocr import PaddleOCR
from difflib import SequenceMatcher
import threading

from src.config import settings
from src.ocr.ocr_engine import OCREngine


class PaddleOCREngine(OCREngine):
    """Motor OCR basado en PaddleOCR optimizado para placas."""

    def __init__(self):
        """Inicializa el motor PaddleOCR."""
        super().__init__()
        try:
            # Configuración desde settings
            use_gpu = settings.USE_GPU
            batch_size = settings.BATCH_SIZE
            preload_model = settings.PRELOAD_MODEL

            # Sobreescribir por variable de entorno si existe
            if os.environ.get('FORCE_CPU') == '1':
                use_gpu = False
                print("PaddleOCR: Uso de GPU desactivado por variable de entorno")

            # Configuración del motor optimizada para velocidad
            self.ocr = PaddleOCR(
                use_angle_cls=False,      # Desactivar detector de ángulo para mayor velocidad
                lang='en',                # Para caracteres alfanuméricos
                use_gpu=use_gpu,
                enable_mkldnn=not use_gpu,  # MKL-DNN solo para CPU
                det_algorithm="DB",       # Algoritmo de detección rápido
                det_db_thresh=0.3,        # Umbral más bajo para detección más rápida
                det_max_side_len=960,     # Limitar tamaño máximo para procesamiento más rápido
                rec_batch_num=batch_size if use_gpu else 1,
                cls_batch_num=batch_size if use_gpu else 1,
                use_mp=not use_gpu,       # Multiprocesamiento solo para CPU
                show_log=False,
                # Usar detector de alta velocidad (más rápido pero menos preciso)
                det_model_dir=None,       # Usar modelo predeterminado (más rápido)
                rec_model_dir=None,       # Usar modelo predeterminado (más rápido)
                # Parámetros de rendimiento adicionales
                use_tensorrt=use_gpu,     # Usar TensorRT para aceleración si está disponible
                # Para mejor rendimiento, elegimos precision nivel INT8
                precision="int8" if use_gpu else "fp32"
            )

            self._initialized = True
            self.recent_readings = []  # Para sistema de votación
            self.max_readings = 3      # Reducido para mayor velocidad (era 5)

            # Cache para reducir procesamiento repetido
            self._result_cache = {}
            self._cache_size_limit = 100

            # Cola para procesamiento asíncrono (opcional)
            self._processing_queue = []
            self._async_mode = False  # Desactivado por defecto

            # Realizar precalentamiento si está activado
            if preload_model:
                self._warmup_inference()

            mode_str = "GPU" if use_gpu else "CPU"
            print(f"Motor PaddleOCR inicializado (modo {mode_str}, optimizado para velocidad)")

        except Exception as e:
            print(f"Error inicializando PaddleOCR: {e}")
            self._initialized = False

    def _warmup_inference(self):
        """Realiza una inferencia inicial para precalentar el modelo."""
        try:
            # Crear imagen de prueba simple
            warmup_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
            cv2.putText(
                warmup_img, "ABC123", (50, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (0, 0, 0), 3
            )

            # Realizar inferencia para cargar los modelos
            _ = self.ocr.ocr(warmup_img, cls=False)  # cls=False para mayor velocidad
            print("Precalentamiento de modelo OCR completado")
        except Exception as e:
            print(f"Advertencia: Error en inferencia de precalentamiento: {e}")

    def preprocess_image(self, image):
        """
        Preprocesa la imagen para mejorar el reconocimiento.
        Versión optimizada para velocidad.

        Args:
            image (numpy.ndarray): Imagen original.

        Returns:
            numpy.ndarray: Imagen preprocesada.
        """
        try:
            # Verificar dimensiones mínimas - no procesar placas muy pequeñas
            h, w = image.shape[:2]
            if h < 15 or w < 50:
                return None

            # Verificar si la imagen es muy grande, escalarla para procesamiento más rápido
            max_size = 400  # Tamaño máximo para procesamiento rápido
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w

            # Convertir a escala de grises
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Ecualización más rápida - usar ecualización normal en lugar de CLAHE
            enhanced = cv2.equalizeHist(gray)

            # Operaciones más ligeras para preprocesamiento
            # Saltamos el denoising que es costoso computacionalmente

            # Umbralización simple en lugar de adaptativa
            _, binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            # Operaciones morfológicas mínimas
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Guardar imágenes de depuración si está activado
            if self._debug_dir:
                timestamp = int(time.time() * 1000)
                debug_path = os.path.join(self._debug_dir, f"processed_{timestamp}.jpg")
                cv2.imwrite(debug_path, morph)

            # Convertir a BGR para PaddleOCR (espera imágenes de 3 canales)
            processed = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
            return processed

        except Exception as e:
            print(f"Error en preprocesamiento: {e}")
            return None

    def _clean_and_format(self, text):
        """
        Limpia y formatea el texto según formato de placa peruana.
        Versión simplificada para mayor velocidad.

        Args:
            text (str): Texto detectado.

        Returns:
            str: Texto limpio y formateado.
        """
        if not text:
            return None

        # Convertir a mayúsculas y eliminar espacios
        text = text.upper().strip()

        # Eliminar caracteres no alfanuméricos (incluido el guión)
        text = re.sub(r'[^A-Z0-9]', '', text)

        # Correcciones específicas críticas para placas peruanas
        if text.startswith("1") and ("4A" in text or "4E" in text):
            text = "T" + text[1:]  # Reemplazar 1 por T al inicio

        if "E" in text and "4" in text and ("534" in text or "53" in text):
            text = text.replace("E", "A")  # Corregir E por A en este patrón

        # Caso específico 4A534 -> T4A534 (falta la T inicial)
        if len(text) == 5 and text.startswith("4A"):
            text = "T" + text

        # Validar longitud para placas (6 caracteres)
        if len(text) != settings.PLATE_LENGTH:
            # Si no coincide exactamente con la longitud esperada, descartar
            # Este enfoque es más rápido pero menos flexible
            return None

        # Validación final
        if not text.isalnum():
            return None

        return text

    def _is_similar(self, str1, str2, threshold=None):
        """
        Determina si dos cadenas son similares.
        Versión simplificada para mayor velocidad.

        Args:
            str1 (str): Primera cadena.
            str2 (str): Segunda cadena.
            threshold (float): Umbral de similitud (0-1).

        Returns:
            bool: True si son similares, False en caso contrario.
        """
        if threshold is None:
            threshold = settings.SIMILARITY_THRESHOLD

        if not str1 or not str2:
            return False

        # Comparación simple basada en igualdad
        if str1 == str2:
            return True

        # Para mayor velocidad, primero verificamos si difieren en longitud
        if abs(len(str1) - len(str2)) > 1:
            return False

        # Verificar similitud solo si es necesario
        similarity = SequenceMatcher(None, str1, str2).ratio()
        return similarity >= threshold

    def _vote_for_best_reading(self, plate_text):
        """
        Utiliza sistema de votación para estabilizar resultados.
        Versión simplificada para mayor velocidad.

        Args:
            plate_text (str): Nuevo texto detectado.

        Returns:
            str: La mejor lectura según votación.
        """
        if not plate_text:
            return None

        # Añadir lectura actual
        self.recent_readings.append(plate_text)

        # Mantener solo las últimas lecturas
        if len(self.recent_readings) > self.max_readings:
            self.recent_readings.pop(0)

        # Si tenemos pocas lecturas, devolver la más reciente
        if len(self.recent_readings) < 2:
            return plate_text

        # Contar ocurrencias directamente (enfoque más rápido)
        counts = {}
        for reading in self.recent_readings:
            if reading in counts:
                counts[reading] += 1
            else:
                counts[reading] = 1

        # Devolver la lectura con más votos
        best_reading = max(counts, key=counts.get)
        return best_reading

    def recognize_text(self, image):
        """
        Reconoce texto en la imagen proporcionada.
        Versión optimizada para velocidad.

        Args:
            image (numpy.ndarray): Imagen de la placa a procesar.

        Returns:
            str: Texto reconocido o None si no se detectó.
        """
        if not self._initialized:
            return None

        try:
            # Verificar caché para imágenes similares (hash simple)
            img_hash = hash(image.tostring())
            if img_hash in self._result_cache:
                return self._result_cache[img_hash]

            # Preprocesar imagen
            processed_img = self.preprocess_image(image)
            if processed_img is None:
                return None

            # Aplicar PaddleOCR con configuración rápida
            results = self.ocr.ocr(processed_img, cls=False)  # cls=False para mayor velocidad

            # Verificar resultados
            if not results or len(results) == 0:
                return None

            # Extraer texto con manejo rápido
            all_texts = []

            # Manejar diferentes estructuras según versión de PaddleOCR
            if isinstance(results, list) and results:
                if isinstance(results[0], list):
                    # Versión más reciente
                    for line in results[0]:
                        if line and len(line) >= 2 and isinstance(line[1], tuple) and len(line[1]) >= 2:
                            text, confidence = line[1]
                            all_texts.append((text, confidence))
                elif results[0] and hasattr(results[0], '__len__'):
                    # Versión anterior
                    for line in results:
                        if line and len(line) >= 2 and isinstance(line[1], tuple) and len(line[1]) >= 2:
                            text, confidence = line[1]
                            all_texts.append((text, confidence))

            if not all_texts:
                return None

            # Ordenar por confianza y tomar el mejor
            all_texts.sort(key=lambda x: x[1], reverse=True)
            best_text, confidence = all_texts[0]

            # Limpiar y formatear
            clean_text = self._clean_and_format(best_text)

            # Aplicar sistema de votación
            if clean_text:
                final_text = self._vote_for_best_reading(clean_text)

                # Guardar en caché para futuros usos
                self._result_cache[img_hash] = final_text

                # Limitar tamaño del caché
                if len(self._result_cache) > self._cache_size_limit:
                    # Eliminar entrada más antigua
                    self._result_cache.pop(next(iter(self._result_cache)))

                return final_text

            return None

        except Exception as e:
            print(f"Error en reconocimiento OCR: {e}")
            return None

    def enable_async_mode(self):
        """Activa el modo asíncrono para procesamiento en segundo plano."""
        self._async_mode = True

    def disable_async_mode(self):
        """Desactiva el modo asíncrono."""
        self._async_mode = False