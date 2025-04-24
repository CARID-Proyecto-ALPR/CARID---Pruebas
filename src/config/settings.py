"""
Configuración global del sistema CARID.
Centraliza todos los parámetros de configuración del proyecto.
"""
import os
from pathlib import Path

# Rutas base
PROJECT_ROOT = Path(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Rutas específicas
VIDEO_PATH = INPUT_DIR / "video_prueba.MOV"
PLATES_DIR = OUTPUT_DIR / "placas"
DATASET_PATH = OUTPUT_DIR / "dataset.csv"
MODEL_PATH = MODELS_DIR / "best.pt"

# Crear directorios si no existen
PLATES_DIR.mkdir(parents=True, exist_ok=True)

# Resolución de visualización
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# Parámetros de optimización de procesamiento
TARGET_FPS_PLATE = 10    # FPS objetivo cuando se detecta una placa
DETECTION_INTERVAL = 3   # Procesar 1 de cada N frames para detección inicial
PLATE_TRACKING_FRAMES = 30  # Frames a procesar a mayor FPS después de detectar una placa
PLATE_COOLDOWN_TIME = 15.0  # Segundos antes de registrar la misma placa nuevamente

# Configuración de detección
CONFIDENCE_THRESHOLD = 0.6   # Umbral de confianza para detecciones
SINGLE_DETECTION_MODE = True  # Si es True, cada placa única se detecta solo una vez
SIMILARITY_THRESHOLD = 0.85  # Umbral para considerar dos placas como similares (0-1)
SAVE_SIMILARITIES = True  # Guardar registro de variantes similares

# Configuración de GPU/CPU
USE_GPU = True  # Si se debe usar GPU (puede ser sobreescrito en tiempo de ejecución)
BATCH_SIZE = 6   # Tamaño de lote para procesamiento
PRELOAD_MODEL = True  # Si se debe precargar el modelo al inicio

# Formato de placas
PLATE_LENGTH = 6  # Longitud exacta de caracteres para placas peruanas
PLATE_REGEX = r'^[A-Z0-9]{6}$'  # Patrón para validación de placas

# Configuración de visualización
COLOR_DETECTION = (0, 255, 0)  # Verde para recuadros de detección
COLOR_TEXT = (0, 0, 255)       # Rojo para texto
FONT_SCALE = 0.7
LINE_THICKNESS = 2

# Configuración de archivos
SAVE_IMAGES = True  # Si se deben guardar imágenes de las placas detectadas
EXPORT_FORMAT = "csv"  # Formato de exportación de datos (csv, json)

# Configuración de depuración
DEBUG_MODE = False  # Activar/desactivar modo de depuración