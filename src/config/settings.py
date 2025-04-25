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
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 800

# Parámetros de optimización de procesamiento
TARGET_FPS_PLATE = 20    # FPS objetivo cuando se detecta una placa (aumentado para mayor fluidez)
DETECTION_INTERVAL = 3   # Procesar 1 de cada N frames para detección inicial
PLATE_TRACKING_FRAMES = 20  # Frames a procesar a mayor FPS después de detectar una placa
PLATE_COOLDOWN_TIME = 10.0  # Segundos antes de registrar la misma placa nuevamente (reducido)

# Configuración de detección
CONFIDENCE_THRESHOLD = 0.60   # Umbral de confianza para detecciones
SINGLE_DETECTION_MODE = True  # Si es True, cada placa única se detecta solo una vez
SIMILARITY_THRESHOLD = 0.85  # Umbral para considerar dos placas como similares (0-1)
SAVE_SIMILARITIES = False  # Guardar registro de variantes similares

# Configuración de GPU/CPU
USE_GPU = True  # Si se debe usar GPU (puede ser sobreescrito en tiempo de ejecución)
BATCH_SIZE = 8   # Tamaño de lote para procesamiento (aumentado para mejor rendimiento)
PRELOAD_MODEL = True  # Si se debe precargar el modelo al inicio

# Optimizaciones de rendimiento
MAX_PLATE_SIZE = 400  # Tamaño máximo para redimensionar placas durante OCR
SKIP_FRAMES_IF_LAGGING = True  # Saltar frames si el procesamiento está retrasado
PERFORMANCE_MODE = "precision"  # Modos: "precision", "balanced", "speed"
MULTITHREADED_OCR = False  # Usar procesamiento en hilos para OCR (experimental)

# Formato de placas
PLATE_LENGTH = 6  # Longitud exacta de caracteres para placas peruanas
PLATE_REGEX = r'^[A-Z0-9]{6}$'  # Patrón para validación de placas

# Configuración de visualización
COLOR_DETECTION = (0, 255, 0)  # Verde para recuadros de detección
COLOR_TEXT = (0, 0, 255)       # Rojo para texto
FONT_SCALE = 1.5
LINE_THICKNESS = 2

# Configuración de archivos
SAVE_IMAGES = True  # Si se deben guardar imágenes de las placas

# Configuración de depuración
DEBUG_MODE = False  # Activar/desactivar modo de depuración