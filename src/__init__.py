"""
Proyecto CARID - Sistema de Detección y Reconocimiento de Placas Vehiculares.

Una solución modular para detectar y reconocer placas vehiculares en video,
con soporte para aceleración GPU y múltiples opciones de OCR.
"""

__version__ = "1.0.0"
__author__ = "Proyecto CARID"
__description__ = "Sistema de Detección y Reconocimiento de Placas Vehiculares"

# Configuración global
from src.config import settings

# Exponer funciones principales para facilitar importación
from src.detection import create_detector
from src.ocr import create_ocr_engine