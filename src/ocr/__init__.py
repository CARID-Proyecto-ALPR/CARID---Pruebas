"""
Módulo de OCR (Reconocimiento Óptico de Caracteres) para placas.
Proporciona interfaces para diferentes motores de OCR.
"""
from src.ocr.ocr_engine import OCREngine
from src.ocr.paddle_ocr import PaddleOCREngine
from src.ocr.easy_ocr import EasyOCREngine


def create_ocr_engine(engine_type="paddle"):
    """
    Fábrica para crear el motor OCR especificado.

    Args:
        engine_type (str): Tipo de motor OCR a crear.
            Opciones: "paddle" (por defecto), "easyocr"

    Returns:
        OCREngine: Instancia del motor OCR solicitado.

    Raises:
        ValueError: Si el tipo de motor OCR no es soportado.
    """
    if engine_type.lower() == "paddle":
        return PaddleOCREngine()
    elif engine_type.lower() == "easyocr":
        return EasyOCREngine()
    else:
        raise ValueError(f"Tipo de motor OCR no soportado: {engine_type}")