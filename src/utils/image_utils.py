"""
Utilidades para procesamiento de imágenes en el sistema CARID.
"""
import cv2
import numpy as np


def resize_image(image, width=None, height=None, keep_ratio=True):
    """
    Redimensiona una imagen a las dimensiones especificadas.

    Args:
        image (numpy.ndarray): Imagen original.
        width (int, optional): Ancho objetivo.
        height (int, optional): Alto objetivo.
        keep_ratio (bool): Si se debe mantener la relación de aspecto.

    Returns:
        numpy.ndarray: Imagen redimensionada.
    """
    if image is None:
        return None

    # Obtener dimensiones originales
    h, w = image.shape[:2]

    # Si no se especifica ninguna dimensión, devolver la imagen original
    if width is None and height is None:
        return image

    # Calcular nuevas dimensiones manteniendo aspect ratio si es necesario
    if keep_ratio:
        if width is None:
            # Calcular ancho proporcional al alto
            r = height / h
            new_width = int(w * r)
            new_height = height
        elif height is None:
            # Calcular alto proporcional al ancho
            r = width / w
            new_width = width
            new_height = int(h * r)
        else:
            # Mantener relación de aspecto y ajustar a la dimensión limitante
            ratio_w = width / w
            ratio_h = height / h
            ratio = min(ratio_w, ratio_h)
            new_width = int(w * ratio)
            new_height = int(h * ratio)
    else:
        # Usar dimensiones especificadas sin mantener aspect ratio
        new_width = width if width is not None else w
        new_height = height if height is not None else h

    # Redimensionar imagen
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized


def enhance_image(image, enhance_contrast=True, denoise=True):
    """
    Mejora la calidad de una imagen.

    Args:
        image (numpy.ndarray): Imagen original.
        enhance_contrast (bool): Si se debe mejorar el contraste.
        denoise (bool): Si se debe reducir el ruido.

    Returns:
        numpy.ndarray: Imagen mejorada.
    """
    if image is None:
        return None

    # Convertir a escala de grises si la imagen es a color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Mejorar contraste
    if enhance_contrast:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Reducir ruido
    if denoise:
        gray = cv2.fastNlMeansDenoising(gray, h=10)

    return gray


def binarize_image(image, adaptive=True, block_size=11, c=2):
    """
    Binariza una imagen (convierte a blanco y negro).

    Args:
        image (numpy.ndarray): Imagen en escala de grises.
        adaptive (bool): Si se debe usar umbralización adaptativa.
        block_size (int): Tamaño del bloque para umbralización adaptativa.
        c (int): Constante para umbralización adaptativa.

    Returns:
        numpy.ndarray: Imagen binarizada.
    """
    if image is None:
        return None

    # Asegurar que la imagen esté en escala de grises
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Aplicar umbralización
    if adaptive:
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, c
        )
    else:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    return binary


def create_plate_image(text, size=(300, 100), background=(255, 255, 255), text_color=(0, 0, 0)):
    """
    Crea una imagen simulada de una placa.
    Útil para pruebas y depuración.

    Args:
        text (str): Texto de la placa.
        size (tuple): Tamaño de la imagen (ancho, alto).
        background (tuple): Color de fondo (B, G, R).
        text_color (tuple): Color del texto (B, G, R).

    Returns:
        numpy.ndarray: Imagen simulada de placa.
    """
    # Crear imagen en blanco
    img = np.ones((size[1], size[0], 3), dtype=np.uint8)
    img[:] = background

    # Calcular posición del texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (size[0] - text_size[0]) // 2
    text_y = (size[1] + text_size[1]) // 2

    # Añadir texto
    cv2.putText(
        img, text, (text_x, text_y),
        font, font_scale, text_color, thickness
    )

    # Añadir borde
    cv2.rectangle(img, (10, 10), (size[0] - 10, size[1] - 10), text_color, 3)

    return img


def save_debug_image(image, prefix, directory, timestamp=None):
    """
    Guarda una imagen para depuración.

    Args:
        image (numpy.ndarray): Imagen a guardar.
        prefix (str): Prefijo para el nombre del archivo.
        directory (str): Directorio donde guardar la imagen.
        timestamp (int, optional): Timestamp para el nombre. Si es None, se usa el actual.

    Returns:
        str: Ruta completa del archivo guardado.
    """
    import os
    import time

    # Crear directorio si no existe
    os.makedirs(directory, exist_ok=True)

    # Generar timestamp si no se proporciona
    if timestamp is None:
        timestamp = int(time.time() * 1000)

    # Construir ruta del archivo
    file_path = os.path.join(directory, f"{prefix}_{timestamp}.jpg")

    # Guardar imagen
    cv2.imwrite(file_path, image)

    return file_path