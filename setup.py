#!/usr/bin/env python
"""
Script de configuración inicial para el Proyecto CARID.
Verifica dependencias, crea estructura de directorios y configura el entorno.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Verifica que la versión de Python sea compatible."""
    if sys.version_info < (3, 8):
        print("Error: Se requiere Python 3.8 o superior.")
        return False

    print(f"✓ Python {platform.python_version()} detectado")
    return True


def create_directory_structure():
    """Crea la estructura de directorios del proyecto."""
    print("\nCreando estructura de directorios...")

    # Directorios principales
    directories = [
        "src/config",
        "src/detection",
        "src/ocr",
        "src/data",
        "src/visualization",
        "src/utils",
        "src/api",
        "tests",
        "tools",
        "input",
        "output/placas",
        "output/debug",
        "models"
    ]

    # Crear cada directorio
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {path}")

    # Crear archivos __init__.py en cada directorio de src
    for directory in [d for d in directories if d.startswith("src/")]:
        init_file = Path(directory) / "__init__.py"
        if not init_file.exists():
            with open(init_file, 'w') as f:
                f.write(f'"""Módulo {directory.split("/")[-1]} del sistema CARID."""\n')
            print(f"  ✓ {init_file}")

    # Crear __init__.py principal
    main_init = Path("src") / "__init__.py"
    if not main_init.exists():
        with open(main_init, 'w') as f:
            f.write(
                '"""Sistema CARID - Detección y Reconocimiento de Placas Vehiculares."""\n\n__version__ = "1.0.0"\n')
        print(f"  ✓ {main_init}")

    print("✓ Estructura de directorios creada correctamente")
    return True


def install_dependencies(gpu_mode=True):
    """
    Instala las dependencias del proyecto.

    Args:
        gpu_mode (bool): Si se debe instalar con soporte GPU.
    """
    print("\nInstalando dependencias...")

    try:
        # Actualizar pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✓ pip actualizado")

        # Instalar dependencias comunes
        common_deps = [
            "numpy==1.23.5",
            "opencv-python==4.8.0.76",
            "pandas==2.0.3",
            "matplotlib==3.7.2",
            "ultralytics==8.0.196",  # Para YOLOv8
            "torch==2.0.1",
            "torchvision==0.15.2",
            "tqdm==4.65.0",
            "pillow==10.0.1",
            "pyyaml==6.0.1",
            "scipy==1.11.3"
        ]

        # Instalar desde archivo requirements si existe
        if Path("requirements.txt").exists():
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✓ Dependencias instaladas desde requirements.txt")
        else:
            # Instalar dependencias comunes (primero numpy para evitar conflictos)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.23.5"])
            print("✓ NumPy instalado")

            for dep in common_deps[1:]:  # Saltar numpy que ya instalamos
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

            # Instalar PaddleOCR según modo
            if gpu_mode:
                # Versión GPU
                print("Instalando PaddlePaddle GPU...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "paddlepaddle-gpu==2.6.1.post120",
                    "-f", "https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html"
                ])
                print("✓ PaddlePaddle GPU instalado")
            else:
                # Versión CPU
                subprocess.check_call([sys.executable, "-m", "pip", "install", "paddlepaddle==2.6.0"])
                print("✓ PaddlePaddle CPU instalado")

            # Instalar PaddleOCR
            subprocess.check_call([sys.executable, "-m", "pip", "install", "paddleocr==2.7.0.3"])
            print("✓ PaddleOCR instalado")

            # Instalar EasyOCR (alternativa)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "easyocr==1.7.0"])
            print("✓ EasyOCR instalado")

        print("✓ Todas las dependencias instaladas correctamente")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error instalando dependencias: {e}")
        return False


def check_cuda():
    """
    Verifica la disponibilidad de CUDA y muestra información.

    Returns:
        bool: True si CUDA está disponible, False en caso contrario.
    """
    print("\nVerificando disponibilidad de GPU/CUDA...")
    cuda_available = False

    try:
        # Verificar con PaddlePaddle (versión actualizada)
        try:
            import paddle
            paddle_cuda = paddle.is_compiled_with_cuda()

            if paddle_cuda:
                print("✓ PaddlePaddle compilado con soporte CUDA")
                try:
                    # Método moderno para obtener dispositivos en PaddlePaddle
                    device_count = len(paddle.static.cuda_places())
                    print(f"✓ GPUs disponibles para PaddlePaddle: {device_count}")
                    cuda_available = device_count > 0
                except Exception as e:
                    print(f"✗ No se pudieron enumerar GPUs con PaddlePaddle: {str(e)}")
            else:
                print("✗ PaddlePaddle no tiene soporte CUDA")
        except ImportError:
            print("✗ No se pudo importar PaddlePaddle")
        except Exception as paddle_e:
            print(f"✗ Error al verificar CUDA con PaddlePaddle: {str(paddle_e)}")

    except Exception as e:
        print(f"Error general verificando CUDA con PaddlePaddle: {e}")

    # Verificar con PyTorch (alternativa)
    try:
        print("Intentando verificar CUDA con PyTorch...")
        import torch
        torch_cuda = torch.cuda.is_available()

        if torch_cuda:
            print("✓ PyTorch detecta CUDA")
            try:
                device_count = torch.cuda.device_count()
                print(f"✓ GPUs disponibles para PyTorch: {device_count}")

                for i in range(device_count):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

                cuda_available = cuda_available or (device_count > 0)
            except Exception as e:
                print(f"✗ Error al obtener información de GPUs con PyTorch: {str(e)}")
                print("  Este error no afectará el funcionamiento del sistema.")
        else:
            print("✗ PyTorch no detecta CUDA")

    except ImportError:
        print("✗ No se pudo importar PyTorch")
    except Exception as e:
        print(f"✗ Error al verificar CUDA con PyTorch: {str(e)}")
        print("  Este error no afectará el funcionamiento del sistema.")

    # Verificar nvidia-smi (solo información adicional)
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if result.returncode == 0:
            print("✓ nvidia-smi ejecutado correctamente")
            cuda_available = True
        else:
            print("✗ nvidia-smi devolvió un error")
    except Exception:
        print("✗ No se pudo ejecutar nvidia-smi")

    if cuda_available:
        print("\n✓ CUDA está disponible y configurado correctamente")
    else:
        print("\n✗ CUDA no está configurado correctamente o no está disponible")
        print("  El sistema funcionará en modo CPU")

    return cuda_available


def check_model():
    """
    Verifica si el modelo YOLO está presente.

    Returns:
        bool: True si el modelo está presente, False en caso contrario.
    """
    print("\nVerificando modelo YOLO...")

    model_path = Path("models") / "best.pt"

    if model_path.exists():
        print(f"✓ Modelo encontrado: {model_path}")
        return True
    else:
        print(f"✗ Modelo no encontrado: {model_path}")
        print("  Coloca tu modelo 'best.pt' en la carpeta 'models/'")
        return False


def create_test_files():
    """
    Crea archivos de prueba y utilidades.

    Returns:
        bool: True si se crearon correctamente, False en caso contrario.
    """
    print("\nCreando archivos de prueba y utilidades...")

    try:
        # Crear archivo de tests básico
        test_file = Path("tests") / "test_detection.py"
        if not test_file.exists():
            with open(test_file, 'w') as f:
                f.write("""
import unittest
import cv2
import numpy as np
from src.detection import create_detector

class TestDetection(unittest.TestCase):
    def test_detector_creation(self):
        detector = create_detector("yolo")
        self.assertIsNotNone(detector)

    def test_empty_detection(self):
        detector = create_detector("yolo")
        if not detector.is_initialized():
            self.skipTest("Detector no inicializado")

        # Crear imagen vacía para test
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(empty_img)

        # No debería detectar nada en una imagen vacía
        self.assertEqual(len(detections), 0)

if __name__ == '__main__':
    unittest.main()
""")
            print(f"✓ Creado: {test_file}")

        # Crear README con información básica
        readme_file = Path("README.md")
        if not readme_file.exists():
            with open(readme_file, 'w') as f:
                f.write("""# Proyecto CARID - Sistema de Detección de Placas Vehiculares

Sistema modular para la detección y reconocimiento de placas vehiculares en video, utilizando YOLOv8 y reconocimiento óptico de caracteres.

## Características

- Detección de placas vehiculares con YOLOv8
- Reconocimiento óptico de caracteres optimizado para placas
- Soporte para GPU/CPU con selección automática
- Arquitectura modular para facilitar extensiones

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tuusuario/proyecto-carid.git
cd proyecto-carid

# Configurar el entorno
python setup.py

# Colocar el modelo YOLO en la carpeta models/
# (El modelo debe llamarse 'best.pt')
```

## Uso

```bash
# Ejecución con selección automática GPU/CPU
python run_carid.py

# Forzar uso de GPU
python run_carid.py --force-gpu

# Forzar uso de CPU
python run_carid.py --force-cpu

# Modo de depuración
python run_carid.py --debug
```

## Estructura del Proyecto

El proyecto sigue una arquitectura modular:

- `src/detection/`: Módulo de detección de placas
- `src/ocr/`: Módulo de reconocimiento óptico de caracteres
- `src/data/`: Módulo de gestión de datos
- `src/visualization/`: Módulo de visualización
- `src/utils/`: Utilidades generales

## Licencia

Este proyecto está bajo la Licencia MIT.
""")
            print(f"✓ Creado: {readme_file}")

        print("✓ Archivos de prueba y utilidades creados correctamente")
        return True

    except Exception as e:
        print(f"Error creando archivos de prueba: {e}")
        return False


def show_final_message(cuda_available):
    """Muestra mensaje final con instrucciones."""
    print("\n" + "=" * 60)
    print("¡CONFIGURACIÓN COMPLETADA!")
    print("=" * 60)

    print("\nEl Proyecto CARID está configurado y listo para ser utilizado.")

    if cuda_available:
        print("\nGPU DETECTADA: El sistema puede utilizar aceleración GPU.")
        print("Para un rendimiento óptimo, ejecute:")
        print("  python run_carid.py")
    else:
        print("\nNo se detectó GPU compatible. El sistema funcionará en modo CPU.")
        print("Para ejecutar el sistema:")
        print("  python run_carid.py --force-cpu")

    print("\nOpciones adicionales:")
    print("  --debug          Activa el modo de depuración")
    print("  --video PATH     Especifica un video de entrada alternativo")
    print("  --ocr TYPE       Selecciona el motor OCR ('paddle' o 'easyocr')")

    print("\nAsegúrese de colocar el modelo YOLOv8 (best.pt) en la carpeta 'models/'")
    print("y un video de prueba en la carpeta 'input/video_prueba.MOV' o especificar")
    print("la ruta con --video")

    print("\n¡Listo para comenzar a detectar placas!")
    print("=" * 60)


def main():
    """Función principal."""
    print("=" * 60)
    print("CONFIGURACIÓN DEL PROYECTO CARID")
    print("=" * 60)

    # Verificar versión de Python
    if not check_python_version():
        return 1

    # Crear estructura de directorios
    if not create_directory_structure():
        return 1

    # Verificar disponibilidad de CUDA
    cuda_available = check_cuda()

    # Instalar dependencias según disponibilidad de GPU
    if not install_dependencies(gpu_mode=cuda_available):
        print("Advertencia: No se pudieron instalar todas las dependencias.")

    # Verificar modelo
    check_model()

    # Crear archivos de prueba
    create_test_files()

    # Mostrar mensaje final
    show_final_message(cuda_available)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nConfiguración interrumpida por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError inesperado: {e}")
        sys.exit(1)