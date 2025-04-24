#!/usr/bin/env python
"""
Script principal para ejecutar el sistema CARID con selección automática de GPU/CPU.
Determina el mejor modo de ejecución basado en pruebas de rendimiento.
"""
import os
import sys
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path

# Importar configuración
from src.config import settings


def parse_arguments():
    """Procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Sistema CARID - Detección de Placas Vehiculares")
    
    # Argumentos generales
    parser.add_argument("--video", type=str, help="Ruta al video de entrada")
    parser.add_argument("--fps", type=int, help="FPS objetivo para seguimiento de placas")
    parser.add_argument("--conf", type=float, help="Umbral de confianza para detecciones")
    parser.add_argument("--interval", type=int, help="Intervalo de frames para detección")
    parser.add_argument("--cooldown", type=float, help="Tiempo entre detecciones de la misma placa")
    
    # Argumentos de ejecución
    parser.add_argument("--debug", action="store_true", help="Activa el modo de depuración")
    parser.add_argument("--force-gpu", action="store_true", help="Forzar uso de GPU")
    parser.add_argument("--force-cpu", action="store_true", help="Forzar uso de CPU")
    parser.add_argument("--skip-test", action="store_true", help="Omitir prueba de rendimiento")
    
    # Argumentos de componentes
    parser.add_argument("--detector", type=str, default="yolo", 
                      choices=["yolo"], help="Detector a utilizar")
    parser.add_argument("--ocr", type=str, default="paddle", 
                      choices=["paddle", "easyocr"], help="Motor OCR a utilizar")
    
    return parser.parse_args()


def test_gpu_performance():
    """
    Realiza una prueba para determinar si la GPU proporciona beneficio.
    
    Returns:
        bool: True si es mejor usar GPU, False si es mejor usar CPU.
    """
    print("Realizando prueba de rendimiento para determinar modo óptimo...")
    
    try:
        # Crear directorio temporal si no existe
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Test con imágenes grandes para beneficiar a GPU
        from src.utils.image_utils import create_plate_image
        import cv2
        
        # Verificar si paddle está disponible con GPU
        try:
            import paddle
            paddle_gpu = paddle.is_compiled_with_cuda()
            if not paddle_gpu:
                print("PaddlePaddle no está compilado con soporte CUDA.")
                print("Se recomienda usar CPU para mejor compatibilidad.")
                return False
        except ImportError:
            print("PaddlePaddle no está instalado correctamente.")
            return False
        
        # Crear OCR para prueba
        from paddleocr import PaddleOCR
        
        # Test con GPU
        print("Probando OCR con GPU...")
        ocr_gpu = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, show_log=False)
        
        # Crear conjunto de prueba (placas grandes para beneficiar GPU)
        test_plates = []
        for i in range(5):
            plate_text = f"TEST{i}123"
            img = create_plate_image(plate_text, size=(600, 200))
            test_plates.append(img)
        
        # Medir tiempo GPU
        start_time = time.time()
        for img in test_plates:
            result_gpu = ocr_gpu.ocr(img)
        gpu_time = time.time() - start_time
        
        # Test con CPU
        print("Probando OCR con CPU...")
        ocr_cpu = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
        
        start_time = time.time()
        for img in test_plates:
            result_cpu = ocr_cpu.ocr(img)
        cpu_time = time.time() - start_time
        
        # Comparar resultados
        print(f"Tiempo GPU: {gpu_time:.4f}s")
        print(f"Tiempo CPU: {cpu_time:.4f}s")
        
        # Calcular aceleración
        if gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"Aceleración GPU: {speedup:.2f}x")
            
            # Determinar el mejor modo (GPU debe ser al menos 20% más rápida)
            if speedup > 1.2:
                print("La GPU proporciona mejor rendimiento. Usando GPU.")
                return True
            else:
                print("La CPU proporciona similar o mejor rendimiento. Usando CPU.")
                return False
        else:
            print("Error en prueba de GPU. Usando CPU por seguridad.")
            return False
            
    except Exception as e:
        print(f"Error durante prueba de rendimiento: {e}")
        print("Por seguridad, se utilizará CPU.")
        return False


def run_main_script(args, use_gpu):
    """
    Ejecuta el script principal con la configuración determinada.
    
    Args:
        args (argparse.Namespace): Argumentos de línea de comandos.
        use_gpu (bool): Si se debe usar GPU.
    """
    # Construir comandos para llamar a main.py
    cmd = [sys.executable, "main.py"]
    
    # Añadir argumentos pasados a este script
    if args.video:
        cmd.extend(["--video", args.video])
    
    if args.fps:
        cmd.extend(["--fps", str(args.fps)])
    
    if args.conf:
        cmd.extend(["--conf", str(args.conf)])
    
    if args.interval:
        cmd.extend(["--interval", str(args.interval)])
    
    if args.cooldown:
        cmd.extend(["--cooldown", str(args.cooldown)])
    
    if args.debug:
        cmd.append("--debug")
    
    # Añadir selección de componentes
    cmd.extend(["--detector", args.detector])
    cmd.extend(["--ocr", args.ocr])
    
    # Añadir selección de GPU/CPU
    if use_gpu:
        cmd.append("--use-gpu")
    else:
        cmd.append("--use-cpu")
    
    # Establecer variables de entorno según el modo
    env = os.environ.copy()
    if not use_gpu:
        env["FORCE_CPU"] = "1"
    
    # Mostrar comando a ejecutar
    print(f"Ejecutando comando: {' '.join(cmd)}")
    print(f"Modo: {'GPU' if use_gpu else 'CPU'}")
    
    # Ejecutar el script principal
    subprocess.run(cmd, env=env)


def main():
    """Función principal."""
    print("=== SISTEMA CARID - DETECCIÓN DE PLACAS VEHICULARES ===\n")
    
    # Procesar argumentos
    args = parse_arguments()
    
    # Determinar si usar GPU o CPU
    use_gpu = True  # Valor predeterminado
    
    if args.force_gpu:
        use_gpu = True
        print("Modo forzado a GPU por argumento --force-gpu")
    elif args.force_cpu:
        use_gpu = False
        print("Modo forzado a CPU por argumento --force-cpu")
    elif not args.skip_test:
        # Realizar prueba de rendimiento automática
        use_gpu = test_gpu_performance()
    else:
        print("Omitiendo prueba de rendimiento. Usando modo predeterminado (GPU).")
    
    # Ejecutar script principal
    run_main_script(args, use_gpu)
    
    print("\n=== Ejecución finalizada ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError inesperado: {e}")
        sys.exit(1)