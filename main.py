#!/usr/bin/env python
"""
Proyecto CARID: Sistema de Detección y Reconocimiento de Placas Vehiculares
Punto de entrada principal del sistema.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Importar módulos del proyecto
from src.config import settings
from src.detection import create_detector
from src.ocr import create_ocr_engine
from src.data.plate_handler import PlateHandler
from src.visualization.display import DisplayManager
from src.utils.performance import FPSCounter


def parse_args():
    """Procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Sistema CARID - Detección de Placas Vehiculares")

    parser.add_argument("--video", type=str, help="Ruta al video de entrada")
    parser.add_argument("--fps", type=int, help="FPS objetivo para seguimiento de placas")
    parser.add_argument("--conf", type=float, help="Umbral de confianza para detecciones")
    parser.add_argument("--interval", type=int, help="Intervalo de frames para detección")
    parser.add_argument("--cooldown", type=float, help="Tiempo entre detecciones de la misma placa")
    parser.add_argument("--debug", action="store_true", help="Activa el modo de depuración")
    parser.add_argument("--use-gpu", action="store_true", help="Forzar uso de GPU")
    parser.add_argument("--use-cpu", action="store_true", help="Forzar uso de CPU")
    parser.add_argument("--detector", type=str, default="yolo",
                        choices=["yolo"], help="Detector a utilizar")
    parser.add_argument("--ocr", type=str, default="paddle",
                        choices=["paddle", "easyocr"], help="Motor OCR a utilizar")

    return parser.parse_args()


def update_settings(args):
    """Actualiza la configuración global basada en argumentos."""
    if args.video:
        settings.VIDEO_PATH = args.video

    if args.fps:
        settings.TARGET_FPS_PLATE = args.fps

    if args.conf:
        settings.CONFIDENCE_THRESHOLD = args.conf

    if args.interval:
        settings.DETECTION_INTERVAL = args.interval

    if args.cooldown:
        settings.PLATE_COOLDOWN_TIME = args.cooldown

    # Configuración de GPU/CPU
    if args.use_gpu:
        settings.USE_GPU = True
        print("Uso de GPU forzado por argumento")
    elif args.use_cpu:
        settings.USE_GPU = False
        print("Uso de CPU forzado por argumento")
    elif os.environ.get("FORCE_CPU") == "1":
        settings.USE_GPU = False
        print("Uso de GPU desactivado por variable de entorno")


def main():
    """Función principal del sistema CARID."""
    print("Iniciando sistema CARID - Detección de Placas Vehiculares")

    # Procesar argumentos y actualizar configuración
    args = parse_args()
    update_settings(args)

    # Mostrar configuración actual
    print(f"Configuración:")
    print(f"- Video: {settings.VIDEO_PATH}")
    print(f"- Usar GPU: {'Sí' if settings.USE_GPU else 'No'}")
    print(f"- Detector: {args.detector}")
    print(f"- Motor OCR: {args.ocr}")

    # Inicializar variables para control de tiempo
    start_time = time.time()  # Inicialización al principio para evitar errores
    frame_count = 0

    # Inicializar componentes
    try:
        # Crear detector según configuración
        detector = create_detector(args.detector)
        if not detector.is_initialized():
            print("Error inicializando el detector.")
            return 1

        # Crear motor OCR según configuración
        ocr_engine = create_ocr_engine(args.ocr)
        if not ocr_engine.is_initialized():
            print("Error inicializando el motor OCR.")
            return 1

        # Inicializar manejador de datos
        data_handler = PlateHandler()

        # Inicializar visualización
        display = DisplayManager("CARID - Detección de Placas")

        # Configurar modo depuración si está activado
        if args.debug:
            debug_dir = Path(settings.OUTPUT_DIR) / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            ocr_engine.set_debug_mode(str(debug_dir))
            print(f"Modo depuración activado. Imágenes guardadas en: {debug_dir}")

    except Exception as e:
        print(f"Error inicializando componentes: {e}")
        return 1

    # Abrir video
    import cv2  # Importar aquí para no depender de cv2 en todo el módulo principal
    try:
        video_path = str(settings.VIDEO_PATH)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"No se pudo abrir el video: {video_path}")

        # Información del video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Video cargado: {frame_width}x{frame_height} @ {video_fps} FPS")

        # Contadores y controles
        frame_count = 0
        fps_counter = FPSCounter()
        start_time = time.time()

        # Variables para control adaptativo
        plate_tracking_countdown = 0
        last_frame_time = time.time()

        print("Procesamiento iniciado. Presione ESC para salir.")

        # Bucle principal de procesamiento
        while True:
            # Leer frame
            ret, frame = cap.read()
            if not ret:
                break

            # Control de tiempo y contadores
            frame_count += 1
            current_time = time.time()

            # Determinar si procesar este frame
            process_this_frame = False

            # En modo seguimiento (después de detectar una placa)
            if plate_tracking_countdown > 0:
                elapsed = current_time - last_frame_time
                target_time = 1.0 / settings.TARGET_FPS_PLATE

                if elapsed >= target_time:
                    process_this_frame = True
                    last_frame_time = current_time
                    plate_tracking_countdown -= 1
            else:
                # Modo normal - procesar 1 de cada N frames
                if frame_count % settings.DETECTION_INTERVAL == 0:
                    process_this_frame = True

            # Crear copia para visualización
            display_frame = frame.copy()

            # Procesar frame si corresponde
            if process_this_frame:
                # Detectar placas
                detections = detector.detect(frame)

                # Si hay detecciones, activar modo seguimiento
                if detections:
                    plate_tracking_countdown = settings.PLATE_TRACKING_FRAMES

                # Procesar cada detección
                for detection in detections:
                    # Extraer región de la placa
                    plate_img = detector.extract_region(frame, detection["box"])

                    # Reconocer texto de la placa
                    plate_text = ocr_engine.recognize_text(plate_img)

                    if plate_text:
                        # Verificar si ya hemos visto esta placa recientemente
                        if not data_handler.has_seen_recently(plate_text):
                            # Registrar la placa
                            data_handler.register_plate(
                                frame_count,
                                plate_text,
                                detection["confidence"],
                                detection["box"],
                                plate_img
                            )

                        # Dibujar detección (siempre)
                        display.draw_detection(
                            display_frame,
                            detection["box"],
                            plate_text,
                            detection["confidence"]
                        )
                    else:
                        # Dibujar solo el recuadro si no se reconoció texto
                        display.draw_detection(
                            display_frame,
                            detection["box"],
                            confidence=detection["confidence"]
                        )

            # Actualizar FPS
            fps = fps_counter.update()

            # Determinar modo actual
            current_mode = "SEGUIMIENTO" if plate_tracking_countdown > 0 else "NORMAL"

            # Mostrar información de estado
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame: {frame_count} | FPS: {fps:.1f} | "
                      f"Modo: {current_mode} | "
                      f"Placas: {data_handler.get_plate_count()} | "
                      f"Placas únicas: {data_handler.get_unique_count()}")

            # Actualizar visualización
            display.show_frame(
                display_frame,
                {
                    "mode": current_mode,
                    "fps": fps,
                    "frame": frame_count,
                    "plates": data_handler.get_plate_count(),
                    "unique_plates": data_handler.get_unique_count()
                }
            )

            # Control de velocidad para el modo normal
            if plate_tracking_countdown == 0:
                # Mantener FPS del video original
                elapsed = time.time() - current_time
                target_time = 1.0 / video_fps
                sleep_time = max(0, target_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Salir con ESC
            if cv2.waitKey(1) == 27:
                break

    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        return 1

    finally:
        # Guardar datos y limpiar
        if hasattr(locals(), 'data_handler') and data_handler.get_plate_count() > 0:
            data_handler.save_data()

        # Estadísticas finales
        total_time = time.time() - start_time
        print("\nEstadísticas finales:")
        print(f"Tiempo total: {total_time:.2f} segundos")
        print(f"Frames procesados: {frame_count}")
        print(f"FPS promedio: {frame_count / total_time:.2f}")

        if 'cap' in locals():
            cap.release()

        if 'display' in locals():
            display.cleanup()

        print("Proceso finalizado.")

    return 0


if __name__ == "__main__":
    sys.exit(main())