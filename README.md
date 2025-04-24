# Proyecto CARID - Sistema de Detección y Reconocimiento de Placas Vehiculares

Un sistema modular y optimizado para la detección y reconocimiento de placas vehiculares en video, con soporte para GPU y múltiples opciones de OCR.

## Características

- **Detección precisa**: Utiliza YOLOv8 para identificar placas vehiculares en video
- **OCR optimizado**: Reconocimiento de texto especializado para placas peruanas
- **Arquitectura modular**: Diseño flexible y extensible con componentes independientes
- **Aceleración GPU/CPU**: Selección automática del mejor modo para tu hardware
- **Filtrado inteligente**: Detecta placas únicas evitando duplicados
- **Visualización en tiempo real**: Interfaz visual para monitoreo de detecciones

## Requisitos

- Python 3.8+
- NVIDIA GPU con CUDA (opcional pero recomendado)
- Modelo YOLOv8 pre-entrenado para detección de placas
- Dependencias listadas en `requirements.txt`

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/proyecto-carid.git
cd proyecto-carid

# 2. Ejecutar script de configuración
python setup.py

# 3. Colocar modelo YOLO en la carpeta correspondiente
# (Asegúrate que el archivo se llame 'best.pt' y esté en la carpeta 'models/')
```

## Uso Básico

```bash
# Ejecución con selección automática de GPU/CPU
python run_carid.py

# Especificar un video de entrada
python run_carid.py --video ruta/al/video.mp4

# Activar modo de depuración (guarda imágenes de procesamiento)
python run_carid.py --debug
```

## Opciones Avanzadas

```bash
# Forzar uso de GPU o CPU
python run_carid.py --force-gpu
python run_carid.py --force-cpu

# Establecer tiempo de enfriamiento entre detecciones (en segundos)
python run_carid.py --cooldown 10.0

# Seleccionar motor OCR
python run_carid.py --ocr paddle  # PaddleOCR (por defecto)
python run_carid.py --ocr easyocr  # EasyOCR (alternativa)

# Ajustar umbral de confianza para detecciones
python run_carid.py --conf 0.5
```

## Estructura del Proyecto

El proyecto sigue una arquitectura modular:

```
Proyecto CARID/
├── src/                     # Código fuente principal
│   ├── config/              # Configuración centralizada
│   ├── detection/           # Detector de placas (YOLO)
│   ├── ocr/                 # Reconocimiento de texto
│   ├── data/                # Manejo de datos detectados
│   ├── visualization/       # Interfaz visual
│   └── utils/               # Utilidades generales
│
├── models/                  # Modelo YOLOv8 pre-entrenado
├── input/                   # Videos de entrada
└── output/                  # Resultados y detecciones
    ├── placas/              # Imágenes de placas detectadas
    └── debug/               # Imágenes de depuración (opcional)
```

## Componentes Principales

1. **Detector (src/detection/)**: Implementa la detección de placas utilizando YOLOv8.

2. **OCR (src/ocr/)**: Proporciona reconocimiento de texto con dos implementaciones:
   - **PaddleOCR**: Motor principal, optimizado para velocidad y precisión
   - **EasyOCR**: Alternativa robusta para mayor compatibilidad

3. **Manejador de datos (src/data/)**: Gestiona el almacenamiento y procesamiento de los datos de placas detectadas, con filtrado de duplicados.

4. **Visualización (src/visualization/)**: Interfaz visual para monitorear detecciones en tiempo real.

## Optimización GPU vs CPU

El sistema evalúa automáticamente si GPU o CPU proporciona mejor rendimiento en tu hardware específico:

- Si tienes una GPU compatible con CUDA, puede ofrecer aceleración significativa
- En algunos casos, CPU puede ser más eficiente para procesamiento de imágenes pequeñas
- Puedes forzar un modo específico usando `--force-gpu` o `--force-cpu`

## Resultados y Exportación

Los resultados se almacenan en la carpeta `output/`:

- `output/placas/`: Imágenes recortadas de las placas detectadas
- `output/dataset.csv`: Datos estructurados de todas las detecciones
- Cuando el modo debug está activado, se generan imágenes adicionales en `output/debug/`

## Solución de Problemas

- **Error de CUDA**: Si encuentras errores relacionados con GPU, intenta usar `--force-cpu`
- **Modelo no encontrado**: Asegúrate de colocar el archivo `best.pt` en la carpeta `models/`
- **Baja precisión de OCR**: Prueba con modo debug para ver las imágenes preprocesadas

## Licencia

Este proyecto está bajo la Licencia MIT.

## Contacto

Para reporte de errores o sugerencias, crea un issue en el repositorio o contacta a los desarrolladores.
