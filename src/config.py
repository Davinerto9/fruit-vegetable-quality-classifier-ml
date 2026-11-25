"""
Configuración central del proyecto: rutas, semilla, clases y parámetros.

Fase CRISP-DM: todas (sobre todo Comprensión del negocio y Preparación de datos)
Uso: from src import config

Este es el ÚNICO lugar donde se definen rutas y parámetros. Los demás módulos,
los notebooks y la app los importan de aquí en vez de escribirlos a mano. Así:
- un cambio (por ejemplo, el tamaño de imagen) se hace en una sola línea y
  llega igual al entrenamiento, la evaluación y la app;
- se evitan los "valores mágicos" repetidos que se desincronizan;
- los experimentos son reproducibles, porque todos usan la misma semilla y
  las mismas particiones.

Constantes:
- ROOT: raíz del repositorio. Las demás rutas se construyen a partir de ella
  para que el código funcione en cualquier computador, sin rutas absolutas.
- DATA_RAW, DATA_OWN, DATA_PROCESSED: datos externos, fotos propias del equipo
  y archivos derivados.
- RYANDPARK_DIR, AGRIFRESHNET_DIR: carpeta de cada dataset externo.
- OWN_LABELS, INDEX_CSV, CROPS_DIR: etiquetas propias, índice de todas las
  imágenes y recortes generados por la segmentación.
- CHECKPOINTS, RESULTS, FIGURES, LOGS: modelos entrenados, métricas,
  gráficas para el informe y registros del entrenamiento.
- IMAGE_EXTENSIONS: extensiones de imagen que se leen (en minúscula).
- SEED: semilla para todo proceso aleatorio (particiones, modelos, CNN).
- IMG_SIZE: tamaño (alto, ancho) de entrada de los modelos.
- QUALITY_CLASSES: categorías de calidad que predicen los modelos.
- RYANDPARK_QUALITY, AGRIFRESHNET_QUALITY: traducción de las carpetas de cada
  dataset a QUALITY_CLASSES.
- PRODUCT_NAMES, PRODUCTS: nombre canónico de cada producto y lista de
  productos con datos de entrenamiento (el alcance del modelo).
- AUG_PREFIX: patrón de las copias aumentadas de AgriFreshNET, para agruparlas
  con su foto original al particionar.
- CAPTURE_STAMP, CAPTURE_NUMBER_BLOCK: reglas para agrupar las ráfagas de ryandpark
  tomadas en el mismo minuto o con números consecutivos.
- SEG_*, CROP_MARGIN, GRAY_TOL: parámetros de la segmentación.
- STATS_SIDE: tamaño de referencia para las estadísticas de calidad de imagen.
- SIZE_THRESHOLDS: intervalos del diámetro normalizado para cada tamaño.
- TEST_SIZE, VAL_SIZE: proporciones de las particiones de prueba y validación.
"""
from pathlib import Path

# --- Rutas -------------------------------------------------------------------

# Raíz del repositorio (este archivo está en src/config.py).
ROOT = Path(__file__).resolve().parents[1]

# Datasets externos descargados (no se suben a git).
DATA_RAW = ROOT / "data" / "raw"
# Fotos tomadas por el equipo y su archivo labels.csv.
DATA_OWN = ROOT / "data" / "own"
# Archivos derivados: index.csv, splits.csv, features.csv y recortes.
DATA_PROCESSED = ROOT / "data" / "processed"

# Kaggle "Fruit Quality Classification" (ryandpark), dataset de referencia del curso.
RYANDPARK_DIR = DATA_RAW / "ryandpark"
# AgriFreshNET (Mendeley Data, DOI 10.17632/42m5tb7yv9.1), aporta la clase intermedia.
AGRIFRESHNET_DIR = DATA_RAW / "agrifreshnet" / "Processed Data"

# Etiquetas de las fotos propias (una fila por foto).
OWN_LABELS = DATA_OWN / "labels.csv"
# Contornos revisados de fotos propias que requieren corrección manual.
OWN_CONTOURS = DATA_OWN / "contours.json"
# Grupos apartados de las particiones tras la revisión visual de los recortes.
SEGMENTATION_EXCLUSIONS = ROOT / "data" / "annotations" / "segmentation_exclusions.csv"
# Índice de imágenes con sus etiquetas disponibles y grupos para las particiones.
INDEX_CSV = DATA_PROCESSED / "index.csv"
# Recortes cuadrados de la fruta generados por la segmentación.
CROPS_DIR = DATA_PROCESSED / "crops"

# Modelos entrenados (.joblib para ML clásico, .keras para la CNN).
CHECKPOINTS = ROOT / "experiments" / "checkpoints"
# Métricas e hiperparámetros de cada experimento.
RESULTS = ROOT / "experiments" / "results"
# Gráficas listas para el informe IEEE.
FIGURES = ROOT / "experiments" / "figures"
# Registros del entrenamiento (historial por época).
LOGS = ROOT / "experiments" / "logs"

# Extensiones de imagen aceptadas; se comparan en minúscula (hay .jpg y .JPG).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# --- Reproducibilidad y entrada de los modelos -------------------------------

# Semilla única para todo lo aleatorio, para reproducir los resultados.
SEED = 42

# Tamaño (alto, ancho) en píxeles al que se redimensionan los recortes.
IMG_SIZE = (128, 128)

# --- Clases y productos ------------------------------------------------------

# Categorías de calidad (de mejor a peor).
QUALITY_CLASSES = ["good", "regular", "bad"]

# Carpetas de ryandpark con etiqueta de calidad aplicable al inventario.
RYANDPARK_QUALITY = {
    "Good Quality_Fruits": "good",
    "Bad Quality_Fruits": "bad",
}
# Se inventaría sin asignar calidad a la foto completa.
RYANDPARK_MIXED_DIR = "Mixed Qualit_Fruits"

# Prefijo de las carpetas de AgriFreshNET (en minúscula y con "_" como espacio) -> calidad.
# "semi fresh" va antes que "fresh" para que se compare primero el prefijo más largo.
AGRIFRESHNET_QUALITY = {
    "semi fresh": "regular",
    "fresh": "good",
    "rotten": "bad",
}

# Nombre del producto en las carpetas (en minúscula) -> nombre canónico.
PRODUCT_NAMES = {
    "apple": "apple",
    "banana": "banana",
    "guava": "guava",
    "lime": "lime",
    "lemon": "lime",  # en "Mixed" de ryandpark aparece como Lemon
    "orange": "orange",
    "pomegranate": "pomegranate",
    "bittermelon": "bitter_melon",
    "cucumber": "cucumber",
    "eggplant": "eggplant",
    "papaya": "papaya",
    "pineapple": "pineapple",
    "tomato": "tomato",
}

# Productos con datos de entrenamiento: el modelo solo es confiable para estos.
PRODUCTS = sorted(set(PRODUCT_NAMES.values()))

# Prefijo de las copias aumentadas de AgriFreshNET ("aug_12_IMG_x.jpg" es copia de "IMG_x.jpg").
AUG_PREFIX = r"^aug_\d+_"

# Fecha y hora en los nombres de ryandpark ("IMG_20190902_111542_1", "20190820_153401_25723",
# "IMG20200728182100"): se captura hasta el minuto para agrupar las ráfagas de una misma escena.
CAPTURE_STAMP = r"(20\d{6})_?(\d{4})\d{2}"
# Las fotos de ryandpark sin fecha ("IMG_2427") se agrupan en bloques de números consecutivos.
CAPTURE_NUMBER_BLOCK = 10

# --- Segmentación ------------------------------------------------------------

# Lado mayor (px) al que se reduce la imagen para segmentar; la máscara se reescala al original.
SEG_WORK_SIZE = 320
# Fracción mínima de la imagen que debe ocupar la fruta; menos indica máscara dudosa.
SEG_MIN_AREA = 0.02
# Fracción máxima de la imagen que puede ocupar la fruta; más indica que se tomó el fondo.
SEG_MAX_AREA = 0.90
# Lado mayor (px) al que se reduce la foto para medir brillo, contraste y nitidez.
STATS_SIDE = 512
# Margen alrededor de la fruta en el recorte cuadrado, como fracción de su lado mayor.
CROP_MARGIN = 0.10
# Promedio máximo de |a-128| + |b-128| (Lab) para considerar una foto en escala de grises.
GRAY_TOL = 4.0

# --- Tamaño y particiones ----------------------------------------------------

# Intervalos [mín, máx) del diámetro normalizado (diámetro equivalente / lado mayor de la imagen).
# Calibrados con las 35 fotos propias de una pieza con tamaño anotado: coinciden con 34 (97 %).
# El corte de "small" se apoya en solo dos ejemplos y no hay ejemplos "large", así que el corte
# superior es una extrapolación. d_norm depende de la distancia de la cámara: estos umbrales
# valen para el protocolo de captura del equipo, no son centímetros. Ver
# docs/supporting_docs/mathematical_and_theoretical_foundations.md §4.
SIZE_THRESHOLDS = {
    "small": (0.0, 0.33),
    "medium": (0.33, 0.55),
    "large": (0.55, 1.01),
}

# --- Aplicación ----------------------------------------------------------------

# Checkpoints que usa la app (en CHECKPOINTS), entrenados sobre el split v3 (sesiones de
# AgriFreshNET completas por partición; ver docs/plan_simple.md). Si split_v3 --choose elige
# otro nivel que "week", cambia el sufijo.
APP_RF_CHECKPOINT = "random_forest_v3_week.joblib"          # calidad (train_classic)
APP_CNN_CHECKPOINT = "cnn_rgb_v3_week.keras"                # calidad (train_cnn --run-name v3_week)
APP_PRODUCT_CHECKPOINT = "product_rf_v3_week.joblib"        # fruta (train_classic)
# Probabilidad mínima para afirmar qué fruta es; por debajo, la app dice que no la reconoce.
APP_MIN_PRODUCT_CONFIDENCE = 0.5
# Modelo que decide la clase en la app: "ensemble" (voto suave RF + CNN), "random_forest" o "cnn".
# La elección final la fija el equipo antes de abrir test; si falta un modelo, la app usa el otro.
APP_MODEL = "ensemble"
# Peso de la CNN en el voto suave p = (1 - w) * p_RF + w * p_CNN. Fijado en 0,5 antes de ver
# resultados (experiments/diagnostics/eval_ensemble_v2.py); no se ajusta con validation.
APP_ENSEMBLE_CNN_WEIGHT = 0.5

# Probabilidad mínima de la clase predicha para mostrarla sin advertencia. Por debajo, la app
# pide revisión manual. PROVISIONAL: elegirlo en validation antes de la entrega.
APP_MIN_CONFIDENCE = 0.55
# Versión de scikit-learn con la que se serializaron los checkpoints clásicos.
SKLEARN_CHECKPOINT_VERSION = "1.7.2"

# Proporción del conjunto de prueba (se usa una sola vez, en la evaluación final).
TEST_SIZE = 0.15
# Proporción del conjunto de validación (para ajustar hiperparámetros).
VAL_SIZE = 0.15
