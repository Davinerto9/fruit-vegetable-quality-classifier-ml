# Clasificador de calidad y tamaño de frutas y verduras

Proyecto final del curso **Algoritmos y Programación III**, Universidad Icesi, Cali, Colombia (2026-2).

**Estado del proyecto:** en desarrollo.

[Guía de avance y tareas pendientes](output/pdf/guia-proyecto-y-tareas.pdf)

## Objetivo
A partir de una foto de **una** fruta o verdura sobre fondo uniforme, el sistema predice:

- **Calidad:** buena (`good`), regular (`regular`) o mala (`bad`), según los [criterios de anotación](docs/criterios-calidad.md).
- **Tamaño:** pequeño, mediano o grande, además de un diámetro normalizado.

Seguimos la metodología **CRISP-DM** y comparamos dos modelos de ML clásico (SVM y Random Forest) con una red neuronal convolucional pequeña entrenada desde cero. El resultado se despliega en una app de **Streamlit** que acepta una imagen cargada o una foto de la cámara.

## Integrantes

| Nombre | GitHub |
|---|---|
| Ismael Barrionuevo | [@DavidRamirez206](https://github.com/DavidRamirez206) |
| David Chicué | [@Davinerto9](https://github.com/Davinerto9) |
| Daniel Martinez | [@damaral2005](https://github.com/damaral2005) |

## Estructura del repositorio
```
data/raw/          imágenes de Kaggle y AgriFreshNET (no se suben a git)
data/own/          fotos propias + labels.csv
data/processed/    índices, particiones y características derivadas
src/data/          carga, segmentación, preprocesamiento y partición
src/features/      extracción de características
src/models/        definición de modelos (CNN)
src/training/      entrenamiento
src/evaluation/    métricas y gráficas
src/app/           app Streamlit
notebooks/         EDA, modelado y resultados
experiments/       logs/, checkpoints/, results/, figures/
docs/              documentación extendida
tests/             pruebas con pytest
```

## Instalación

### Con conda (recomendado)
Desde la raíz del repositorio:
```bash
conda env create -f environment.yml
conda activate fruit-quality
```
Si cambia `requirements.txt`, actualiza el ambiente con:
```bash
conda env update -f environment.yml --prune
```

### Con venv
```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Datos
1. Descarga [Fruit Quality Classification](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) de Kaggle. Coloca las carpetas `Good Quality_Fruits`, `Bad Quality_Fruits` y `Mixed Qualit_Fruits` dentro de `data/raw/ryandpark/`.
2. Descarga los datos procesados de [AgriFreshNET](https://data.mendeley.com/datasets/42m5tb7yv9/1). Coloca las carpetas `Fresh`, `Semi-Fresh` y `Rotten` de cada producto en `data/raw/agrifreshnet/Processed Data/`, directamente debajo de esa ruta.

Las fotos propias del equipo van en `data/own/`, con sus etiquetas en `data/own/labels.csv`.

Genera el inventario local después de preparar las tres fuentes:

```bash
python -m src.data.inventory
```

El comando escribe `data/processed/index.csv` con una fila por archivo y rutas relativas al repositorio. Consulta los [campos y criterios de agrupación](docs/criterios-calidad.md#inventario-de-imágenes) antes de usar el índice.

## Uso
Todos los comandos se ejecutan desde la raíz del repositorio, con el ambiente activado.

**Notebooks:**
```bash
jupyter lab
```
Abre [el notebook de análisis exploratorio](notebooks/01_eda.ipynb) después de generar el inventario. Incluye conteos por fuente, producto y clase, el desbalance por archivo y grupo, y ejemplos de las tres calidades. Si usas VS Code u otro Jupyter instalado fuera del ambiente, registra el kernel una sola vez y luego selecciona "Python (fruit-quality)":
```bash
python -m ipykernel install --user --name fruit-quality --display-name "Python (fruit-quality)"
```

**Pruebas:**
```bash
pytest
```

**App** (disponible cuando se implemente `src/app/app.py`):
```bash
streamlit run src/app/app.py
```

## Créditos

### Datos
- **FruitNet**, el dataset original: V. Meshram y K. Patil, "FruitNet: Indian fruits image dataset with quality for machine learning applications", *Data in Brief*, vol. 40, art. 107686, 2022. DOI: [10.1016/j.dib.2021.107686](https://doi.org/10.1016/j.dib.2021.107686). El artículo se publicó con licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Fruit Quality Classification**, la versión de Kaggle que usamos: R. Park, publicada en [Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) con licencia CC0 (dominio público). Es una redistribución de FruitNet, por eso también citamos el trabajo original.
- **Fotos propias:** tomadas por los integrantes del equipo para este proyecto.

### Plantilla
La estructura del repositorio parte de la plantilla sugerida en el curso, que a su vez se basa en la del [Data Science Working Group de Code for San Francisco](https://github.com/sfbrigade/data-science-wg).

### Librerías

| Librería | Licencia |
|---|---|
| [NumPy](https://numpy.org/) | BSD-3-Clause |
| [pandas](https://pandas.pydata.org/) | BSD-3-Clause |
| [OpenCV](https://opencv.org/) (`opencv-python`) | Apache 2.0 (OpenCV), MIT (empaquetado) |
| [scikit-learn](https://scikit-learn.org/) | BSD-3-Clause |
| [TensorFlow / Keras](https://www.tensorflow.org/) | Apache 2.0 |
| [Matplotlib](https://matplotlib.org/) | Matplotlib License (basada en PSF) |
| [seaborn](https://seaborn.pydata.org/) | BSD-3-Clause |
| [Streamlit](https://streamlit.io/) | Apache 2.0 |
| [joblib](https://joblib.readthedocs.io/) | BSD-3-Clause |
| [pytest](https://pytest.org/) | MIT |
| [JupyterLab](https://jupyter.org/) e [ipykernel](https://github.com/ipython/ipykernel) | BSD-3-Clause |

## Licencia
El código de este repositorio se distribuye bajo la licencia **GNU General Public License v2.0**; consulta [LICENSE](LICENSE). Los datos de terceros conservan sus licencias originales, indicadas en [Créditos](#créditos).
