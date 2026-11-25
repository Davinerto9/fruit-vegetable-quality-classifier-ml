# Clasificador de calidad y tamaño de frutas y verduras

Proyecto final del curso **Algoritmos y Programación III**, Universidad Icesi, Cali, Colombia (2026-2).

**Estado del proyecto:** en desarrollo.

[Contexto y estado detallado del proyecto](docs/contexto-y-estado-proyecto.md)

[Guía del equipo: evaluación y auditoría](docs/modeling_team_guide.md) · [Cierre de modelado](docs/modeling_session_wrapup.md) · [Script para exportar resumen PDF](scripts/export_modeling_summary_pdf.py)

[Guía de avance y tareas pendientes](output/pdf/guia-proyecto-y-tareas.pdf)

[Resumen de avance y pendientes del 15 de noviembre](output/pdf/avance-y-pendientes-del-equipo.pdf)

[Metodología CRISP-DM, diagrama y estado actual](docs/crisp-dm.md) · [Revisión de anotaciones y segmentación](docs/revision-anotaciones.md)

## Objetivo
A partir de una foto de **una** fruta o verdura sobre fondo uniforme, el sistema predice:

- **Calidad:** buena (`good`), regular (`regular`) o mala (`bad`), según los [criterios de anotación](docs/criterios-calidad.md).
- **Tamaño:** pequeño, mediano o grande, además de un diámetro normalizado.

Seguimos la metodología **CRISP-DM**. Sobre el split v2 por ráfagas, los candidatos son Random Forest (F1 macro en validation 0,857), una CNN de 4 bloques (0,858) y su ensamble 0,5/0,5 (0,875, con menos errores `bad→good`); la regresión logística y la SVM lineal quedan como líneas base. El conjunto test sigue reservado. El seguimiento de los experimentos está en [modeling_fronts_tracking.md](docs/modeling_fronts_tracking.md). **La versión que se sustenta sigue el [plan simple](docs/plan_simple.md):** la app identifica la fruta y su estado, con modelos reentrenados sin la fuga de AgriFreshNET. La aplicación de **Streamlit** permite cargar una foto o tomarla con la cámara y muestra la calidad, las probabilidades y el tamaño relativo; consulta [Despliegue](docs/06_deployment.md). Consulta la [guía de modelado y auditoría](docs/modeling_team_guide.md) antes de reproducir resultados.

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
src/training/      entrenamiento
src/evaluation/    métricas y gráficas
src/app/           app Streamlit (app.py) e inferencia (inference.py)
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

El siguiente comando comprueba la segmentación y genera recortes de las fotos propias y una muestra externa.

```bash
python -m src.data.prepare --pilot
```

Puedes consultar las decisiones de preparación en `data/processed/pilot-crops/manifest.csv`. Al ejecutar el comando sin `--pilot`, se prepara el inventario completo en `data/processed/crops/`.

Después de revisar los recortes candidatos, crea las particiones con:

```bash
python -m src.data.split
```

El manifiesto incluye, además de la decisión de cada foto, el número de frutas detectadas y estadísticas de imagen (dimensiones, brillo, contraste y nitidez) que usa el análisis exploratorio. El comando de partición genera `data/processed/crops/splits.csv` y reporta la cobertura por calidad, fuente y producto. Cada grupo pertenece a una sola partición. Los grupos marcados en la [revisión visual](data/annotations/segmentation_exclusions.csv) se excluyen, incluidas sus variantes. Para probar el flujo sobre una muestra, pasa `--manifest data/processed/pilot-crops/manifest.csv`.

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

**App** (requiere `random_forest_v2_burst_grid.joblib`; con `cnn_rgb_v2_burst_e3b.keras` y TensorFlow usa el ensamble):
```bash
streamlit run src/app/app.py
```
Detalles de arquitectura, decisiones y pruebas de uso en [Despliegue](docs/06_deployment.md).

## Créditos

### Datos
- **FruitNet**, el dataset original: V. Meshram y K. Patil, "FruitNet: Indian fruits image dataset with quality for machine learning applications", *Data in Brief*, vol. 40, art. 107686, 2022. DOI: [10.1016/j.dib.2021.107686](https://doi.org/10.1016/j.dib.2021.107686). El artículo se publicó con licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Fruit Quality Classification**, la versión de Kaggle que usamos: R. Park, publicada en [Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) con licencia CC0 (dominio público). Es una redistribución de FruitNet, por eso también citamos el trabajo original.
- F. Risdin, H. Salauddin, T. A. Tufa y M. J. Hossain, **AgriFreshNET Freshness and Shelf-Life Image Dataset**, versión 1, Mendeley Data, 2026. [DOI 10.17632/42m5tb7yv9.1](https://data.mendeley.com/datasets/42m5tb7yv9/1) · [Licencia CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Cambios realizados:** sus carpetas Fresh, Semi Fresh y Rotten se mapearon a nuestras clases good, regular y bad, y las imágenes se segmentan, recortan y redimensionan para entrenar los modelos. Los autores no respaldan este proyecto.
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
