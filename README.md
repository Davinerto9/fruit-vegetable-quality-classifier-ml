# Clasificador de calidad y tamaño de frutas y verduras

Proyecto final del curso **Algoritmos y Programación III**, Universidad Icesi, Cali, Colombia (2026-2).

**Estado del proyecto:** terminado. Modelo final (Random Forest) en test: F1 macro del estado 0,596 [IC 95 %: 0,522 – 0,677] y 79,5 % de acierto en la fruta ([Evaluación](docs/05_evaluation.md)). Informe IEEE en [`report/informe.pdf`](report/informe.pdf) · [video del proyecto](https://youtu.be/gwlolsmqH44).

[Plan simple: qué hace el sistema y por qué es honesto](docs/plan_simple.md) · [Metodología CRISP-DM](docs/crisp-dm.md) · [Criterios de calidad](docs/criterios-calidad.md)

## Objetivo
A partir de una foto de **una** fruta o verdura sobre fondo uniforme, el sistema dice:

- **Qué fruta o verdura es**, entre los 13 productos que conoce; si no está seguro, lo dice.
- **Su estado:** buena (`good`), regular (`regular`) o mala (`bad`), según los [criterios de anotación](docs/criterios-calidad.md).
- **Su tamaño relativo:** pequeño, mediano o grande según cuánto ocupa la foto (no centímetros).

Seguimos la metodología **CRISP-DM**. La fruta la identifica un Random Forest. Para el estado se comparan regresión logística, SVM lineal, Random Forest, una red convolucional de 4 bloques y el promedio de RF y CNN; el modelo final es el **Random Forest**, elegido en validación porque empata en F1 y deja pasar la mitad de frutas malas como buenas. Los hiperparámetros se eligen con `GridSearchCV` y validación cruzada por grupos, de modo que fotos casi iguales nunca quedan a la vez en entrenamiento y evaluación ([por qué importa](docs/03_data_preparation.md#cómo-llegamos-al-reparto-final)). La app de **Streamlit** recibe una foto o la cámara; consulta [Despliegue](docs/06_deployment.md).

## Integrantes

| Nombre | GitHub |
|---|---|
| Ismael Barrionuevo | [@DavidRamirez206](https://github.com/DavidRamirez206) |
| David Chicué | [@Davinerto9](https://github.com/Davinerto9) |
| Daniel Martinez | [@damaral2005](https://github.com/damaral2005) |

## Estructura del repositorio
```
app/app.py                      la aplicación (Streamlit)
models/                         los 5 modelos entrenados (se crean al entrenar y se suben a Git)
results/                        métricas en JSON, predicciones y figures/ para el informe
notebooks/
  01_eda.ipynb                  cuántas fotos hay de cada fruta, estado y fuente
  02_modelos_clasicos.ipynb     rejillas de hiperparámetros y resultados de los clásicos
  03_cnn.ipynb                  arquitectura y curvas de aprendizaje de la CNN
  04_resultados.ipynb           comparación final, matrices de confusión, fuga, celular
src/
  main.py                       un comando por paso: python -m src.main <paso>
  config.py                     rutas, clases, semilla y parámetros (un solo lugar)
  data/
    inventory.py                1. lista todas las fotos con su fruta, estado y grupo
    segmentation.py             separa la fruta del fondo; rechaza fotos que no sirven
    prepare.py                  2. aplica la segmentación y guarda recortes de 128x128
    split.py                    3. reparte en train / val / test por grupos (sin fuga)
    features.py                 4. los 359 atributos de color, textura y forma
  models/model_definitions.py   los 5 modelos y sus rejillas de hiperparámetros
  training/train_models.py      5. GridSearchCV por grupos + CNN; guarda en models/
  prediction/predictor.py       foto nueva -> rechazo, fruta, estado y tamaño (lo usa la app)
  evaluation/
    evaluate_models.py          6. métricas en val; test una sola vez, con candado
    leak_check.py               prueba de fuga (opcional)
    phone_test.py               fotos de celular: mitad para entrenar, mitad para la prueba final
data/                           raw/ (descargas), own/ (fotos propias), processed/ (derivados)
docs/                           documentación CRISP-DM
tests/                          pruebas con pytest
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
3. Descarga [AFruitDB](https://data.mendeley.com/datasets/bz65dz2pbj/1) ("Download All") y descomprímelo en `data/raw/afruitdb/`, de modo que queden `data/raw/afruitdb/Apple/1st grade/...`. Se usan manzana, banano, mango, papaya y tomate; la uva birmana queda fuera.

Las fotos propias del equipo van en `data/own/`, con sus etiquetas en `data/own/labels.csv`.

## Cómo usar

Todos los comandos se corren desde la raíz del repositorio, con el ambiente activado.

### Probar la app (con los modelos ya entrenados de `models/`)
```bash
streamlit run app/app.py
```
Sube una foto de **una** fruta sobre un fondo liso. La app dice qué fruta es, su estado y su tamaño, o explica por qué rechaza la foto. Sin TensorFlow funciona solo con el Random Forest. Detalles en [Despliegue](docs/06_deployment.md).

### Rehacer todo desde los datos

**Antes de entrenar: las fotos de celular del curso.** Se parten por sesión (misma cámara y día) en dos mitades: una **entrena** el modelo y la otra queda como **prueba final**, que el modelo nunca ve. El estado de las dos se etiqueta a mano y a ciegas (detalles en [`src/evaluation/phone_test.py`](src/evaluation/phone_test.py)):
```bash
python -m src.main celular ingerir      # fotos en data/external/raw/<Fruta>/; recorta y parte en mitades
python -m src.main celular hojas        # crea data/external/labeling/annotator_A.csv y sheet_A.html
#   en annotator_A.csv escribir good / regular / bad solo en la primera foto de cada pieza
#   (las casillas vacías copian la anterior), mirando sheet_A.html
python -m src.main celular finalizar    # pasa la mitad "entrenar" al entrenamiento
```
Para **agregar fotos después** (por ejemplo, mangos nuevos), ponlas en `data/external/raw/<Fruta>/` y vuelve a correr `celular ingerir` y `celular hojas`: solo se procesan las fotos nuevas, las anteriores conservan su mitad y sus etiquetas, y el programa dice en qué número de la hoja empiezan las nuevas. No se puede después de `celular evaluar`.
Si se saltan este paso, el modelo entrena solo con Kaggle, AgriFreshNET, AFruitDB y `data/own/`.

**Después**, con los datos descargados (ver arriba), un solo comando corre los pasos 1 a 6 en orden:
```bash
python -m src.main todo
```
O paso por paso, para entender qué hace cada uno:

| # | Comando | Qué hace | Deja |
|---|---|---|---|
| 1 | `python -m src.main inventario` | Lista cada foto con su fruta, estado y grupo | `data/processed/index.csv` |
| 2 | `python -m src.main recortar` | Separa la fruta del fondo y la recorta a 128x128 (~30 min) | `data/processed/crops/` |
| 3 | `python -m src.main dividir` | Reparte 70/15/15 en train, val y test, sin separar fotos hermanas; suma a train la mitad "entrenar" del celular | `crops/splits.csv` |
| 4 | `python -m src.main atributos` | Calcula los 359 atributos de cada recorte | `data/processed/features.npz` |
| 5 | `python -m src.main entrenar` | Elige hiperparámetros con validación cruzada y entrena (clásicos ~20 min; CNN ~45 min con GPU) | `models/` |
| 6 | `python -m src.main evaluar` | Mide todos los modelos en val | `results/evaluacion_val.json` |

`todo` se salta los pasos que ya están al día: si después etiquetan más fotos de celular y vuelven a correr `celular finalizar`, `todo` rehace desde el paso 3. Luego se abren los notebooks 02 a 04 para ver tablas y gráficas.

### Al final, una sola vez
```bash
python -m src.main fuga                            # (opcional) cuánto se inflaba la nota sin agrupar
python -m src.main congelar                        # anota la huella de modelos y reparto
python -m src.main evaluar --test --confirmo       # test: no se puede repetir
python -m src.main celular evaluar --confirmo      # la mitad "prueba" del celular: tampoco se repite
```

### Notebooks y pruebas
```bash
jupyter lab
pytest
```
Si usas VS Code u otro Jupyter instalado fuera del ambiente, registra el kernel una sola vez y selecciona "Python (fruit-quality)":
```bash
python -m ipykernel install --user --name fruit-quality --display-name "Python (fruit-quality)"
```

## Créditos

### Datos
- **FruitNet**, el dataset original: V. Meshram y K. Patil, "FruitNet: Indian fruits image dataset with quality for machine learning applications", *Data in Brief*, vol. 40, art. 107686, 2022. DOI: [10.1016/j.dib.2021.107686](https://doi.org/10.1016/j.dib.2021.107686). El artículo se publicó con licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Fruit Quality Classification**, la versión de Kaggle que usamos: R. Park, publicada en [Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) con licencia CC0 (dominio público). Es una redistribución de FruitNet, por eso también citamos el trabajo original.
- F. Risdin, H. Salauddin, T. A. Tufa y M. J. Hossain, **AgriFreshNET Freshness and Shelf-Life Image Dataset**, versión 1, Mendeley Data, 2026. [DOI 10.17632/42m5tb7yv9.1](https://data.mendeley.com/datasets/42m5tb7yv9/1) · [Licencia CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Cambios realizados:** sus carpetas Fresh, Semi Fresh y Rotten se mapearon a nuestras clases good, regular y bad, y las imágenes se segmentan, recortan y redimensionan para entrenar los modelos. Los autores no respaldan este proyecto.
- M. U. Mojumdar, S. Islam, M. A. Mamun, R. Hasan, S. M. T. Siddiquee y N. R. Chakraborty, "AFruitDB: A comprehensive dataset of six commonly used Asian fruits for advanced grading and biodiversity insights", *Data in Brief*, 2025. DOI: [10.1016/j.dib.2025.111380](https://doi.org/10.1016/j.dib.2025.111380). Datos en [Mendeley Data, DOI 10.17632/bz65dz2pbj.1](https://data.mendeley.com/datasets/bz65dz2pbj/1) · [Licencia CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). **Cambios realizados:** solo se usan manzana, banano, mango, papaya y tomate; sus grados 1st, 2nd y 3rd se mapearon a good, regular y bad, y las imágenes se segmentan, recortan y redimensionan. Los autores no respaldan este proyecto.
- **Fotos propias:** tomadas por los integrantes del equipo para este proyecto, salvo las seis de mango de `data/own/`, que provienen de la carpeta compartida del curso (ver abajo).
- **Fotos de celular del curso:** carpeta compartida *Photos of Fruits and Vegetables | APO III* (Google Drive), aportada por los estudiantes de Algoritmos y Programación III 2026-2. La mitad de sus sesiones se usa para entrenar y la otra mitad como prueba final; no se redistribuye en este repositorio.

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
