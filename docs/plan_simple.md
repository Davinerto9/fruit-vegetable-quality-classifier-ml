# Plan simple: fruta + estado, sin trampa

**Objetivo.** Con una foto, la app dice **qué fruta es** y **en qué estado está** (buena, regular o mala), con cifras que se puedan defender: medidas sin trampa y con modelos sencillos.

## Qué hace el sistema

1. **Separa la fruta del fondo** con OpenCV (`src/data/segmentation.py`). Si hay varias frutas o el fondo se confunde con la fruta, pide otra foto en lugar de adivinar.
2. **Describe la fruta** con 359 números de color, textura y forma (`src/features/classic.py`).
3. **Fruta:** un Random Forest con esos números elige entre los 12 productos que conoce. Si su probabilidad es menor al 50%, la app dice que no la reconoce.
4. **Estado:** un Random Forest y una red convolucional (4 bloques) dan cada uno una probabilidad por estado; la app promedia las dos, mitad y mitad.

| Pieza | Modelo | Por qué |
| --- | --- | --- |
| Fruta | Random Forest (200 árboles) | Simple, rápido y explicable; usa los mismos atributos. |
| Estado (clásicos) | Regresión logística, SVM lineal, Random Forest | La consigna pide al menos dos modelos clásicos; la logística es la línea base. |
| Estado (profundo) | CNN de 4 bloques Conv-BatchNorm-ReLU-MaxPool | La consigna pide un modelo de deep learning. |
| Estado final | Promedio RF + CNN | Una línea de código y menos errores de vender fruta mala como buena. |

## Sin trampa: tres reglas

1. **Fotos parecidas, del mismo lado.** AgriFreshNET fotografió las mismas frutas durante semanas. Por eso todas las fotos de una misma fruta y semana quedan juntas en *train*, en *validación* o en *prueba* (`src/data/split_v3.py`). Lo mismo con las ráfagas de Kaggle.
2. **Hiperparámetros con validación cruzada.** Se eligen con `GridSearchCV` y `StratifiedGroupKFold(3)` de scikit-learn, solo con *train* y respetando los grupos de la regla 1. La CNN decide cuándo parar con un 10% de *train*, no con validación.
3. **Validación y prueba, una vez.** Ninguna se usa para elegir nada. La prueba tiene un candado: si se intenta abrir otra vez, el programa se niega.

## Métricas (y cómo explicarlas)

- **Accuracy (fruta):** de cada 100 fotos, en cuántas acertó la fruta.
- **F1 macro (estado):** el promedio del F1 de buena, regular y mala, cada una con el mismo peso, así que no se puede inflar acertando solo la clase más común.
- **Matriz de confusión:** qué se confunde con qué.
- **Errores mala→buena:** cuántas frutas malas pasarían como buenas. Es el error más caro para un comprador.
- **Intervalo de confianza:** el rango donde probablemente está la cifra real, calculado remuestreando grupos de fotos.

## Pasos (en la máquina con los datos y la GPU)

Desde la raíz del repo, con el ambiente activado (`pip install -r requirements.txt`). Con `S=data/processed/crops/splits_v3_week.csv` y `F=data/processed/classic_features_v3_week.npz`:

| # | Qué | Comando | Tiempo |
| --- | --- | --- | --- |
| 1 | Repartir las fotos sin trampa | `python -m src.data.split_v3 --choose` | 1 min |
| 2 | *(Opcional)* medir cuánto inflaba la trampa | `PYTHONPATH=. python experiments/diagnostics/agri_leak_ladder.py` | 20 min |
| 3 | Atributos para el nuevo reparto | `python -m src.features.subset_classic_cache --split $S --output $F` | 1 min |
| 4 | Modelos clásicos (estado y fruta) | `python -m src.training.train_classic --split $S --features $F --tag v3_week` | 15–25 min |
| 5 | CNN | `python -m src.training.train_cnn --split $S --run-name v3_week --epochs 100 --patience 12 --reduce-lr-patience 4 --filters 32,64,128,256 --cache --es-holdout 0.1` | ~45 min GPU |
| 6 | Validación, una vez | `python -m src.evaluation.model_audit` | 5 min |
| 7 | Fotos de celular | descargar la carpeta de Drive en `data/external/raw/<Fruta>/`, luego `python -m src.data.external_pool ingest` y `python -m src.data.external_pool sheets` | 10 min |
| 8 | Etiquetar el estado a ciegas | una persona llena `data/external/labeling/annotator_A.csv` mirando `sheet_A.html`, sin abrir la app. *(Opcional: otra persona llena B y se corre `external_eval agreement`.)* Luego `python -m src.evaluation.external_eval finalize` | 1 h |
| 9 | Evaluar fotos de celular, una vez | `python -m src.evaluation.external_eval evaluate --confirm-external` | 2 min |
| 10 | Congelar y abrir prueba, una vez | `sha256sum <split> <checkpoints> > experiments/results/pretest_freeze.sha256` (Git Bash), luego `python -m src.evaluation.model_audit --partition test --confirm-final-test` | 5 min |

Si el paso 1 elige otro nivel que `week`, cambia el sufijo en los comandos y en `src/config.py` (`APP_*_CHECKPOINT`).

## Límites que hay que decir en la sustentación

- **"Cualquier foto" tiene condiciones:** una sola fruta, fondo liso y distinto a la fruta, y que la fruta esté completa. La app rechaza las fotos que no cumplen, en lugar de inventar.
- **Solo conoce 12 productos.** El mango, por ejemplo, no está: la app debe decir que no lo reconoce.
- **El tamaño es relativo al encuadre,** no centímetros.
- **El estado es la apariencia externa,** no la inocuidad ni el interior de la fruta.

## Qué quedó fuera, a propósito

Para mantenerlo simple no se usa transfer learning, búsqueda de variantes de la CNN, calibración del ensamble ni adaptación de dominio. Se pueden mencionar como trabajo futuro.
