# 4. Modelado

## Objetivo y datos de entrada

Se clasifican recortes de una pieza en `good`, `regular` o `bad`. Los modelos se entrenaron y compararon con el split grupal `data/processed/crops/splits.csv` (70/15/15 objetivo), de hash SHA-256 `5b37044ba55c4769e0f2d503222ed3d7104293d24e9484a127d2691171b223e3`. Validation tiene 3.557 recortes y 1.979 grupos. El conjunto test no se ha usado.

Los modelos clásicos reciben 359 atributos de color RGB/HSV, textura LBP y forma, extraídos de la pieza enmascarada. La CNN recibe recortes RGB de 128 × 128 con el fondo anulado mediante la máscara.

## Modelos entrenados

| Modelo | Entrada | Resultado en validation | Artefacto local |
| --- | --- | ---: | --- |
| Regresión logística | 359 atributos; `StandardScaler`; pesos balanceados | F1 macro 0.6845; balanced accuracy 0.7010 | `experiments/checkpoints/logistic_baseline.joblib`; `experiments/results/logistic_baseline.json` |
| SVM lineal | 359 atributos; `LinearSVC`; C=1.0 por CV grupal | F1 macro 0.6978; balanced accuracy 0.7022 | `experiments/checkpoints/svm.joblib`; `experiments/results/svm.json` |
| Random Forest | 359 atributos; 160 árboles; `max_features=sqrt`; `min_samples_leaf=2` | F1 macro 0.8424; balanced accuracy 0.8363 | `experiments/checkpoints/random_forest.joblib`; `experiments/results/random_forest.json` |
| CNN RGB | 128 × 128 × 3; mejor checkpoint de la época 40 | F1 macro 0.7988; balanced accuracy 0.8139 | `experiments/checkpoints/cnn_rgb.keras`; `experiments/results/cnn_rgb.json` |

La selección de SVM y Random Forest usó `StratifiedGroupKFold` de tres folds solo sobre train, con F1 macro como objetivo. El checkpoint CNN ya existe. Su JSON de evaluación indica `checkpoint_evaluation_only: true`: la corrida que produjo las métricas finales de CNN cargó el checkpoint y no volvió a entrenar.

## Lectura provisional de validation

Random Forest tiene la mayor F1 macro agregada, seguido por CNN RGB. El resultado no basta para declarar un modelo final: existe un sesgo de fuente importante, y todavía no se han inspeccionado individualmente los casos marcados por la auditoría.

| Clase | Soporte | F1 Random Forest | F1 CNN |
| --- | ---: | ---: | ---: |
| `good` | 1.405 | 0.8798 | 0.8300 |
| `regular` | 705 | 0.7557 | 0.7017 |
| `bad` | 1.447 | 0.8917 | 0.8647 |

La clase `regular` en validation proviene completamente de AgriFreshNET: 705 recortes y 251 grupos (100%). `good` y `bad` tienen ejemplos de varias fuentes. Por ello, el rendimiento sobre `regular` no demuestra generalización a otros orígenes; fuente, apariencia y etiqueta pueden estar confundidas.

La comparación fila a fila de RF y CNN encontró 2.686 aciertos de ambos, 210 fallos solo de RF, 378 fallos solo de CNN y 283 fallos de ambos. Por tanto, 871/3.557 filas (24,5%) tienen al menos uno de esos dos modelos equivocado y requieren priorización para revisión. El Jaccard de los conjuntos de error es 0.3249: hay complementariedad parcial, pero no debe usarse para proponer un ensemble antes de auditar los casos.

## Riesgo de reproducibilidad de scikit-learn

Los checkpoints clásicos se serializaron con scikit-learn 1.7.2. La auditoría posterior se ejecutó en un ambiente con 1.9.1 y generó `InconsistentVersionWarning`. scikit-learn no garantiza compatibilidad de objetos persistidos entre versiones. En consecuencia, las métricas agregadas de entrenamiento previas siguen como referencia registrada, pero las predicciones por imagen, el desglose de errores y la comparación RF/CNN son provisionales hasta repetir la auditoría con scikit-learn 1.7.2. `requirements.txt` fija ahora esa versión recomendada para el proyecto; no reentrenar para resolver la advertencia, primero alinear el ambiente y volver a puntuar.

## Checkpoints, resultados y privacidad de datos

Los checkpoints viven en `experiments/checkpoints/` y se excluyen de Git por tamaño. Los JSON, CSV y curvas viven en `experiments/results/` y `experiments/logs/`; pueden contener rutas locales y no sustituyen los datos originales. Para compartir el trabajo, subir el código, la documentación y resultados derivados acordados por el equipo; no subir datasets ni checkpoints sin revisar el tamaño, licencia y política del repositorio.

## Cierre de modelado

Los cuatro modelos están entrenados y tienen métricas iniciales en validation. El modelado aún no está congelado: primero se debe repetir la auditoría de validation con la versión compatible, revisar los 871 casos señalados y cerrar problemas de etiquetas, máscaras y fuente. No abrir test durante estos pasos. Después se documentan hashes de los artefactos congelados y se hace una única evaluación final en test.
