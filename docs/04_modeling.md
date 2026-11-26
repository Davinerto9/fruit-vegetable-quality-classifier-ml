# 4. Modelado

## Qué se entrena

A partir del recorte de **una** pieza (ver [preparación](03_data_preparation.md)), el sistema responde dos preguntas:

| Pregunta | Modelo | Entrada |
| --- | --- | --- |
| ¿Qué fruta o verdura es? | Random Forest de fruta (12 productos) | 359 atributos: color RGB/HSV, textura LBP y forma |
| ¿En qué estado está? | Regresión logística (línea base), SVM lineal, Random Forest y CNN | Los mismos 359 atributos; la CNN recibe el recorte RGB de 128 × 128 con el fondo anulado |

El estado que muestra la app es el **promedio 50/50 de las probabilidades del Random Forest y la CNN**. El peso se fijó de antemano y no se ajusta con validación. Las ecuaciones de cada modelo están en los [fundamentos matemáticos](supporting_docs/mathematical_and_theoretical_foundations.md#6-modelos).

## Cómo se eligen los hiperparámetros

- **Modelos clásicos** (`src/training/train_classic.py`): `GridSearchCV` de scikit-learn con `StratifiedGroupKFold(3)` y métrica F1 macro, solo con *train*. Los grupos (fotos de la misma ráfaga o sesión) nunca quedan a la vez en entrenamiento y evaluación de un fold. El mejor modelo se reentrena con todo *train*.

  | Modelo | Rejilla |
  | --- | --- |
  | Regresión logística | C ∈ {0,1; 1; 10} |
  | SVM lineal | C ∈ {0,1; 1; 10} |
  | Random Forest (estado) | `min_samples_leaf` ∈ {1, 5, 20} × `max_depth` ∈ {sin límite, 20} × `max_features` ∈ {√p, 0,1} |
  | Random Forest (fruta) | `min_samples_leaf` ∈ {1, 5} × `max_features` ∈ {√p, 0,1} |

- **CNN** (`src/training/train_cnn.py`): 4 bloques Conv 3×3 → BatchNorm → ReLU → MaxPool (32, 64, 128 y 256 filtros), pooling global promedio, *dropout* 0,35 y softmax; 389.667 parámetros. Adam con reducción del learning rate en meseta. La parada temprana usa el 10% de los grupos de *train* (`--es-holdout 0.1`), no validación.
- **Desbalance:** pesos de clase calculados solo con *train*.

## Cómo llegamos a esta configuración

| Versión | Qué cambió | Resultado en validación (F1 macro del estado) |
| --- | --- | --- |
| v1 | Primer reparto de fotos | RF 0,842 y CNN 0,799, **inflados**: fotos casi idénticas de Kaggle quedaban a la vez en entrenamiento y validación. |
| v2 | Reparto por ráfagas de Kaggle | RF 0,857, CNN de 4 bloques 0,858 y promedio 0,875. La CNN pasó de 3 a 4 bloques porque con 3 no alcanzaba (subajuste); aumentar datos con giros empeoró. Seguía la trampa de AgriFreshNET. |
| **v3** | Reparto por sesiones de AgriFreshNET y validación usada una sola vez | Pendiente de reentrenar con el [plan simple](plan_simple.md). |

Las cifras de v1 y v2 solo explican el camino; las de la sustentación son las de v3.

## Reproducibilidad

Los comandos, en orden y con tiempos, están en el [plan simple](plan_simple.md) (pasos 1 a 6). Los checkpoints se guardan en `experiments/checkpoints/`, fuera de Git por tamaño. Los resultados, en `experiments/results/classic_<tag>.json` y `cnn_rgb_<run>.json`.
