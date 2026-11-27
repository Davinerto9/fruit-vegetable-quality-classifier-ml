# 4. Modelado

## Qué se entrena

A partir del recorte de **una** pieza (ver [preparación](03_data_preparation.md)), el sistema responde dos preguntas:

| Pregunta | Modelo | Entrada |
| --- | --- | --- |
| ¿Qué fruta o verdura es? | Random Forest de fruta (13 productos) | 359 atributos: color RGB/HSV, textura LBP y forma |
| ¿En qué estado está? | Regresión logística (línea base), SVM lineal, Random Forest y CNN | Los mismos 359 atributos; la CNN recibe el recorte RGB de 128 × 128 con el fondo anulado |

El estado que muestra la app es el **promedio 50/50 de las probabilidades del Random Forest y la CNN**. El peso se fijó de antemano y no se ajusta con validación. Las ecuaciones de cada modelo están en los [fundamentos matemáticos](supporting_docs/mathematical_and_theoretical_foundations.md#6-modelos).

## Con qué fotos se entrena

*Train* del reparto por grupos (Kaggle, AgriFreshNET y `data/own/`) más la mitad "entrenar" de las fotos de celular del curso, ya etiquetadas. Cada sesión de celular es un grupo, así que la validación cruzada tampoco mezcla fotos de una misma sesión. La otra mitad del celular no se usa aquí: es la prueba final.

## Cómo se eligen los hiperparámetros

- **Modelos clásicos** (definidos en `src/models/model_definitions.py`, entrenados en `src/training/train_models.py`): `GridSearchCV` de scikit-learn con `StratifiedGroupKFold(3)` y métrica F1 macro, solo con *train*. Los grupos (fotos de la misma ráfaga o sesión) nunca quedan a la vez en entrenamiento y evaluación de un fold. El mejor modelo se reentrena con todo *train*.

  | Modelo | Rejilla |
  | --- | --- |
  | Regresión logística | C ∈ {0,1; 1; 10} |
  | SVM lineal | C ∈ {0,1; 1; 10} |
  | Random Forest (estado) | `min_samples_leaf` ∈ {1, 5, 20} × `max_depth` ∈ {sin límite, 20} × `max_features` ∈ {√p, 0,1} |
  | Random Forest (fruta) | `min_samples_leaf` ∈ {1, 5} × `max_features` ∈ {√p, 0,1} |

- **CNN** (mismos archivos): una red pequeña hecha desde cero, sin modelos preentrenados. 4 bloques Conv 3×3 → BatchNorm → ReLU → MaxPool (32, 64, 128 y 256 filtros), pooling global promedio, *dropout* 0,35 y softmax; 389.667 parámetros. Adam con reducción del learning rate en meseta. La parada temprana usa el 10 % de los grupos de *train*, no val. Los ajustes están en `CNN_SETTINGS`; su justificación, en el [notebook 03](../notebooks/03_cnn.ipynb).
- **Desbalance:** pesos de clase calculados solo con *train*.

## Cómo llegamos a esta configuración

| Intento de reparto ([preparación](03_data_preparation.md#cómo-llegamos-al-reparto-final)) | Resultado en val (F1 macro del estado) |
| --- | --- |
| 1 | RF 0,842 y CNN 0,799, **inflados**: fotos casi idénticas de Kaggle quedaban a la vez en train y val. |
| 2 | RF 0,857, CNN de 4 bloques 0,858 y promedio 0,875. La CNN pasó de 3 a 4 bloques porque con 3 no alcanzaba (subajuste); aumentar datos con giros empeoró. Seguía la fuga de AgriFreshNET. |
| **Final** | Pendiente de reentrenar (`python -m src.main entrenar`). |

Las cifras de los intentos 1 y 2 solo explican el camino; las de la sustentación son las del reparto final.

## Reproducibilidad

`python -m src.main entrenar` guarda los cinco modelos en `models/` con nombres fijos (`calidad_random_forest.joblib`, `calidad_cnn.keras`, `fruta_random_forest.joblib`, …) y los resúmenes en `results/entrenamiento_clasicos.json` y `results/entrenamiento_cnn.json`. Los modelos se suben a Git para que la app funcione sin reentrenar; si alguno pasa de 100 MB, el entrenamiento lo avisa. Los notebooks [02](../notebooks/02_modelos_clasicos.ipynb) y [03](../notebooks/03_cnn.ipynb) leen esos resúmenes.
