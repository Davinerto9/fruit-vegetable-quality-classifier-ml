# 4. Modelado

## Objetivo y datos de entrada

Se clasifican recortes de una pieza en `good`, `regular` o `bad`. Todos los modelos vigentes se entrenaron y compararon sobre el **split v2 por ráfagas** (`data/processed/crops/splits_v2_burst.csv`, sha256 `b0ac08f52b3871061d17b6629b3ec3815640afd4c93a3cecf3e78bb5eac7abc9`): 16.272 recortes de train y 3.795 de validation en 852 grupos. Test no se ha usado.

Los modelos clásicos reciben 359 atributos de la pieza enmascarada: color RGB/HSV, textura LBP y forma. La CNN recibe recortes RGB de 128 × 128 con el fondo anulado mediante la máscara. Las ecuaciones y la justificación de cada modelo están en los [fundamentos matemáticos](supporting_docs/mathematical_and_theoretical_foundations.md#6-modelos).

> **Por qué v2.** El split v1 tenía una fuga por ráfagas en Kaggle (ver [preparación](03_data_preparation.md#fuga-detectada-en-el-split-v1-y-split-v2-vigente)). Las cifras de v1 (RF 0,842, CNN 0,799) ya no son válidas como resultado y solo se conservan para comparar.

## Modelos candidatos

| Modelo | Configuración | Selección | F1 macro val v2 | `bad→good` | Checkpoint |
| --- | --- | --- | ---: | ---: | --- |
| Regresión logística (línea base) | `StandardScaler`, C=1, pesos balanceados | Fija | 0,714 | 165 | `logistic_baseline_v2_burst.joblib` |
| SVM lineal | `LinearSVC`, C=10 | CV agrupada, 3 valores de C | 0,717 | 173 | `svm_v2_burst.joblib` |
| Random Forest | 160 árboles, `min_samples_leaf=5`, `max_depth=20`, `max_features=0.1` | CV agrupada, 30 configuraciones, regla fijada antes | 0,857 | 121 | `random_forest_v2_burst_grid.joblib` |
| CNN E3b | 4 bloques (32-64-128-256), 389.667 parámetros, ReduceLROnPlateau, EarlyStopping (paciencia 12); mejor época 38 de 50 | Validation, entre E1, E2 y E3b | 0,858 | 90 | `cnn_rgb_v2_burst_e3b.keras` |
| Ensamble RF + CNN E3b | Voto suave 0,5/0,5, peso fijado antes de evaluar | — | **0,875** | **79** | Usa los dos anteriores |

Los checkpoints viven en `experiments/checkpoints/` y no se suben a Git por tamaño. Los JSON de resultados están en `experiments/results/` y los diagnósticos en `experiments/diagnostics/out/`.

## Experimentos de la CNN

Cada experimento cambia una sola variable respecto al anterior aceptado.

| ID | Cambio | F1 val v2 | Decisión | Lectura |
| --- | --- | ---: | --- | --- |
| Baseline v2 | Misma configuración que v1 | 0,748 | — | EarlyStopping cortó en la época 23 por oscilaciones de la pérdida. |
| E1 | 100 épocas, ReduceLROnPlateau, paciencia 12 | 0,797 | Aceptado | Converge (mejor época 52 de 64) y no sobreajusta: el límite era de capacidad. |
| E2 | E1 + 8 simetrías del cuadrado en train | 0,746 | Rechazado | Regularizar un modelo que no sobreajusta empeora (Δ −0,051 [−0,073; −0,031]). |
| E3a | 3 bloques con 64-128-256 filtros | — | Detenida | 197 s por época: demasiado costosa. |
| E3b | Cuarto bloque (32-64-128-256) | 0,858 | **Candidata** | Δ +0,060 [+0,036; +0,084] frente a E1. El campo receptivo pasa de 22 a 46 px. Sobreajusta desde la época 30, pero EarlyStopping lo contiene. |

## Lectura provisional

- **RF y la CNN E3b empatan** en F1 macro (Δ +0,001 [−0,027; +0,028]). La CNN comete un 26% menos de errores `bad→good`.
- **El ensamble** mejora todos los indicadores de riesgo: recall de `bad` 0,926, `bad→good` 79 y `regular→good` 115. Frente a RF, la diferencia global no es significativa (Δ +0,018 [−0,004; +0,042]), pero sí lo es en AgriFreshNET (+0,044 [+0,012; +0,077]).
- **Por fuente**, el F1 del ensamble es 0,949 en Kaggle (solo `good`/`bad`) y 0,843 en AgriFreshNET (tres clases). La segunda es la cifra representativa del problema de tres clases.
- **Sesgo de selección:** la CNN se eligió en validation entre tres variantes, y validation también se usó para EarlyStopping. Su 0,858 y el 0,875 del ensamble son algo optimistas.

## Decisión pendiente (antes de test)

El equipo debe fijar por escrito el modelo final entre RF solo, CNN E3b sola y ensamble 0,5/0,5. La recomendación es el **ensamble 0,5/0,5 fijo**, justificado por el costo del error `bad→good` más que por F1. La app ya está preparada para las tres opciones (`config.APP_MODEL`). Antes de decidir conviene cerrar la prueba de ráfagas en AgriFreshNET ([preparación](03_data_preparation.md)), porque la ventaja del ensamble está justamente en esa fuente.

## Reproducibilidad

- Clásicos, con scikit-learn 1.7.2 y `S=data/processed/crops/splits_v2_burst.csv`, `F=data/processed/classic_features_v2_burst.npz`:
  - `python -m src.features.subset_classic_cache --split $S --output $F`
  - `python -m src.training.classical_models --features $F --split $S --tag v2_burst`
  - `python -m src.training.rf_grid --features $F --split $S --tag v2_burst_grid`
- CNN E3b: `python -m src.training.train_cnn --split data/processed/crops/splits_v2_burst.csv --run-name v2_burst_e3b --epochs 100 --patience 12 --reduce-lr-patience 4 --filters 32,64,128,256 --cache`, en GPU con `TF_DETERMINISTIC_OPS=1`.
- Ensamble: `experiments/diagnostics/eval_ensemble_v2.py`.
- Seguimiento completo de los experimentos: [modeling_fronts_tracking.md](modeling_fronts_tracking.md).
