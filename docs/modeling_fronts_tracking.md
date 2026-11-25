# Seguimiento de los cinco frentes de modelado

**Responsables:** equipo de modelado (CNN, Random Forest, sesgo de fuente, fotos propias, sobreajuste).
**Fecha del diagnóstico:** 24 de noviembre de 2026, sobre el commit `ad0b21f` de `main`.
**Test:** no se usó. No existen `pretest_freeze.sha256`, `test_evaluation.lock` ni salidas `final_test_*`. Ningún script de entrenamiento lee filas de test. Solo se consultaron conteos de metadatos de test, para auditar el split.

Scripts y salidas: [experiments/diagnostics/](../experiments/diagnostics/README.md). Reproducción de validation: [validation_audit_repro.txt](../experiments/diagnostics/out/validation_audit_repro.txt).

## Tabla de seguimiento

| Frente | Estado | Evidencia principal | Siguiente paso | Requiere acuerdo |
|---|---|---|---|---|
| Fix CNN | **E3b candidata** (E1 aceptado, E2 rechazado, E3a detenida por costo) | Baseline v2 0.748 → E1 0.797 → **E3b (4 bloques) 0.8575** (Δ frente a E1 +0.060, IC [+0.036, +0.084]). Empata con el RF (Δ +0.001, IC [−0.027, +0.028]) con menos `bad→good` (90 frente a 121) y recall de bad 0.916. Hay señal de sobreajuste (loss train 0.095 frente a val 0.334), contenida por EarlyStopping. | Ensamble 0,5/0,5 probado: val 0.875, `bad→good` 79 (Δ frente al RF +0.018 [−0.004, +0.042]; frente a la CNN +0.018 [+0.008, +0.028]). Elegir RF, CNN o ensamble antes de test | Sí (equipo, incluida la GUI) |
| Fix Random Forest | Candidato elegido por CV agrupada sobre v2 | Base v2: CV 0.810, val 0.840. Rejilla de 30 configuraciones; elegida leaf=5, depth=20, `max_features`=0.1: CV 0.816, val 0.857 (Δ +0.017, IC [+0.005, +0.028]), F1 regular 0.783, `bad→good` 121. La brecha train–CV sigue en ~0.18. | Variante LBP uniform (opcional); fijar el candidato antes del freeze | No |
| `regular` solo en AgriFreshNET | Confusor confirmado. El atajo no está demostrado. | El 100% de `regular` no propio es de AgriFreshNET. La sonda de fuente alcanza 0.976 de balanced accuracy. La transferencia entre dominios cae de AUC 0.98 a 0.72. | Reportar las métricas por fuente y conseguir `regular` de otra fuente | Recolección de datos |
| Pocas fotos propias en validation | Confirmado, con corrección parcial a la auditoría | Validation: 1 recorte (1 grupo). Test: 4 recortes en 3 grupos (la auditoría decía 0). Train: 32 recortes en 4 grupos. | Protocolo de recolección (abajo) | Recolección de datos |
| Riesgo de overfitting y fuga | **Fuga confirmada en Kaggle; split v2 generado** | `splits.csv` usa grupos por archivo, no por ráfaga. El 99% de las filas Kaggle de validation comparte ráfaga con train. Con CV agrupada por ráfaga, el F1 Kaggle baja de 0.968 a 0.932. El split v2 no tiene ninguna ráfaga repartida. | Reentrenar los modelos sobre v2 con checkpoints versionados | Sí (cómputo de la CNN) |

Un frente se marca cerrado solo cuando cumple su criterio de aceptación con evidencia. Por ahora no hay ninguno cerrado.

## F0. Fuga por ráfagas de Kaggle (hallazgo nuevo, bloqueante)

- **Problema confirmado (error de implementación del pipeline).** El commit `90f498a` agrupa las fotos de ryandpark por ráfaga de captura en `src/data/inventory.py`. El `index.csv` actual sí tiene esos grupos: 746 ráfagas `ryandpark:capture:*`. Sin embargo, `crops/manifest.csv` y `splits.csv` conservan los grupos anteriores, uno por archivo, en las 19.526 filas de Kaggle. Lo más probable es que el manifiesto se haya generado desde un `index.csv` previo al reagrupamiento. Los tiempos de modificación son 12:09 para el índice y 12:11 para el manifiesto, y `prepare.py` copia la columna `group` del índice.
- **Evidencia.** 645 de las 746 ráfagas aparecen en más de una partición. 1.539 de los 1.553 recortes Kaggle de validation, y 1.600 de los 1.615 de test, pertenecen a una ráfaga que también tiene recortes en train. Ejemplo: `Banana_Bad/IMG_7702…7705` quedó repartido entre train, val y test.
- **Impacto medido sin tocar test.** Con la misma configuración de RF y StratifiedGroupKFold de 3 folds sobre train:

  | Agrupamiento | F1 macro global | F1 Kaggle (good/bad) | F1 AgriFreshNET (3 clases) |
  |---|---:|---:|---:|
  | Por archivo (split actual) | 0.8196 ± 0.0002 | 0.9675 ± 0.0020 | 0.7551 ± 0.0005 |
  | Por ráfaga (`index.csv`) | 0.8079 ± 0.0059 | 0.9319 ± 0.0113 | 0.7582 ± 0.0076 |

  La cifra de Kaggle está inflada en unos 3,5 puntos. AgriFreshNET no cambia.
- **Corrección propuesta (requiere acuerdo).** Generar un split v2 en una ruta nueva, sin sobrescribir `splits.csv`. Para eso, se toma la columna `group` del `index.csv` vigente, se aplican las mismas exclusiones y se ejecuta el mismo `assign_groups`. No hace falta repetir la segmentación. Después hay que reentrenar los cuatro modelos sobre v2 con checkpoints nuevos. Como test nunca se evaluó, rehacer el split no introduce sesgo de selección. Se conservan el split v1 y sus checkpoints para comparar.
- **Split v2 generado (24-nov-2026, aprobado por el equipo).** Lo produce `python -m src.data.split_v2`, que usa el mismo `assign_groups`, la semilla 42 y las mismas exclusiones visuales. Se escribe en `data/processed/crops/splits_v2_burst.csv` (sha256 `b0ac08f52b3871061d17b6629b3ec3815640afd4c93a3cecf3e78bb5eac7abc9`) y conserva la columna `group_v1`. `splits.csv` (v1, `5b37044b…`), `manifest.csv` e `index.csv` no cambiaron. Comparación completa: [out/compare_splits.txt](../experiments/diagnostics/out/compare_splits.txt).

  | | v1 | v2 |
  |---|---:|---:|
  | Recortes | 23.687 | 23.543 (144 menos: ráfagas completas con un archivo excluido) |
  | Grupos | 13.192 | 5.677 |
  | train / val / test | 69,5 / 15,0 / 15,5 % | 69,1 / 16,1 / 14,8 % |
  | Ráfagas Kaggle en más de una partición | 645 | **0** |
  | Grupos Kaggle en val (bad / good) | 586 / 645 | 48 / 77 |
  | Recortes Kaggle en val (bad / good) | 769 / 784 | 1.105 / 770 |
  | AgriFreshNET en val (good / regular / bad) | 620 / 705 / 678 | 617 / 709 / 590 |
  | Fotos propias en val / test | 1 / 4 | 4 (3 grupos) / 1 |

  Consecuencias:
  - En v2, Kaggle aporta solo 125 grupos independientes a validation. Las métricas de Kaggle tendrán más varianza, pero ese es el tamaño efectivo real de la muestra.
  - Agrupar por minuto puede unir frutas distintas tomadas en el mismo minuto. Es una decisión conservadora: puede reducir datos, pero no introduce fuga.
  - Test v2 tampoco se ha evaluado.
  - **Los checkpoints actuales se entrenaron con v1 y no se pueden evaluar en validation v2**, porque 1.539 recortes Kaggle de v1 quedaron en train y esas ráfagas pueden aparecer en val v2. Hay que reentrenar.
  - `classical_models.py` exige que el caché de atributos coincida exactamente con el split, así que hay que filtrarlo a las filas de v2. No hace falta reextraer atributos.

- **Modelos clásicos reentrenados sobre v2 (24-nov-2026).** Se usó el mismo código, la misma rejilla y la misma semilla; solo cambió el split. El caché de atributos se derivó sin reextraer con `python -m src.features.subset_classic_cache`, y `classical_models.py --tag v2_burst` guardó salidas versionadas. Checkpoints: `logistic_baseline_v2_burst.joblib`, `svm_v2_burst.joblib` y `random_forest_v2_burst.joblib`. Los checkpoints v1 conservan su hash. Entrenamiento con scikit-learn 1.7.2 (log en `out/train_classic_v2_burst.log`) y evaluación con `experiments/diagnostics/eval_classic_split.py`.

  | Modelo | Versión | CV F1 (train, agrupada) | F1 macro val | Recall bad | `bad→good` | F1 regular | Brecha train/val | F1 Kaggle (2 cl.) | F1 AgriFreshNET (3 cl.) |
  |---|---|---:|---:|---:|---:|---:|---:|---:|---:|
  | Logística | v1 | — | 0.6845 | 0.689 | 149 | 0.560 | 0.064 | 0.878 | 0.570 |
  | Logística | v2 | — | 0.7140 | 0.789 | 165 | 0.590 | 0.026 | 0.871 | 0.616 |
  | SVM lineal (C=10) | v1 | 0.7041 | 0.6978 | 0.750 | 155 | 0.545 | 0.049 | 0.895 | 0.600 |
  | SVM lineal (C=10) | v2 | 0.6755 | 0.7169 | 0.820 | 173 | 0.573 | 0.020 | 0.881 | 0.628 |
  | Random Forest (leaf=2) | v1 | 0.8196 | 0.8424 | 0.868 | 114 | 0.756 | 0.158 | 0.968 | 0.779 |
  | Random Forest (leaf=2) | v2 | 0.8103 | 0.8402 | 0.903 | 120 | 0.733 | 0.160 | 0.961 | 0.777 |

  IC 95% del F1 de RF por bootstrap de grupos (`out/bootstrap_ci.txt`):
  - v2: global 0.840 [0.812, 0.865]; Kaggle 0.961 [0.936, 0.980], con 122 grupos; AgriFreshNET 0.777 [0.745, 0.809].
  - En v1, el IC de Kaggle ([0.959, 0.977]) es artificialmente estrecho, porque cuenta cada foto de una ráfaga como un grupo independiente.

  **Lectura.**
  - Las filas de val v1 y v2 son conjuntos distintos, así que la diferencia en val no es una mejora ni un deterioro del modelo.
  - Lo comparable es la CV agrupada de train: en RF baja de 0.820 a 0.810, en línea con la inflación estimada antes. En val v2, Kaggle queda en 0.961, pero con un IC tres veces más ancho.
  - RF elige la misma configuración en v1 y v2 y conserva la brecha train/val (~0.16): memoriza train.
  - La logística y la SVM mejoran en val v2, con brechas menores. Es plausible que se deba a la composición de val v2: menos AgriFreshNET difícil (eggplant, papaya) y más `bad` de Kaggle. No es evidencia de que los modelos sean mejores.
  - **Estas cifras v2 pasan a ser el baseline honesto de los modelos clásicos.** La CNN v2 sigue pendiente, y sin ella el audit de RF y CNN sobre v2 no se puede hacer.

- **Pendiente.** AgriFreshNET también tiene nombres con fecha y hora (`IMG_20260114_010528`), y no se agrupa por ráfaga. Es una hipótesis por comprobar con el mismo método de CV.

## Frente 1. CNN

**Configuración real** (`src/training/train_cnn.py`): 3 bloques Conv3×3 (32, 64, 128 filtros), con BatchNorm, ReLU y MaxPool en cada uno; después GlobalAveragePooling, Dropout 0.35 y Dense softmax. Usa Adam con lr 1e-3, batch 64, 40 épocas y `class_weight` balanceado calculado sobre train. La entrada es RGB de 128×128 con el fondo anulado por la máscara. **Sin data augmentation**, sin scheduler y sin L2. Hay EarlyStopping con paciencia 6 sobre `val_loss` y ModelCheckpoint con `save_best_only`. Semilla 42.

**Verificado.**
- El mejor checkpoint es el de la época 40 de 40 (`val_loss` 0.4498; en la época 39 fue 0.5152). EarlyStopping nunca se activó.
- En la época 40, train tuvo loss 0.435 y accuracy 0.821. La accuracy de train se mide con dropout activo y la pérdida está ponderada por clase, así que no se compara directamente con validation.
- `val_loss` es muy inestable: 0.46 → 0.66 → 0.46 → 0.68 entre las épocas 31 y 36. Esto es compatible con un lr fijo alto y con las estadísticas de BatchNorm.
- La CNN comete menos `bad→good` que RF: 89 (6,2% de `bad`) frente a 114 (7,9%).

**Interpretación.** No hay señales de sobreajuste estadístico: train no se separa de val y ambas pérdidas bajan. Hay evidencia de que el entrenamiento no convergió. Que la mejor época sea la última no basta para afirmar subentrenamiento: también podría ser ruido de una curva tan oscilante.

**Errores de implementación encontrados.**
- `--evaluate-checkpoint` sobrescribe `experiments/results/cnn_rgb.json`. Se perdieron `fit_seconds`, `class_weight` e historial, y nunca se versionaron.
- `train()` escribe siempre en `cnn_rgb.keras`. Cualquier experimento nuevo destruiría el baseline.

**Experimentos propuestos, de uno en uno, sobre el mismo split** (cada uno cambia una sola variable respecto al anterior aceptado):

| ID | Hipótesis | Cambio | Criterio de aceptación |
|---|---|---|---|
| E0 | (infraestructura) | Añadir `--run-name` con salidas versionadas y que la evaluación escriba `cnn_rgb_eval.json` | Reproduce 0.7988 con el checkpoint original |
| E1 | El modelo no convergió | 100 épocas, ReduceLROnPlateau (factor 0.5, paciencia 4), EarlyStopping con paciencia 12 | Mejor época antes del final, `val_loss` menos oscilante, F1 val ≥ baseline + 0.01 |
| E2 | Falta invarianza geométrica | E1 + augmentation solo geométrica (flip, rotación de 90°, zoom leve). Sin cambios de tono ni saturación. Brillo ±10% como mucho, en una variante aparte | F1 val y recall `bad` mejoran sin aumentar `bad→good` |
| E3 | Capacidad o learning rate | Rejilla pequeña: lr {1e-3, 3e-4} × dropout {0.25, 0.5} | Elegir con F1 val y reportar la tabla completa, incluidos los resultados negativos |

**E0 hecho (24-nov-2026).** `train_cnn.py --run-name X` guarda `cnn_rgb_X.keras`, `.json` y los logs `_epochs.csv` y `_history.json`, y falla si ya existen. También registra los segundos por época. Sin `--run-name`, las rutas de entrenamiento no cambian. `--evaluate-checkpoint` ahora escribe en `<checkpoint>_eval.json` y no sobrescribe el JSON de entrenamiento.

**Costo medido** (`experiments/diagnostics/out/cnn_timing_v2.txt`, split v2, 2 épocas):
- La GPU sí funciona si se exporta `LD_LIBRARY_PATH=/usr/lib/wsl/lib:<venv>/lib/python3.11/site-packages/nvidia/*/lib`, porque las librerías CUDA de pip ya están instaladas. Sin esa variable, TensorFlow cae a CPU.
- **GPU: 73–129 s por época. CPU: 335–352 s por época.** El tiempo de sistema alto sugiere que el cuello es la lectura de PNG desde `/mnt/c`.
- Las corridas solo son comparables si usan el mismo dispositivo y `TF_DETERMINISTIC_OPS=1`.

Estimación en GPU, a unos 100 s por época:

| Corrida | Épocas máx. | Horas aprox. |
|---|---:|---:|
| Baseline v2 (misma configuración que v1) | 40 | 1,1 |
| E1 | 100 | ≤ 2,8 (menos si para antes) |
| E2 | 100 | ≤ 2,8 |
| E3 (4 combinaciones) | 100 c/u | ≤ 11 |

En CPU, todo tarda unas 3,4 veces más.

**Caché en memoria (hecho).** `train_cnn.py --cache` guarda en memoria los PNG decodificados (uint8) y aplica el redimensionado y la máscara después. `experiments/diagnostics/check_cnn_cache.py` comprobó que los tensores y el orden son idénticos a los de la versión sin caché: 700 recortes, 2 pasadas, con y sin shuffle, máx|dif| = 0. **Una época en GPU pasa de ~100 s a ~22–26 s** (mediana 25,8 s). Además, el entrenamiento es determinista entre corridas: las épocas 1 y 2 reprodujeron exactamente las de la medición previa (`val_loss` 1.1443 y 0.8502). Con esto, E1 cuesta ≤ 45 min y E3 ≤ 3 h.

**Baseline CNN v2** (`cnn_rgb_v2_burst_baseline.keras`, misma configuración que v1, GPU, `--cache`). Log en `experiments/diagnostics/out/train_cnn_v2_burst_baseline.log`.
- **EarlyStopping lo detuvo en la época 23; la mejor fue la 17** (`val_loss` 0.515). La pérdida de train seguía bajando (0.835 → 0.516) y `val_loss` oscila mucho (0.69 → 1.33 en la época 6; 0.54 → 0.74 en la 20). Con paciencia 6, esas oscilaciones cortan el entrenamiento antes de converger.
- Esto refuerza la hipótesis de E1: el problema es la optimización (lr fijo, paciencia corta), no el sobreajuste.

  Comparación en val v2, mismas filas (`eval_cnn_vs_rf_v2.py`, `out/cnn_vs_rf_v2_burst_baseline.txt`):

  | Modelo | Versión | F1 macro val | Recall bad | `bad→good` | F1 regular | `regular→good` | `regular` predicho en Kaggle | Kaggle | AgriFreshNET | Mejor época |
  |---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
  | CNN RGB | v1 (val v1) | 0.7988 | 0.837 | 89 | 0.702 | 72 | 1 | 0.944 | 0.713 | 40/40 |
  | CNN RGB | v2 baseline | 0.7478 | 0.807 | 200 | 0.621 | 186 | 12 | 0.913 | 0.648 | 17/23 |
  | RF | v2_burst_grid | 0.8569 | 0.897 | 121 | 0.783 | 122 | 1 | 0.954 | 0.799 | — |

  Δ F1 (CNN − RF) con bootstrap pareado por grupos:
  - Global: −0.109, IC [−0.133, −0.085].
  - Kaggle: −0.040, IC [−0.073, −0.013].
  - AgriFreshNET: −0.152, IC [−0.183, −0.120].

  Errores por pareja: ambos aciertan 2.824; solo falla la CNN en 497; solo falla RF en 123; ambos fallan en 351.

  Las filas v1 y v2 son distintas, así que la caída de la CNN de 0.799 a 0.748 mezcla dos efectos: el split sin fuga y una corrida que se detuvo en la época 23 en vez de llegar a la 40. En v2, la CNN ya **no** comete menos `bad→good` que RF (200 frente a 121). El argumento de la auditoría a favor de la CNN no se sostiene con el split corregido.

**E1 ejecutado (24-nov-2026), aceptado.** `cnn_rgb_v2_burst_e1.keras`. Respecto al baseline v2 cambia una sola cosa, la optimización: `--epochs 100 --patience 12 --reduce-lr-patience 4 --reduce-lr-factor 0.5 --min-lr 1e-5`. Corrió en GPU con `--cache` y `TF_DETERMINISTIC_OPS=1`, en 30 min. Log en `out/train_cnn_v2_burst_e1.log`.
- **Curvas.** Las épocas 1 a 20 son idénticas a las del baseline (determinismo). El learning rate baja en las épocas 21, 26, 38, 44, 56, 60 y 64, y desde la 40 `val_loss` se estabiliza en 0.41–0.43. **La mejor época fue la 52 de 64**, con `val_loss` 0.412 y lr 6.25e-5. EarlyStopping se activó por falta de mejora, no por oscilaciones.
- **Sobreajuste.** En la época 52, train tiene loss 0.380 y accuracy 0.846 (con dropout y pérdida ponderada); val tiene loss 0.412 y accuracy 0.817. Las curvas siguen juntas: no hay sobreajuste estadístico. El techo de ~0.41 sugiere un límite de capacidad o de señal, no de optimización.

  | Modelo | Versión | F1 macro val | Recall bad | `bad→good` | F1 regular | `regular→good` | `regular` en Kaggle | Kaggle | AgriFreshNET | Mejor época |
  |---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
  | CNN RGB | v2 baseline | 0.7478 | 0.807 | 200 | 0.621 | 186 | 12 | 0.913 | 0.648 | 17/23 |
  | CNN RGB | **v2 E1** | **0.7973** | 0.870 | 135 | 0.713 | 97 | 1 | 0.908 | 0.729 | 52/64 |
  | RF | v2_burst_grid | 0.8569 | 0.897 | 121 | 0.783 | 122 | 1 | 0.954 | 0.799 | — |

  Δ pareados con bootstrap de grupos (`out/paired_cnn_e1_vs_baseline.txt`, `out/cnn_vs_rf_v2_burst_e1.txt`):
  - E1 − baseline: global **+0.050 [+0.029, +0.071]**; AgriFreshNET +0.081 [+0.051, +0.112]; Kaggle −0.006 [−0.026, +0.014].
  - E1 − RF: global −0.060 [−0.085, −0.034]; Kaggle −0.046 [−0.086, −0.013]; AgriFreshNET −0.070 [−0.103, −0.038].
  - Por pareja: ambos aciertan 2.915; solo falla la CNN en 406; solo falla RF en 187; ambos fallan en 287.

- **Decisión.** E1 cumple el criterio de aceptación: la mejor época llega antes del final, la mejora es mayor que 0.01 con un IC que excluye el 0, y `bad→good` baja de 200 a 135. **E1 pasa a ser la configuración base para E2.** La CNN sigue por debajo del RF, de forma significativa. E1 comete menos `regular→good` que RF (97 frente a 122), pero algo más `bad→good` (135 frente a 121).

**E2 ejecutado (24-nov-2026), rechazado (resultado negativo).** Es E1 más `--augment`: en train, cada imagen recibe una de las 8 simetrías del cuadrado (flip horizontal, flip vertical, transposición) con semilla stateless. No hay interpolación ni cambios de color, brillo o saturación. `experiments/diagnostics/check_cnn_augment.py` comprobó que:
- Las 512 imágenes son exactamente una simetría de su original.
- Las 8 simetrías aparecen con la misma frecuencia (57–70).
- Las etiquetas y la suma de píxeles se conservan.
- Es reproducible, y con shuffle la simetría de cada imagen cambia entre épocas (402 de 409).

Duró 28 min y la mejor época fue la 48 de 60.

  | Modelo | Versión | F1 macro val | Recall bad | `bad→good` | F1 regular | `regular→good` | Kaggle | AgriFreshNET | Loss train / val (mejor época) |
  |---|---|---:|---:|---:|---:|---:|---:|---:|---|
  | CNN RGB | v2 E1 | **0.7973** | 0.870 | 135 | 0.713 | 97 | 0.908 | 0.729 | 0.380 / 0.412 |
  | CNN RGB | v2 E2 (E1 + simetrías) | 0.7461 | 0.835 | 198 | 0.636 | 139 | 0.903 | 0.645 | 0.554 / 0.529 |

  Δ E2 − E1 (bootstrap pareado por grupos, `out/paired_cnn_e2_vs_e1.txt`): global **−0.051 [−0.073, −0.031]**; AgriFreshNET −0.084 [−0.114, −0.054]; Kaggle −0.004 [−0.032, +0.023]. La caída se concentra en los productos de AgriFreshNET: berenjena 0.696 → 0.531, piña 0.711 → 0.562, tomate 0.738 → 0.629, bitter melon 0.739 → 0.649, pepino 0.692 → 0.619. Naranja y manzana casi no cambian.

  **Interpretación.**
  - *Comprobado:* la augmentation empeoró el ajuste de train (loss 0.38 → 0.55). Como E1 no sobreajustaba, regularizar más le quitó capacidad útil.
  - *Hipótesis no comprobada:* en AgriFreshNET la orientación de la pieza o la dirección de la luz y los brillos es constante y aporta información. Puede ser una señal real de la forma de captura o un atajo propio del dataset; el resultado de E2 no distingue entre ambas.

  **Decisión.** Se conserva E1 como CNN candidata. No se prueban más variantes de augmentation geométrica. La rejilla E3 (lr × dropout) apunta a la regularización y tiene poca probabilidad de ayudar sin sobreajuste: se propone sustituirla por una prueba de **capacidad** (más filtros o un cuarto bloque).

**E3 de capacidad (24-nov-2026).** Usa la configuración de E1 (sin augmentation) y solo cambia `--filters`.
- **E3a (64-128-256): detenida por costo.** Tardaba 197 s por época en la RTX 3050, lo que suponía 3–5 h. Se paró en la época 3, sin resultado. Las épocas parciales quedan en `out/cnn_e3a_partial_epochs.csv`.
- **E3b (32-64-128-256, cuarto bloque): aceptada.** `cnn_rgb_v2_burst_e3b.keras`, con 389.667 parámetros entrenables; el modelo de 3 bloques tiene unos 94 mil (cálculo a partir de la arquitectura). Duró 55 min (mediana de 73 s por época) y la mejor época fue la **38 de 50**. Log en `out/train_cnn_v2_burst_e3b.log`.

  | Modelo | Versión | F1 macro val | Recall bad | `bad→good` | F1 regular | `regular→good` | `regular` en Kaggle | Kaggle | AgriFreshNET | Loss train / val (mejor época) |
  |---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
  | CNN RGB | v2 E1 (3 bloques) | 0.7973 | 0.870 | 135 | 0.713 | 97 | 1 | 0.908 | 0.729 | 0.380 / 0.412 |
  | CNN RGB | **v2 E3b (4 bloques)** | **0.8575** | **0.916** | **90** | 0.776 | 128 | 3 | 0.938 | **0.823** | 0.095 / 0.334 |
  | RF | v2_burst_grid | 0.8569 | 0.897 | 121 | 0.783 | 122 | 1 | 0.954 | 0.799 | 0.993 (F1 train) |

  Δ pareados con bootstrap de grupos (`out/paired_cnn_e3b_vs_e1.txt`, `out/cnn_vs_rf_v2_burst_e3b.txt`):
  - E3b − E1: global **+0.060 [+0.036, +0.084]**; Kaggle +0.030 [+0.007, +0.054]; AgriFreshNET +0.094 [+0.060, +0.127]. Mejora en 11 de 12 productos; la mayor subida es en pepino (0.69 → 0.91).
  - E3b − RF: global **+0.001 [−0.027, +0.028]**, un empate estadístico. Kaggle −0.016 [−0.049, +0.011]; AgriFreshNET +0.023 [−0.014, +0.061]. Ninguna diferencia es significativa.
  - Por pareja: ambos aciertan 3.045; solo falla RF en 286; solo falla la CNN en 276; ambos fallan en 188. Los errores se solapan poco (Jaccard 0.25), así que un ensamble podría ayudar.

  **Interpretación.**
  - *Comprobado:* la limitación de E1 era de capacidad. Con un cuarto bloque, el mapa que llega al pooling global es de 8×8 y el campo receptivo pasa de 22×22 a 46×46 px (36% del recorte) y `val_loss` baja de 0.41 a 0.33.
  - *Nuevo riesgo:* **ahora sí hay señal de sobreajuste estadístico.** Desde la época 30, la pérdida de train sigue bajando (0.12 → 0.08) mientras `val_loss` oscila entre 0.33 y 0.50 sin mejorar, y la brecha de loss en la mejor época es 0.095 frente a 0.334. EarlyStopping con `restore_best_weights` lo contiene.
  - Para E3b, la regularización (más dropout o L2) y la augmentation vuelven a ser hipótesis razonables. El rechazo de E2 se midió sobre un modelo que no sobreajustaba y no se traslada automáticamente a E3b.
  - *Sesgo de selección:* entre las CNN se eligió por validation (E1, E2 y E3b son tres comparaciones), así que 0.8575 es algo optimista. Cualquier variante nueva aumenta ese sesgo. Para limitarlo, se propone cerrar la búsqueda de CNN con a lo sumo una variante más.

  **Decisión.** **E3b pasa a ser la CNN candidata**: empata con el RF en F1 macro y comete un 26% menos de errores `bad→good` (90 frente a 121), con recall de `bad` 0.916. RF y CNN quedan como candidatos finales; el criterio de desempate acordado con el equipo (por ejemplo, `bad→good` o un ensamble) debe fijarse **antes** de abrir test.

## Ensamble RF + CNN E3b (validation v2, 24-nov-2026)

**Regla fijada antes de ver resultados:** voto suave con pesos iguales, p = 0,5·p_RF + 0,5·p_CNN, y la clase predicha es el argmax. No se ajustó nada en validation. Script: `experiments/diagnostics/eval_ensemble_v2.py`; salida en `out/ensemble_rf_cnn_e3b.txt` y `experiments/results/ensemble_rf_v2_burst_grid_cnn_v2_burst_e3b.json`.

| Modelo | F1 macro val | Recall bad | `bad→good` | F1 regular | `regular→good` | Kaggle | AgriFreshNET |
|---|---:|---:|---:|---:|---:|---:|---:|
| RF v2_burst_grid | 0.8569 | 0.897 | 121 | 0.783 | 122 | 0.954 | 0.799 |
| CNN v2 E3b | 0.8575 | 0.916 | 90 | 0.776 | 128 | 0.938 | 0.823 |
| **Ensamble 0,5/0,5** | **0.8751** | **0.926** | **79** | **0.799** | **115** | 0.949 | **0.843** |

Δ pareados con bootstrap de grupos (IC 95%):
- Ensamble − RF: global +0.018 [−0.004, +0.042], que **no es significativo**; AgriFreshNET +0.044 [+0.012, +0.077]; Kaggle −0.005 [−0.027, +0.015].
- Ensamble − CNN: global +0.018 [+0.008, +0.028]; Kaggle +0.011 [+0.001, +0.025]; AgriFreshNET +0.020 [+0.009, +0.033]. Es significativo en los tres cortes.

Los modelos coinciden en el 84,3% de las fotos, y ahí la accuracy es 0.951. En las 594 fotos donde discrepan, el ensamble acierta el 58,4%, frente al 46,5% del RF y el 48,1% de la CNN.

**Sensibilidad al peso de la CNN (solo descriptiva; no se usa para elegir):** con w = 0 / 0,1 / 0,2 / 0,3 / 0,5 / 0,7 / 1, el F1 es 0.857 / 0.877 / 0.885 / 0.883 / 0.875 / 0.866 / 0.858 y `bad→good` es 121 / 96 / 87 / 76 / 79 / 85 / 90. Todos los pesos intermedios superan a cada modelo por separado. Que el máximo esté con menos peso para la CNN es compatible con que sus probabilidades sean más extremas que las del RF: el peso nominal no equivale a la influencia real de cada modelo. Elegir w = 0,2 con esta curva sería seleccionar sobre validation; si se quiere ajustar el peso, hay que hacerlo con predicciones fuera de fold en train (CV agrupada) o calibrando antes de combinar.

**Interpretación.** El ensamble mejora todos los indicadores de riesgo a la vez: el menor número de `bad→good` (79), el mayor recall de `bad` (0.926) y el menor número de `regular→good` (115). Eso es coherente con errores poco correlacionados (Jaccard 0.25). Frente a la CNN la ganancia es significativa; frente al RF es probable, pero el IC global incluye el 0. La mejora se concentra en AgriFreshNET, y en Kaggle no hay diferencia.

**Costo para la app (equipo de GUI).** Hay que ejecutar segmentación, extracción de atributos y RF, más la CNN, por cada foto, y ambos checkpoints deben viajar juntos. Esto cambia el contrato de inferencia y hay que acordarlo con ese equipo.

**Decisión pendiente del equipo, antes de test:** elegir el modelo final entre (a) RF solo, (b) CNN E3b sola o (c) ensamble 0,5/0,5 fijo. Con la evidencia de validation, (c) es la opción con mejor F1 y menos errores caros. Si se elige (c), el peso queda fijo en 0,5 y no se ajusta después.

## Frente 2. Random Forest

**Configuración real del checkpoint:** `n_estimators=160`, `max_depth=None`, `min_samples_leaf=2` (lo eligió la CV), `min_samples_split=2`, `max_features='sqrt'` (unos 19 atributos de 359), `class_weight='balanced_subsample'`, `bootstrap=True` y sin OOB. Rejilla probada: 3 combinaciones con StratifiedGroupKFold de 3 folds. CV: 0.8115, **0.8196** y 0.8136.

**359 atributos, confirmado:** 6 canales (R, G, B, H, S, V) × 16 bins = 96, más LBP de 256 bins y 7 de forma. No se escalan (los árboles no lo necesitan) y se extraen por imagen sin parámetros ajustados, así que no hay fuga por preprocesamiento. Importancia por familia: LBP 0.49, H 0.12, G 0.09, B 0.085, S 0.075, R 0.05, V 0.05, forma 0.035.

**Reconstrucción de las cifras de la auditoría (validation):**

| Corte | n | Clases presentes | F1 macro sobre las clases presentes | F1 macro con las 3 clases |
|---|---:|---|---:|---:|
| Global | 3.557 | 3 | 0.8424 | 0.8424 |
| Kaggle | 1.553 | good, bad | 0.9684 | 0.6456 (regular sin soporte) |
| AgriFreshNET | 2.003 | 3 | 0.7785 | 0.7785 |
| Train | 16.456 | 3 | 0.9999 | — |

El 0.968 es un promedio sobre 2 clases y el 0.779 sobre 3, así que no son comparables directamente. Además, el 0.968 está inflado por F0.

**Errores por producto.** La fracción de recortes donde RF o CNN fallan es mayor en los productos exclusivos de AgriFreshNET: pepino 0.50, papaya 0.49, berenjena 0.47, tomate 0.41 y piña 0.41. Es menor en los de Kaggle: granada 0.04, lima 0.05 y guayaba 0.06.

**Rejilla agrupada sobre v2 (ejecutada el 24-nov-2026).** Se ejecuta con `python -m src.training.rf_grid --tag v2_burst_grid` y tardó 20 min en CPU. Log: `experiments/diagnostics/out/rf_grid_v2_burst.log`; el detalle por fold está en `experiments/results/random_forest_v2_burst_grid.json`.

- **Protocolo.** StratifiedGroupKFold de 3 folds con las ráfagas de v2, solo sobre train. Rejilla: `min_samples_leaf` {1, 2, 5, 10, 20} × `max_depth` {None, 20, 12} × `max_features` {sqrt (~19), 0.1 (~36)}, con 160 árboles. Regla fijada antes de ver resultados: entre las configuraciones a menos de 1 desviación estándar de la mejor CV, gana la de menos `bad→good` en CV y, si empatan, la de menor brecha train–CV. Validation v2 se usó una sola vez, para confirmar.
- **CV.** La mejor fue leaf=2, sin límite de profundidad y `max_features`=0.1, con 0.8185 ± 0.0070. Ocho configuraciones quedaron a menos de 1 desviación estándar; la regla eligió **leaf=5, `max_depth`=20, `max_features`=0.1** (CV 0.8161 ± 0.0090, 563 `bad→good` en CV). Usar 400 árboles no cambia nada (0.8186 frente a 0.8185), así que se queda en 160. Con `max_features`=0.1, el F1 de CV es mayor que con sqrt en las 15 combinaciones de leaf y profundidad. Con leaf ≥ 10 la brecha baja (0.12–0.17), pero cae el F1 (0.77–0.80).
- **Confirmación en val v2, con las mismas filas que el RF v2 base:**

  | Modelo | Versión | F1 macro val | Recall bad | `bad→good` | F1 regular | `regular→good` | F1 train | Profundidad / hojas (mediana) | Kaggle | AgriFreshNET |
  |---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
  | RF leaf=2, sqrt, sin límite | v2_burst | 0.8402 | 0.903 | 120 | 0.733 | 167 | 1.000 | 29 / 1432 | 0.961 | 0.777 |
  | RF leaf=5, 0.1, depth 20 | v2_burst_grid | **0.8569** | 0.897 | 121 | **0.783** | **122** | 0.993 | 20 / 818 | 0.954 | **0.799** |

  Bootstrap pareado por grupos con 2.000 réplicas (`out/paired_bootstrap_rf.txt`), Δ = grid − base:
  - Global: +0.017, IC [+0.005, +0.028].
  - AgriFreshNET: +0.022, IC [+0.007, +0.037].
  - Kaggle: −0.008, IC [−0.018, +0.002], que no es significativo.

- **Decisión: `random_forest_v2_burst_grid.joblib` pasa a ser el RF candidato.**
  - La mejora se concentra en `regular` (F1 +0.05; 45 `regular→good` menos) y en AgriFreshNET.
  - No empeora el error caro (`bad→good` 121 frente a 120).
  - Usa la mitad de hojas.
  - Limitaciones: la brecha train–CV sigue en torno a 0.18, así que el RF todavía memoriza. La mejora en val es una sola confirmación, no una estimación sin sesgo, porque la selección se hizo por CV. En Kaggle no hay mejora.

**Otros experimentos pendientes:** rejilla agrupada sobre `min_samples_leaf` {1, 2, 5, 10, 20} × `max_depth` {None, 20, 12} × `max_features` {sqrt, 0.1} × `n_estimators` {160, 400}. Criterio: CV agrupada por ráfaga. Como desempate, el menor `bad→good` y la menor brecha train/CV. Se reporta la estabilidad entre folds y por fuente. Opcional: probar LBP `uniform` (59 bins) como cambio de atributos aislado. El OOB no sustituye la CV agrupada porque las muestras bootstrap fuera de bolsa siguen compartiendo ráfaga o imagen original con las de dentro de bolsa.

## Frente 3. `regular` y AgriFreshNET

**Tabla fuente × clase (recortes del split actual):**

| Partición | Fuente | good | regular | bad |
|---|---|---:|---:|---:|
| train | AgriFreshNET | 3.149 | 3.122 | 2.745 |
| train | Kaggle | 3.801 | 0 | 3.607 |
| train | Propias | 20 | 11 | 1 |
| val | AgriFreshNET | 620 | 705 | 678 |
| val | Kaggle | 784 | 0 | 769 |
| val | Propias | 1 | 0 | 0 |
| test | AgriFreshNET | 643 | 641 | 771 |
| test | Kaggle | 872 | 0 | 743 |
| test | Propias | 3 | 0 | 1 |

Proporciones por fuente: AgriFreshNET tiene 34% good, 34% regular y 32% bad. Kaggle tiene 52% good, 0% regular y 48% bad. Además, **el producto revela la fuente**: 6 productos solo existen en AgriFreshNET, 4 solo en Kaggle, y solo banano y naranja tienen ambas fuentes (el desglose producto × fuente × clase está en `out/data_audit.txt`).

**Predicciones `regular` en validation por fuente:** RF hace 0 en Kaggle y 621 en AgriFreshNET (705 reales). CNN hace 1 en Kaggle, 929 en AgriFreshNET y 1 en propias; es decir, la CNN sobrepredice `regular` en AgriFreshNET.

**Evidencia.**
1. *Sesgo de etiquetas (confirmado, es del dataset).* Kaggle solo tiene las carpetas Good y Bad, así que cualquier pieza intermedia de Kaggle quedó como good o bad.
2. *Sesgo de adquisición (confirmado).* Las imágenes crudas de Kaggle son de 256×256 y las de AgriFreshNET de 512×512. La nitidez mediana (varianza del Laplaciano a 128 px) es 292 en Kaggle frente a 566–1066 en AgriFreshNET. El fondo del recorte es más claro en AgriFreshNET (184 frente a 146–166).
3. *Los atributos codifican la fuente.* Un RF entrenado en train para predecir la fuente con los mismos 359 atributos obtiene 0.976 de balanced accuracy en val. Restringido a banano y naranja good+bad, obtiene 0.975. Solo con color: 0.969; solo LBP: 0.888; solo forma: 0.848.
4. *Cambio de dominio (confirmado).* Un RF entrenado solo con AgriFreshNET, evaluado en Kaggle (val), da AUC good-vs-bad de 0.72, frente a 0.98 dentro de su dominio. Uno entrenado solo con Kaggle, evaluado en AgriFreshNET good+bad, da AUC 0.75 frente a 0.99. El modelo entrenado solo con AgriFreshNET predice `regular` en 305 de 1.553 recortes Kaggle; el modelo combinado, en 0. Eso indica que el modelo combinado aprendió a no predecir `regular` para imágenes con aspecto de Kaggle. Es un uso de la fuente coherente con las etiquetas de entrenamiento, pero no demuestra que reconozca deterioro intermedio en otros dominios.

**Conclusión defendible.** Hay un confusor fuente-clase y un cambio de dominio fuerte. Que la sonda acierte la fuente no prueba que el clasificador de calidad la use como atajo. Lo que sí queda demostrado es que la señal aprendida no transfiere bien entre fuentes. Hoy no hay datos `regular` fuera de AgriFreshNET en val ni en test para medir el atajo. Las 11 fotos propias `regular` están en train.

**Mitigación.**
1. Reportar siempre las métricas por fuente, con su soporte.
2. Conseguir `regular` de otra fuente: fotos propias nuevas (ver frente 4) o piezas etiquetadas de la carpeta Mixed de Kaggle.
3. Opcional: normalizar la resolución (reducir AgriFreshNET a la resolución de Kaggle antes de extraer atributos) como experimento controlado de invarianza.

No se cambia ninguna etiqueta.

## Frente 4. Fotos propias

**Auditoría.** El manifiesto tiene 42 fotos: 37 candidatas, 1 excluida (racimo) y 4 en revisión (mango). Las 37 candidatas forman 7 grupos:

| Partición | Recortes | Grupos | Detalle |
|---|---:|---:|---|
| train | 32 | 4 | banana_session_001 (16: 9 good, 6 regular, 1 bad), orange_session_001 (14), mango_session_001 (2) |
| val | 1 | 1 | apple_001 (good). RF y CNN la clasifican bien; la logística la predice como regular. |
| test | 4 | 3 | orange_001, orange_003 y papaya_session_001 (1 good y 1 bad) |

Corrección a la auditoría: test **sí** tiene fotos propias (4), pero no son suficientes para estimar nada. En la práctica hay unas 6 sesiones independientes, y dos de ellas concentran el 81% de los recortes. Los grupos sin `size` (mango) y la procedencia de los mangos (`fruits/mango/<uuid>.jpg`) quedan para el responsable de datos.

**Meta justificada.** Para estimar la accuracy con un semiancho de IC 95% de ±10 puntos se necesita n ≈ 1.96²·0.25/0.1² ≈ 96 piezas independientes. Para un recall por clase con un semiancho de unos ±15 puntos se necesitan unas 40 piezas por clase. **Objetivo: 40 piezas distintas por clase (unas 120), de al menos 6 sesiones, 2 o más lugares y 2 o más teléfonos.** Una pieza solo se fotografía una vez para evaluación.

**Protocolo (alternativa A, recomendada).**
1. Conjunto `external_own_v1` separado del split. Nunca entra a train ni se usa para seleccionar modelos.
2. Se congela un modelo antes de mirar sus resultados en este conjunto.
3. Dos integrantes anotan de forma independiente siguiendo `docs/criterios-calidad.md`, se calcula el kappa de Cohen y se resuelven los desacuerdos (o se descartan las fotos ambiguas).
4. Cada foto registra sesión, dispositivo, lugar y fecha.
5. Se evalúa una sola vez y se reportan las métricas por clase con su IC. La alternativa B (split v2 con grupos propios repartidos) solo tiene sentido si hay suficientes sesiones nuevas. Si no se recolecta, se aplica la alternativa C: declarar que las métricas no cubren el dominio de fotos de celular.

## Frente 5. Riesgo de overfitting: matriz de riesgos

| Riesgo | Tipo | Estado | Evidencia | Mitigación |
|---|---|---|---|---|
| Ráfagas de Kaggle repartidas entre particiones | Fuga | **Confirmado** | 99% de val Kaggle comparte ráfaga con train; F1 Kaggle CV 0.968 → 0.932 | Split v2 (F0) |
| Ráfagas de AgriFreshNET por fecha y hora | Fuga | Hipótesis | Los nombres llevan timestamp y no se agrupan | Misma prueba de CV |
| Aumentos de AgriFreshNET con su original | Fuga | Controlado | 0 originales en más de una partición; 8.233 de 13.074 son `aug_` | — |
| Duplicados exactos | Fuga | Controlado | 18 conjuntos md5, ninguno entre particiones ni con etiquetas contradictorias | — |
| Escalado o atributos ajustados fuera de train | Fuga | Descartado | StandardScaler está dentro del Pipeline; los atributos no tienen parámetros; los pesos de clase se calculan solo con train | — |
| CNN E3b (4 bloques) | Sobreajuste estadístico | Confirmado, contenido | Desde la época 30 la loss de train baja (0.12 → 0.08) y la de val no mejora; en la mejor época, 0.095 frente a 0.334 | EarlyStopping con restore_best_weights; regularización como única variante adicional |
| RF memoriza train | Sobreajuste estadístico | Esperado | Train 0.9999, CV 0.82, val 0.84; ~1460 hojas para 9.234 grupos | Rejilla de regularización; reportar CV como estimador |
| CNN de 3 bloques | Subentrenamiento | Resuelto | E1 corrigió la optimización; E3b, la capacidad | — |
| Selección repetida sobre val | Sesgo de selección | Bajo, pero crecerá | Hasta ahora la selección se hizo por CV; val solo se usó para reportar | Seleccionar por CV agrupada; val como confirmación |
| Cambio de dominio entre fuentes | Generalización | Confirmado | AUC entre dominios de 0.72 a 0.75 | Declarar el alcance; métricas por fuente |
| Soporte insuficiente del dominio propio | Falta de evidencia | Confirmado | 1 recorte en val | Frente 4 |
