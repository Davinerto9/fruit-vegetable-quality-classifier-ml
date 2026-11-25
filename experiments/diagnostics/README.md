# Diagnósticos de los cinco frentes (modelado)

Scripts de solo lectura usados para el diagnóstico del 24-nov-2026 (commit `ad0b21f`).
**Ninguno lee predicciones ni métricas de test.** Los que entrenan modelos auxiliares (sondas, transferencia entre dominios, CV) usan solo filas `train` y evalúan en `val` o en folds internos de train. `crop_stats.py` calcula estadísticas descriptivas de imagen para todas las filas (incluidas las de test, sin etiquetas predichas); el análisis (`pred_an.py`) excluye test.

Ejecutar desde la raíz, con scikit-learn 1.7.2 (versión de los checkpoints):

```bash
PYTHONPATH=. python experiments/diagnostics/data_audit.py
```

| Script | Qué mide | Salida |
|---|---|---|
| `data_audit.py` | fuente × clase × partición, producto × fuente, fotos propias por partición/grupo | `out/data_audit.txt` |
| `rf_diag.py` | RF: hiperparámetros, profundidad, brecha train/val por fuente, importancias, sonda de fuente | `out/rf_diag.txt` |
| `xdomain.py` | RF entrenado en una fuente y evaluado en la otra (solo train → val) | `out/xdomain.txt` |
| `burst.py` | grupos de ráfaga de `index.csv` vs grupos usados en `splits.csv` | consola |
| `compare_splits.py` | integridad y cobertura de v1 vs v2 (solo metadatos) | `out/compare_splits.txt` |
| `eval_classic_split.py` | evalúa checkpoints clásicos en val de un split (v1 o v2): fuente, producto, `bad→good`, brecha | `experiments/results/classic_eval_*.json`, `out/classic_eval_*.txt` |
| `bootstrap_ci.py` | IC 95% del F1 de RF por bootstrap de grupos | `out/bootstrap_ci.txt` |
| `paired_bootstrap_rf.py` | Δ F1 pareado (bootstrap de grupos) entre RF grid y RF base sobre val v2 | `out/paired_bootstrap_rf.txt` |
| `check_cnn_cache.py` | equivalencia exacta de tensores y orden con y sin `--cache` | `out/check_cnn_cache.txt` |
| `eval_cnn_vs_rf_v2.py` | CNN v2 vs RF v2 en val v2 (mismas filas): fuente, producto, Δ pareado | `experiments/results/cnn_vs_rf_*.json`, `out/cnn_vs_rf_*.txt` |
| `paired_cnn_runs.py` | Δ F1 pareado entre dos corridas CNN sobre val v2 | `out/paired_cnn_*.txt` |
| `check_cnn_augment.py` | `--augment`: simetrías exactas, etiquetas conservadas, reproducible, varía entre épocas | `out/check_cnn_augment.txt` |
| `eval_ensemble_v2.py` | ensamble RF + CNN (voto suave 0,5/0,5 fijo) en val v2; curva por peso solo descriptiva | `out/ensemble_rf_cnn_e3b.txt` |
| `cv_burst.py` | CV agrupada en train con grupos por archivo vs por ráfaga | consola |
| `crop_stats.py` | brillo, contraste, saturación, nitidez, fondo, md5, dHash por recorte | `out/crop_stats.csv` (no versionado, ~11 MB) |
| `pred_an.py` | predicciones `regular` por fuente, `bad→good`, errores por producto, duplicados | consola |

Resultados consolidados: [docs/modeling_fronts_tracking.md](../../docs/modeling_fronts_tracking.md).
