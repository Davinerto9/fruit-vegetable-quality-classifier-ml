# 5. Evaluación

## Estado

Hay una evaluación completa en **validation v2** (3.795 recortes, 852 grupos) para los cinco candidatos. Test no se ha consultado (`test_used: false`; no existen `pretest_freeze.sha256` ni `test_evaluation.lock`). Las cifras de validation v1 que aparecían antes en este documento (RF 0,842, CNN 0,799) estaban afectadas por la fuga por ráfagas de Kaggle y se reemplazaron.

## Resultados en validation v2

| Modelo | F1 macro [IC 95% por grupos] | Recall `bad` | `bad→good` | F1 `regular` | Kaggle (2 clases) | AgriFreshNET (3 clases) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Regresión logística | 0,714 | 0,789 | 165 | 0,590 | 0,871 | 0,616 |
| SVM lineal | 0,717 | 0,820 | 173 | 0,573 | 0,881 | 0,628 |
| Random Forest | 0,857 [0,834 – 0,878] | 0,897 | 121 | 0,783 | 0,954 | 0,799 |
| CNN E3b | 0,858 [0,831 – 0,881] | 0,916 | 90 | 0,776 | 0,938 | 0,823 |
| **Ensamble 0,5/0,5** | **0,875** | **0,926** | **79** | **0,799** | 0,949 | **0,843** |

Diferencias pareadas con bootstrap por grupos (IC 95%):

- CNN E3b − RF: +0,001 [−0,027; +0,028]. Empate.
- Ensamble − RF: +0,018 [−0,004; +0,042] global, no significativo; en AgriFreshNET, +0,044 [+0,012; +0,077].
- Ensamble − CNN: +0,018 [+0,008; +0,028]. Significativo global, en Kaggle y en AgriFreshNET.

Fuentes: `experiments/diagnostics/out/ensemble_rf_cnn_e3b.txt`, `cnn_vs_rf_v2_burst_e3b.txt`, `classic_eval_v2_burst.txt` y `experiments/results/*_v2_burst*.json`.

## Hallazgos

1. **Fuga corregida.** En v1, el 99% de los recortes de Kaggle en validation compartía ráfaga con train, y el F1 de Kaggle estaba inflado en unos 3,5 puntos. El split v2 no tiene ninguna ráfaga repartida.
2. **Confusor fuente–clase.** Todos los `regular` no propios vienen de AgriFreshNET. Una sonda que predice la fuente con los mismos atributos acierta el 97,6%, y la transferencia entre fuentes cae de AUC 0,98 a 0,72. El atajo no está demostrado, pero la señal aprendida no transfiere bien. Por eso se reportan siempre las métricas por fuente.
3. **Fotos de celular sin medir.** Validation v2 tiene 4 fotos propias, y RF y la CNN fallan 3. Con n = 4 no se puede estimar el desempeño, pero el resultado es coherente con el cambio de dominio. El conjunto externo `external_own_v1` (al menos 40 piezas por clase) es la forma de medirlo.
4. **Errores complementarios.** RF y la CNN coinciden en el 84,3% de las fotos; en las 594 donde discrepan, el ensamble acierta el 58,4%, frente al 46,5% de RF y el 48,1% de la CNN.
5. **Sesgo de selección.** La CNN se eligió en validation entre tres variantes, así que las cifras de la CNN y del ensamble son algo optimistas. La estimación sin sesgo será la de test.

## Secuencia antes de abrir test

> La secuencia vigente es la del [plan simple](plan_simple.md): split v3 sin la fuga de AgriFreshNET, reentrenamiento con `GridSearchCV` por grupos, validation una vez, fotos de celular etiquetadas a ciegas y evaluadas una vez, y test una vez. Los pasos de abajo son el resumen anterior.

1. Probar si las ráfagas por fecha y hora de AgriFreshNET producen fuga (CV agrupada por minuto). Si la hay, generar un split v3 y reentrenar RF y E3b con la misma configuración, sin buscar variantes nuevas.
2. Fijar por escrito el modelo final y su criterio (recomendado: ensamble 0,5/0,5, por el costo del error `bad→good`).
3. Guardar el *freeze*: commit, hash del split, hashes de los checkpoints y versiones de las librerías.
4. Evaluar `external_own_v1` una sola vez con el modelo congelado, si se completa la recolección.
5. Ejecutar test una sola vez con `src/evaluation/model_audit.py --partition test --confirm-final-test`.

## Métricas y reporte final

Se reportan F1 macro, balanced accuracy, precisión, recall y F1 por clase, matriz de confusión y soporte, todo con IC por grupos y desglosado por fuente. Se destacan los errores `bad→good` y `regular→good`. El resultado de test será una estimación bajo las fuentes disponibles, no una garantía de rendimiento con fotos de celular.

## Artefactos

- Resultados v2: `experiments/results/*_v2_burst*.json` y `ensemble_rf_v2_burst_grid_cnn_v2_burst_e3b.json`.
- Predicciones por recorte: `experiments/diagnostics/out/val_v2_predictions_v2_burst_*.csv`.
- Diagnósticos de fuente, dominio, fuga y CI: `experiments/diagnostics/out/`.
- Seguimiento y matriz de riesgos: [modeling_fronts_tracking.md](modeling_fronts_tracking.md).
- Auditoría v1 (histórica): `experiments/results/validation_audit.json`.
