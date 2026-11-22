# Cierre de sesión: modelado y auditoría

**Fecha:** 21 de noviembre de 2026  
**Estado:** validación inicial; test sigue reservado.

## Qué queda hecho

- Se generó el conjunto de atributos clásicos (359 atributos) y se entrenaron regresión logística, SVM lineal y Random Forest.
- Se entrenó una CNN RGB de 128 × 128; el checkpoint seleccionado corresponde a la época 40. La evaluación posterior cargó ese checkpoint y no repitió entrenamiento.
- Los cuatro modelos se midieron en las mismas 3.557 filas de validation y el mismo split 70/15/15.
- Se generó `validation_audit.json`, la tabla de predicciones y el CSV de casos de error. `test_used` permanece en `false`.

## Métricas iniciales

| Modelo | F1 macro | Balanced accuracy |
| --- | ---: | ---: |
| Regresión logística | 0.6845 | 0.7010 |
| SVM lineal | 0.6978 | 0.7022 |
| Random Forest | 0.8424 | 0.8363 |
| CNN RGB | 0.7988 | 0.8139 |

Por clase, RF obtuvo F1 de 0.8798 en `good`, 0.7557 en `regular` y 0.8917 en `bad`. CNN obtuvo 0.8300, 0.7017 y 0.8647 respectivamente.

## Hallazgos y límites

1. **Desacuerdo entre modelos:** RF y CNN acertaron ambos 2.686 casos; solo RF falló 210; solo CNN falló 378; ambos fallaron 283. La unión de errores RF/CNN es 871 casos (24,5% de validation); Jaccard de errores: 0.3249. El CSV de revisión de los cuatro modelos contiene 1.426 filas porque también incluye errores de regresión logística y SVM.
2. **Sesgo de fuente:** los 705 ejemplos `regular` (251 grupos) de validation son todos de AgriFreshNET. No se puede afirmar que el modelo generalice esa clase a otras fuentes.
3. **Auditoría incompleta:** los 871 registros deben revisarse visualmente con su etiqueta, fuente, máscara y predicciones. El CSV con sufijo `_reviewed` es una copia sin columnas de decisión y no prueba que se revisó.
4. **Mango:** dos muestras aceptadas quedaron en train dentro de un solo grupo/sesión; no hay mango en validation/test. Cuatro máscaras siguen sin resolver.
5. **Versiones:** los checkpoints clásicos se crearon con scikit-learn 1.7.2. El audit de validation cargó en 1.9.1 y emitió `InconsistentVersionWarning`. Repetir audit con 1.7.2 antes de confiar en comparaciones por fila.
6. **Generalización:** las métricas son preliminares. El set de imágenes fue filtrado y split grupal, pero solo se revisó visualmente una muestra estratificada pequeña de candidatos.
7. **Advertencias de métricas por subgrupo:** algunos cortes por fuente/producto no contienen las tres clases. Los warnings de clases ausentes en `y_true` son consecuencia de ese soporte desigual; leer métricas junto con el conteo por clase.

## Siguiente secuencia

1. Crear/actualizar el entorno del proyecto con scikit-learn 1.7.2 y comprobar la versión.
2. Repetir solo el audit de validation; no reentrenar la CNN para puntuar su checkpoint.
3. Revisar y registrar decisión para cada uno de los 871 casos. Mantener registro auditable de motivo, revisor y fecha.
4. Decidir cómo tratar mango y la ausencia de `regular` multifuente: resolver máscaras/etiquetas, conseguir cobertura defendible, o restringir el alcance declarado.
5. Si se cambian datos o pipeline, rehacer solo los experimentos afectados y volver a comparar en validation.
6. Cuando el equipo acepte los datos, el pipeline y los modelos, congelar hashes de split/checkpoints/versiones y correr test una sola vez con el comando protegido. No usar test para elegir modelo.
7. Terminar interfaz, informe IEEE, diagrama CRISP-DM, anexos/referencias y video de sustentación.

## Estimación de avance global

**Aproximadamente 55%.** El flujo de datos y split grupal están implementados, los cuatro modelos tienen checkpoints y métricas iniciales de validation, y el análisis de errores ya se puede reproducir. La parte restante incluye auditoría visual de los 871 casos, alineación de versión, decisiones de fuente/mango, congelamiento y test final de una sola ejecución, además de interfaz y entrega académica final. Es una estimación de gestión del avance, no una métrica objetiva.

## Artefactos principales

- [Modelado](04_modeling.md)
- [Evaluación](05_evaluation.md)
- [Guía del equipo](modeling_team_guide.md)
- `experiments/results/validation_audit.json`
- `experiments/results/validation_predictions.csv`
- `experiments/results/validation_error_cases.csv`
