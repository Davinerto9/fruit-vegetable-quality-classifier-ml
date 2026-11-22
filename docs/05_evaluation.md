# 5. Evaluación

## Estado

Hay una evaluación inicial completa en validation para regresión logística, SVM lineal, Random Forest y CNN RGB. No se ha consultado test (`test_used: false`). Los resultados están registrados en `experiments/results/`; el auditor genera además predicciones por recorte y una lista priorizada de errores.

| Modelo | F1 macro validation | Balanced accuracy |
| --- | ---: | ---: |
| Regresión logística | 0.6845 | 0.7010 |
| SVM lineal | 0.6978 | 0.7022 |
| Random Forest | 0.8424 | 0.8363 |
| CNN RGB | 0.7988 | 0.8139 |

## Hallazgos del audit de validation

`experiments/results/validation_audit.json` corresponde a validation: 3.557 filas, 1.979 grupos, sin uso de test.

- La clase `regular` tiene 705 filas de 251 grupos y el 100% proviene de AgriFreshNET. No hay `regular` de ryandpark ni fotos propias en validation. Este confusor impide afirmar generalización entre fuentes para esa clase.
- Para RF/CNN, 2.686 filas fueron correctas en ambos modelos; RF falló y CNN acertó en 210; CNN falló y RF acertó en 378; ambos fallaron en 283.
- La unión de fallos de RF o CNN contiene 871 filas (24,5% de validation). El CSV `validation_error_cases.csv` contiene 1.426 filas porque también incluye los errores de regresión logística o SVM; por tanto, si se auditan los cuatro modelos, esa es la cola completa. `validation_error_cases_reviewed.csv` actualmente duplica las 1.426 filas y las mismas columnas del CSV original, y no acredita una revisión completada.
- La similitud Jaccard entre los conjuntos de error de RF y CNN es 0.3249; la complementariedad es parcial. No crear un ensemble sin revisar si los fallos son de datos, preprocesamiento o comportamiento genuino del modelo.
- Las advertencias de métricas sobre clases ausentes en `y_true` aparecen en desgloses por fuente/producto cuando ese subgrupo no contiene todas las etiquetas. No indican que haya fallado el proceso global; sí indican que los F1 de clase ausente no son estimables y que debe mostrarse el soporte antes de interpretar el promedio.

### Advertencia de versión

Los clasificadores clásicos fueron serializados con scikit-learn 1.7.2. El audit reciente cargó esos objetos con scikit-learn 1.9.1 y emitió `InconsistentVersionWarning`. Aunque el proceso terminó y guardó archivos, la predicción serializada entre versiones no tiene garantía de compatibilidad. Antes de usar los errores por imagen para decidir cambios, instalar 1.7.2 en el entorno de análisis y volver a correr validation. El pin del entorno se establece en `requirements.txt`.

## Secuencia segura antes de test

1. Alinear scikit-learn 1.7.2 y volver a generar el audit de validation. Confirmar que los checkpoints cargan sin advertencia y registrar de nuevo métricas y hashes.
2. Revisar visualmente los 871 casos en contexto (imagen original, máscara, etiqueta, fuente, producto y predicciones de los cuatro modelos). Registrar por caso `review_status`, `decision`, `reason` y responsable. Separar error real de etiqueta/máscara, caso ambiguo, limitación de captura y error del modelo.
3. Analizar el sesgo de fuente: `regular` aparece solo en AgriFreshNET. Buscar ejemplos `regular` de otros orígenes con etiquetas defendibles, o restringir explícitamente el alcance de la conclusión. No duplicar o mover filas de validation/test para balancear.
4. Resolver las cuatro segmentaciones pendientes de mango o excluir/documentar mango del alcance. Las dos muestras aceptadas se movieron juntas a train; no existe evaluación independiente de mango.
5. Si se cambian etiquetas, máscaras, features o entrenamiento, regenerar solo los artefactos afectados, repetir CV/entrenamiento según corresponda y volver a medir validation. No ajustar usando resultados de test.
6. Cuando el equipo cierre la revisión, guardar un manifiesto de congelamiento (commit, hash del split, hashes de checkpoints y versiones), bloquearlo y ejecutar el test una sola vez mediante el comando protegido documentado por `src/evaluation/model_audit.py`.

## Métricas y reporte final

Reportar F1 macro, balanced accuracy, precisión/recall/F1 por clase, matriz de confusión y soporte. Destacar falsos negativos de `bad`, el comportamiento de `regular`, resultados por fuente/producto solo con soporte suficiente y limitaciones de dominio. No comparar subgrupos sin mostrar su tamaño. El resultado de test será una estimación final bajo el split y las fuentes disponibles, no garantía de rendimiento en condiciones reales.

## Artefactos

- Resumen y métricas por fuente/producto: `experiments/results/validation_audit.json`.
- Predicciones por fila: `experiments/results/validation_predictions.csv`.
- Casos señalados: `experiments/results/validation_error_cases.csv`.
- Copia de trabajo para revisión: `experiments/results/validation_error_cases_reviewed.csv` (no se considera revisada mientras no tenga decisiones y notas verificables).
- Procedimiento para el equipo: [Guía de modelado y auditoría](modeling_team_guide.md).
