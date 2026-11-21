# 5. Evaluación

## Objetivo y conjunto de evaluación

La evaluación determina si los modelos satisfacen los criterios definidos en Comprensión del negocio y si los resultados son útiles para la inspección asistida. Se utilizará el test reservado por grupos una vez que el modelo y sus parámetros se hayan seleccionado con validación.

La línea base, SVM, Random Forest y CNN se evaluarán con las mismas imágenes de prueba y el mismo protocolo. El conjunto de prueba no se usará para ajustar modelos ni reglas.

## Métricas previstas

- **F1 macro:** métrica principal que pondera por igual good, regular y bad.
- **Precisión, exhaustividad y F1 por clase:** muestran falsos positivos y falsos negativos por etiqueta.
- **Matriz de confusión y soporte:** documentan los conteos detrás de cada métrica.
- **Exactitud balanceada:** complemento para comparar la tasa de aciertos por clase ante el desbalance.
- **Resultados por fuente y producto:** solo cuando haya suficientes ejemplos para una interpretación responsable.

La definición y el fundamento matemático de estas métricas se presentan en [Fundamentos matemáticos y teóricos](supporting_docs/mathematical_and_theoretical_foundations.md).

## Análisis de errores y generalización

El análisis revisará especialmente casos bad predichos como good y errores entre clases contiguas. Las imágenes con error se inspeccionarán para distinguir problemas de etiqueta, máscara, fondo, fuente, producto o predicción. Se compararán entrenamiento, validación y prueba para detectar sobreajuste y diferencias de generalización.

Un promedio agregado no será suficiente si la clase regular está asociada principalmente a una fuente. Se reportarán tamaños de subgrupo, errores por fuente/producto y limitaciones; los grupos con poca cobertura no justificarán afirmaciones generales.

Para tamaño, se contrastará el diámetro normalizado con máscaras revisadas. El acuerdo con categorías small/medium/large se medirá únicamente si hay suficientes etiquetas y condiciones de captura comparables. En otro caso, se reportará solo como medida geométrica relativa.

## Regla de decisión y siguientes pasos

El candidato final debe compararse con la línea base en test y reportar las métricas por clase, en particular la exhaustividad de bad. Si no supera la línea base, comete errores críticos o parece aprovechar la fuente de origen, no se forzará un despliegue: se documentará el hallazgo y se regresará a preparación o modelado.

## Artefactos y estado

La fase debe producir tablas de métricas, matriz de confusión, figuras vectoriales, registro de errores, hiperparámetros, modelo seleccionado y decisión de despliegue. Al 21 de noviembre de 2026 no hay modelos entrenados, métricas ni resultados de prueba. Esta fase está definida como protocolo, no completada.
