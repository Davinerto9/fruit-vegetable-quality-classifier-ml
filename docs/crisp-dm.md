# Metodología CRISP-DM

Este índice organiza la metodología del clasificador en seis documentos, uno por fase, siguiendo la estructura del proyecto AgroQuality Classifier del semestre anterior. Ese proyecto sirve como referencia de organización y trazabilidad; su contenido y sus cifras no se trasladan al proyecto actual 2026-2.

La adaptación sigue la guía IBM SPSS Modeler CRISP-DM Guide, versión 18.3, y la consigna de Algoritmos y Programación III. CRISP-DM se aplica como ciclo iterativo: errores y hallazgos pueden llevar a revisar datos, etiquetas, preparación o modelos.

## Fases

1. [Comprensión del negocio](01_business_understanding.md)
2. [Comprensión de los datos](02_data_understanding.md)
3. [Preparación de los datos](03_data_preparation.md)
4. [Modelado](04_modeling.md)
5. [Evaluación](05_evaluation.md)
6. [Despliegue](06_deployment.md)

## Documentos de apoyo

- [Workflow personalizado](supporting_docs/crisp_dm_workflow.md)
- [Ética e impactos](supporting_docs/ethics_considerations.md)
- [Fundamentos matemáticos y teóricos](supporting_docs/mathematical_and_theoretical_foundations.md)
- [Criterios de calidad visual](criterios-calidad.md)
- [Revisión de anotaciones y segmentación](revision-anotaciones.md)

## Alcance y estado del proyecto

El prototipo recibe una fotografía de una fruta o verdura individual sobre un fondo sencillo y clasifica su apariencia externa visible como good, regular o bad. Calcula además un diámetro equivalente normalizado respecto al lado mayor de la imagen. Esta medida expresa ocupación relativa de la fotografía, no tamaño físico en centímetros. La solución apoya la inspección humana; no determina inocuidad, estado interno, precio ni aptitud para el consumo.

Se combinan Fruit Quality Classification de Kaggle, AgriFreshNET y fotos propias. La comparación incluye regresión logística, SVM lineal, Random Forest y CNN RGB. La interfaz prevista es Streamlit. Al 21 de noviembre de 2026, el inventario y la partición completa por grupos están generados y verificados estructuralmente. La revisión visual de candidatos fue estratificada, no individual para todo el conjunto. Los cuatro modelos se entrenaron y se compararon inicialmente en validation; la auditoría por imagen debe repetirse con scikit-learn 1.7.2 porque la corrida reciente cargó los checkpoints clásicos en 1.9.1. Test continúa reservado. La aplicación Streamlit está implementada; falta su demostración con los checkpoints reales.

| Fase | Estado documentado |
| --- | --- |
| Comprensión del negocio | Alcance, criterios iniciales, riesgos, ética e impactos definidos. |
| Comprensión de los datos | Inventario y EDA inicial disponibles; persisten diferencias entre fuentes y conflictos. |
| Preparación | Etiquetas pendientes adjudicadas; manifiesto actualizado y split completo generado/verificado. Falta completar la revisión visual de candidatos fuera de la muestra estratificada. |
| Modelado | Extractor clásico, regresión logística, SVM, Random Forest y CNN RGB implementados y entrenados. Los checkpoints locales no se suben a Git. |
| Evaluación | Hay métricas iniciales de validation para los cuatro modelos; el audit es provisional por la diferencia de scikit-learn 1.7.2/1.9.1 y la revisión visual está pendiente. Test sigue reservado y sin consultar. |
| Despliegue | App Streamlit implementada (`src/app/`), con el mismo preprocesamiento del entrenamiento; falta la demostración con checkpoints reales. |

La documentación del plan no se considera evidencia de una implementación o resultado. Cada fase se actualizará cuando sus artefactos reales estén disponibles.
