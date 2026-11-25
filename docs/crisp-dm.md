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

Se combinan Fruit Quality Classification de Kaggle, AgriFreshNET y fotos propias. La comparación incluye regresión logística, SVM lineal, Random Forest, CNN RGB y un ensamble RF + CNN. La interfaz es Streamlit. Al 25 de noviembre de 2026:

- Se detectó y corrigió una fuga por ráfagas de Kaggle en el split v1. Todos los modelos vigentes se entrenaron sobre el split v2.
- Los candidatos en validation v2 son RF (0,857), CNN E3b (0,858) y el ensamble 0,5/0,5 (0,875).
- Test sigue reservado y la decisión final está pendiente.

| Fase | Estado documentado |
| --- | --- |
| Comprensión del negocio | Alcance, criterios iniciales, riesgos, ética e impactos definidos. |
| Comprensión de los datos | Inventario y EDA inicial disponibles; persisten diferencias entre fuentes y conflictos. |
| Preparación | Split v2 por ráfagas, sin la fuga de v1. Falta comprobar las ráfagas por fecha y hora de AgriFreshNET y completar la revisión visual de candidatos. |
| Modelado | RF elegido por CV agrupada sobre una rejilla de 30 configuraciones; CNN E1/E2/E3b documentadas con resultados negativos; ensamble con peso fijo. Checkpoints versionados `*_v2_burst*`, fuera de Git. |
| Evaluación | Validation v2 con IC por grupos, métricas por fuente, sonda de fuente y transferencia entre dominios. El auditor de test está preparado para v2. Falta fijar el modelo final, hacer el freeze y abrir test. |
| Despliegue | App Streamlit implementada con el ensamble v2 y fallback a RF; falta la demostración con los checkpoints reales. |

La documentación del plan no se considera evidencia de una implementación o resultado. Cada fase se actualizará cuando sus artefactos reales estén disponibles.
