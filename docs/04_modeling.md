# 4. Modelado

## Objetivo y entrada

El objetivo es aprender una función que asigne a la imagen segmentada de una pieza una de tres clases de apariencia: good, regular o bad. El modelado comienza cuando se hayan revisado los ejemplos de preparación y creado la partición completa por grupos.

La clasificación recibe dos representaciones comparables de las piezas: atributos manuales de color, textura y forma para los modelos clásicos, y recortes RGB de 128 × 128 píxeles para la CNN. La medida geométrica de tamaño se calcula aparte desde la máscara; no es una cuarta clase ni un modelo supervisado mientras no existan etiquetas suficientes.

## Modelos considerados

| Modelo | Representación de entrada | Rol |
| --- | --- | --- |
| Regresión logística | Histograma de color de la pieza y atributos seleccionados | Línea base interpretable para evaluar si los modelos más complejos aportan valor. |
| SVM | Atributos de color, textura y forma | Modelo clásico para separar clases con fronteras lineales o con kernel. |
| Random Forest | Atributos de color, textura y forma | Ensamble de árboles para modelar relaciones entre atributos. |
| CNN pequeña desde cero | Recorte RGB de 128 × 128 | Aprende patrones espaciales de manchas, bordes y textura. |

SVM y Random Forest cubren los dos modelos clásicos exigidos; la CNN cubre deep learning. La regresión logística sirve de línea base adicional. KNN, árboles individuales y transfer learning son alternativas posibles; se documentará si se descartan por tiempo, rendimiento, tamaño del conjunto o capacidad de cómputo.

## Diseño experimental y ajuste

- Usar el mismo split por grupos para todos los modelos y mantener imágenes de una misma pieza, ráfaga o aumento en una sola partición.
- Ajustar SVM y Random Forest con búsqueda en rejilla y validación cruzada que respete los grupos. Usar estratificación por grupo cuando la cobertura de clases lo permita y registrar limitaciones.
- Seleccionar atributos, escalar variables y calcular pesos de clase solo con datos de entrenamiento dentro de cada partición de validación cruzada.
- Entrenar la CNN con el split de grupos, vigilar curvas de entrenamiento y validación, y ajustar arquitectura, regularización y épocas mediante validación.
- No usar el conjunto de prueba para seleccionar atributos, épocas, umbrales ni hiperparámetros.
- Comparar candidatos en validación mediante F1 macro, errores por clase, estabilidad y factibilidad de despliegue.

La formulación matemática, la función de pérdida y la justificación de los modelos se desarrollan en [Fundamentos matemáticos y teóricos](supporting_docs/mathematical_and_theoretical_foundations.md).

## Registro y artefactos

Cada ejecución debe registrar semilla, versión del índice y manifiesto, atributos o formato de entrada, transformaciones, particiones, hiperparámetros, métricas, tiempo de cómputo e incidencias. Los modelos, configuraciones, curvas y resultados se guardarán bajo experiments/ con nombres que permitan identificar el experimento.

## Estado y conclusión de fase

Los directorios src/features, src/models y src/training contienen únicamente inicializadores; no hay extractores formales de atributos, modelos ni entrenamiento implementados. El diseño de comparación está definido, pero los resultados no existen todavía.

La fase de modelado podrá cerrar cuando los candidatos se hayan entrenado, sus ajustes y artefactos estén registrados, y la selección esté justificada con validación. Si los errores apuntan a etiquetas, máscaras o atajos de fuente, se regresará a comprensión de datos o preparación.
