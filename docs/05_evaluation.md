# 5. Evaluación

## Reglas

1. **Validación y prueba se usan una sola vez.** Ninguna de las dos elige hiperparámetros, modelos ni pesos: eso se decide con validación cruzada dentro de *train* ([modelado](04_modeling.md)).
2. **La prueba tiene candado.** `src/evaluation/model_audit.py --partition test --confirm-final-test` exige un archivo `pretest_freeze.sha256` con el split y los modelos congelados, y crea `test_evaluation.lock`. Un segundo intento se rechaza.
3. **Las fotos de celular** del pool de la clase se etiquetan a ciegas (sin abrir la app) y se evalúan una vez, también con candado (`src/evaluation/external_eval.py`).

## Métricas

| Métrica | Para qué | Cómo se explica |
| --- | --- | --- |
| Accuracy | Fruta | De cada 100 fotos, en cuántas acertó la fruta. |
| F1 macro | Estado | Promedio del F1 de buena, regular y mala con el mismo peso; no se infla acertando solo la clase más común. |
| Matriz de confusión | Ambas | Qué se confunde con qué. |
| Errores mala→buena | Estado | Frutas malas que pasarían como buenas: el error más caro. |
| F1 por fuente | Estado | Kaggle solo tiene buena y mala; AgriFreshNET tiene las tres. La cifra representativa de tres clases es la de AgriFreshNET. |
| Intervalo de confianza 95% | Todas | Rango probable de la cifra real, calculado remuestreando grupos o sesiones de fotos. |

## Resultados

| Conjunto | Estado | Fuente |
| --- | --- | --- |
| Validación v3 | Pendiente de reentrenar ([plan simple](plan_simple.md), paso 6) | `experiments/results/validation_v3_week_*.json` |
| Fotos de celular | Pendiente de etiquetar (pasos 7 a 9) | `experiments/results/external_evaluation.json` |
| Prueba | Reservada hasta congelar (paso 10) | `experiments/results/final_test_v3_week_*.json` |

Como referencia del camino, en validación v2 el promedio RF + CNN obtuvo F1 macro 0,875 (0,843 en AgriFreshNET), con 79 frutas malas clasificadas como buenas. Esa cifra todavía tenía la fuga de AgriFreshNET y la CNN se había elegido con validación, así que no es la que se sustenta.

## Hallazgos que se mantienen

1. **Fotos casi iguales inflan la nota.** En v1, el 99% de las fotos de Kaggle en validación tenía una "gemela" de la misma ráfaga en entrenamiento, y el F1 de Kaggle subía unos 3,5 puntos. AgriFreshNET tiene el mismo problema por sesiones de fotos de las mismas piezas; v3 lo corrige.
2. **La clase regular viene casi toda de AgriFreshNET.** Un modelo que adivina la fuente con los mismos atributos acierta el 97,6%, y lo aprendido en una fuente se transfiere mal a la otra. Por eso se reportan las métricas por fuente y se prueba con fotos de celular.
3. **Fotos de celular.** En validación v2 solo había 4 fotos propias, y los modelos fallaron 3. Las fotos del pool de la clase son la forma de medirlo de verdad.

## Lo que el resultado no garantiza

La cifra de prueba estima el desempeño con fotos parecidas a las de los datasets. No garantiza el mismo desempeño con cualquier celular, luz o fondo, ni con frutas que el modelo no conoce.
