# 5. Evaluación

## Reglas

1. **Validación y prueba se usan una sola vez.** Ninguna de las dos elige hiperparámetros, modelos ni pesos: eso se decide con validación cruzada dentro de *train* ([modelado](04_modeling.md)).
2. **La prueba tiene candado.** `python -m src.main congelar` guarda la huella de `splits.csv` y de los cinco modelos en `results/pretest_freeze.sha256`. `python -m src.main evaluar --test --confirmo` comprueba que nada cambió y crea `results/test_evaluation.lock`. Un segundo intento, o reentrenar después, se rechaza.
3. **Las fotos de celular** del pool de la clase se parten por sesión (misma cámara y día) antes de etiquetarlas: la mitad "entrenar" se suma a train y la mitad "prueba" se evalúa una sola vez, con candado. El programa se niega a evaluar si alguna foto o sesión de la prueba aparece en el entrenamiento (`src/evaluation/phone_test.py`). Las dos mitades se etiquetan a ciegas, sin abrir la app. Una sesión es un bloque de fotos de la misma fruta y cámara con menos de 10 minutos entre sí; las fotos idénticas a `data/own/` se apartan. El resultado se da también por fruta, separando las que tuvieron fotos de celular en el entrenamiento (pueden ser las mismas piezas en otro momento, así que su cifra es más optimista) de las que nunca se vieron en celular.

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

Tablas y gráficas en el [notebook 04](../notebooks/04_resultados.ipynb).


| Conjunto | Estado | Fuente |
| --- | --- | --- |
| Val | Pendiente de reentrenar (`python -m src.main evaluar`) | `results/evaluacion_val.json` |
| Fotos de celular (mitad prueba) | Pendiente de etiquetar (`python -m src.main celular …`) | `results/evaluacion_celular.json` |
| Test | Reservada hasta congelar | `results/evaluacion_test.json` |

Como referencia del camino, con el intento de reparto 2 el promedio RF + CNN obtuvo F1 macro 0,875 (0,843 en AgriFreshNET), con 79 frutas malas clasificadas como buenas. Esa cifra todavía tenía la fuga de AgriFreshNET y la CNN se había elegido con validación, así que no es la que se sustenta.

## Hallazgos que se mantienen

1. **Fotos casi iguales inflan la nota.** En el primer reparto, el 99 % de las fotos de Kaggle en validación tenía una "gemela" de la misma ráfaga en entrenamiento, y el F1 de Kaggle subía unos 3,5 puntos. AgriFreshNET tiene el mismo problema por semanas de fotos de las mismas piezas; el reparto final lo corrige.
2. **La clase regular viene casi toda de AgriFreshNET.** Un modelo que adivina la fuente con los mismos atributos acierta el 97,6%, y lo aprendido en una fuente se transfiere mal a la otra. Por eso se reportan las métricas por fuente y se prueba con fotos de celular.
3. **Fotos de celular.** En val solo había 4 fotos propias, y los modelos fallaron 3. Por eso la mitad de las fotos del pool de la clase entrena el modelo (para que aprenda cómo se ve una foto de celular) y la otra mitad lo mide.

## Lo que el resultado no garantiza

La cifra de prueba estima el desempeño con fotos parecidas a las de los datasets. No garantiza el mismo desempeño con cualquier celular, luz o fondo, ni con frutas que el modelo no conoce.
