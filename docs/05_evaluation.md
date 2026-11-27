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

Tablas y gráficas en el [notebook 04](../notebooks/04_resultados.ipynb). Modelo final: **Random Forest**, elegido en val antes de abrir test ([despliegue](06_deployment.md)).

| Conjunto | Fotos (grupos) | Estado: F1 macro RF [IC 95%] | Malas como buenas | Fruta: accuracy |
| --- | --- | --- | ---: | ---: |
| Val | 3.605 (129) | 0,762 [0,697 – 0,850] | 41 | 86,9 % |
| **Test** (una sola vez) | 4.169 (184) | **0,596 [0,522 – 0,677]** | **40** | **79,5 %** |
| Celular, mitad prueba (una sola vez) | 302 (6 sesiones) | 0,295 [0,227 – 0,334] | 2 de 4 | 31,8 % |

En test, por fuente: Kaggle 0,93 (solo buena/mala), AgriFreshNET 0,43 y AFruitDB 0,61.

Los otros modelos en test: regresión logística 0,560, SVM 0,566, CNN 0,603 (190 malas como buenas) y ensamble 0,621 (101 malas como buenas). El ensamble saca algo más de F1 en test, pero deja pasar más del doble de frutas malas. La elección se hizo antes de ver test y no se cambia.

### Por qué test baja frente a val

- **Casi toda la caída es de `regular` en AgriFreshNET.** De sus 786 `regular` en test, RF llama `bad` a 570. Pasa sobre todo en banano, pepino, piña y berenjena: entre 87 % y 93 % de las regulares de cada una salen `bad`.
- **Casi todos los productos de AgriFreshNET tienen solo de 2 a 5 lotes de `regular`**, y los lotes enteros se reparten (así se evita la fuga). A val le tocaron lotes de naranja, papaya, tomate y berenjena, que RF generaliza bien. A test le tocaron lotes de banano, pepino y piña, cuyo `regular` se parece más a `bad`. Con tan pocos lotes, la diferencia entre val y test depende mucho de qué lote cayó en cada lado. Por eso la cifra que se sustenta es la de test.

### Fotos de celular

| Fruta | Fotos | Acierta la fruta | ¿Fotos de celular de esa fruta en train? |
| --- | ---: | ---: | --- |
| Banano | 18 | 100 % | sí |
| Manzana | 35 | 89 % | sí |
| Naranja | 49 | 94 % | no |
| Guayaba | 91 | 1 % | no |
| Limón | 70 | 0 % | no |
| Mango | 39 | 0 % | no |

- **Fruta.** El modelo llama "naranja" a los limones y guayabas, que en estas fotos son amarillo-verdosos como las naranjas locales, y "papaya" al mango. Los recortes se revisaron a mano y están bien hechos: es un cambio de dominio, no un error del código.
- **Estado.** Casi todas las buenas (250 de 302) salen buenas, pero las regulares no se detectan. Solo hay 4 malas, todas de mango: con tan pocas, esa cifra no permite concluir nada.
- **Lo que muestra.** Las frutas que tuvieron fotos de celular en la mitad "entrenar" se reconocen; las que no, fallan. El camino para mejorar es tomar fotos con celular de todas las frutas para entrenar.

## Hallazgos que se mantienen

1. **Fotos casi iguales inflan la nota.** En el primer reparto, el 99 % de las fotos de Kaggle en validación tenía una "gemela" de la misma ráfaga en entrenamiento, y el F1 de Kaggle subía unos 3,5 puntos. AgriFreshNET tiene el mismo problema por semanas de fotos de las mismas piezas; el reparto final lo corrige.
2. **La clase regular viene casi toda de AgriFreshNET.** Un modelo que adivina la fuente con los mismos atributos acierta el 97,6%, y lo aprendido en una fuente se transfiere mal a la otra. Por eso se reportan las métricas por fuente y se prueba con fotos de celular.
3. **Fotos de celular.** En val solo había 4 fotos propias, y los modelos fallaron 3. Por eso la mitad de las fotos del pool de la clase entrena el modelo (para que aprenda cómo se ve una foto de celular) y la otra mitad lo mide. El resultado lo confirma: solo se reconocen bien las frutas que tuvieron fotos de celular en el entrenamiento.

## Lo que el resultado no garantiza

La cifra de prueba estima el desempeño con fotos parecidas a las de los datasets. No garantiza el mismo desempeño con cualquier celular, luz o fondo, ni con frutas que el modelo no conoce. La prueba con celular lo muestra: con fotos reales, el modelo solo es confiable para reconocer las frutas que ya vio en fotos de celular, y no detecta bien el estado `regular`.
