# Contexto y estado del proyecto

**Revisión:** 21 de noviembre de 2026  
**Repositorio activo:** `Davinerto9/fruit-vegetable-quality-classifier-ml`  
**Rama revisada:** `main`, en `813ec0e` (`docs: describe burst grouping and AgriFreshNET changes`)

Este archivo conserva el contexto para continuar el proyecto sin confundir la entrega actual con la referencia del semestre anterior. Las rutas, conteos y estados de datos aquí descritos corresponden a la copia local revisada en la fecha indicada; los archivos generados dentro de `data/processed/` están ignorados por Git y pueden cambiar al regenerarlos.

## 1. Qué pide la entrega 2026-2

Las consignas leídas son `0_Proyecto Final 26-2 (1).pdf` y `1_competencias_consigna_Actualizada.pdf`.

El proyecto grupal de Algoritmos y Programación III debe resolver un problema real de analítica con datos diversos y metodología CRISP-DM. Para este caso, la entrada es una foto de una fruta o verdura individual sobre fondo simple; las salidas son una calidad de al menos tres clases y una estimación de tamaño (categoría relativa o diámetro normalizado). Se pide una interfaz para cargar foto o capturarla con cámara.

La entrega debe combinar el conjunto público Fruit Quality Classification de Kaggle con al menos 30–50 fotos propias anotadas por calidad y tamaño. Debe comparar como mínimo dos modelos clásicos de ML y uno de deep learning, ajustar hiperparámetros, definir métricas antes de evaluar, analizar errores y generalización, y mostrar evidencia concreta de las seis fases CRISP-DM. También requiere diagrama CRISP-DM personalizado, informe final de máximo 7 páginas en formato IEEE, referencias realmente utilizadas y un video de hasta 10 minutos.

La consigna complementaria evalúa tres competencias:

- **PI1, ética:** identificar decisiones éticas, vincularlas al código profesional y documentar medidas como privacidad, consentimiento, sesgos y uso responsable.
- **PI2, impactos:** analizar impactos sociales, económicos, ambientales y globales, con riesgos, beneficios y mitigaciones.
- **PI3, matemáticas:** formular el problema, justificar el método frente a alternativas, explicar cálculos y métricas, interpretar resultados y extraer conclusiones técnicas con evidencia en el informe, código y anexos.

## 2. Problema y alcance que entendí

El producto es un prototipo de inspección visual que clasifica la **apariencia externa visible** de una pieza como `good`, `regular` o `bad`. `regular` representa deterioro inicial; `bad`, señales visibles más marcadas, como podredumbre u hongos. No mide inocuidad ni estado interno. El tamaño se plantea como medida relativa a la imagen, porque las fotos no tienen escala física ni distancia de cámara controlada. Por tanto, el diámetro normalizado no equivale a centímetros ni permite inferir tamaño real sin un protocolo de captura/calibración.

El objetivo debe presentarse como apoyo a la inspección humana, no como autoridad para decidir consumo, descarte o valor comercial. Las clases de las fuentes no son idénticas: las carpetas `Good`/`Bad` de Kaggle se mapean directamente, y `Fresh`/`Semi-Fresh`/`Rotten` de AgriFreshNET se mapean a `good`/`regular`/`bad`. Esa correspondencia debe explicarse y validarse, en particular porque casi toda la clase `regular` proviene de AgriFreshNET.

## 3. Datos disponibles y situación observada

Fuentes configuradas en `src/config.py`:

| Fuente | Archivos en el índice local | Observación |
| --- | ---: | --- |
| Fruit Quality Classification / `ryandpark` | 19.526 | Kaggle aporta principalmente `good` y `bad`; `Mixed` queda sin etiqueta por pieza hasta segmentarla y anotarla. |
| AgriFreshNET | 14.160 | Aporta las tres etiquetas mapeadas; las copias aumentadas se agrupan con su original. |
| Fotos propias | 42 | 41 tienen calidad confirmada; cuatro mangos siguen en revisión de segmentación y una foto requiere etiqueta por pieza. |
| **Total del índice** | **33.728** | El índice clasifica cada registro por fuente, producto, calidad disponible, grupo y estado. |

El conteo local actualizado es 16.408 `good`, 4.735 `regular`, 11.510 `bad` y 1.075 sin calidad asignable (`needs_instance_labels`); no quedan registros `pending_quality`. Además, hay 68 archivos en 17 grupos `conflicting_quality` por copias exactas con etiquetas incompatibles.

En fotos propias, el tamaño anotado está concentrado en 34 `medium` y 2 `small`; no hay ejemplos `large`. Las calidades de mango 001-006 quedaron confirmadas para la pieza en primer plano; tamaño e ID siguen vacíos. Mango_002 y mango_005 pasan segmentación automática; las otras cuatro fotos quedan en revisión por dos piezas detectadas. La foto de racimo de banano no entra como una sola pieza. La identificación incierta de bananos y mandarinas se conserva agrupada por sesión para reducir fuga entre particiones.

El inventario está agrupado para evitar que versiones de una misma captura pasen a distintos conjuntos: por ráfaga de tiempo/numeración en Kaggle, por original y aumentos en AgriFreshNET, y por pieza o sesión en fotos propias. En fuentes externas también se unen copias idénticas por contenido; los conflictos siguen marcados para excluirlos mientras no se resuelvan.

## 4. Qué está implementado en el repositorio activo

La estructura separa datos, código, documentación, experimentos y pruebas. El código implementado hasta ahora se concentra en la fase de preparación:

- `src/config.py`: rutas relativas, clases, nombres canónicos de productos, semilla 42, dimensiones 128×128, umbrales provisionales de tamaño, parámetros de segmentación y proporciones objetivo 70/15/15.
- `src/data/inventory.py`: lee las tres fuentes, aplica las etiquetas disponibles, registra los estados pendientes/conflictivos y construye grupos para las particiones.
- `src/data/segmentation.py`: intenta extraer la fruta usando color de fondo en Lab, máscara inicial, GrabCut, detección de piezas separadas/pegadas y controles de plausibilidad. Produce recorte con fondo para CNN, recorte enmascarado para características clásicas y diámetro equivalente normalizado.
- `src/data/prepare.py`: genera manifiestos, recortes y máscaras; conserva decisiones (`candidate`, `review`, `excluded`), estadística de imagen y estado/método de segmentación. Acepta contornos manuales revisados para algunos casos propios.
- `src/data/split.py`: asigna grupos completos a train/val/test, buscando cobertura de clases y proporciones; guarda `splits.csv` junto al manifiesto.
- `notebooks/01_eda.ipynb`: notebook breve de EDA del inventario: cobertura, clases, desbalance y ejemplos. No hay notebooks de entrenamiento ni evaluación en el repositorio actual.
- `docs/crisp-dm.md`, `docs/criterios-calidad.md`, `docs/revision-anotaciones.md` y el diagrama SVG: contienen el marco metodológico, criterios de etiqueta, auditoría de casos y decisiones iniciales de ética/impactos.
- `tests/`: pruebas unitarias sintéticas de inventario, preparación, segmentación y asignación por grupos. Están escritas, pero no se ejecutaron durante esta revisión.

**Estado real de artefactos locales:** `data/processed/index.csv` tiene 33.728 filas. El manifiesto completo (`data/processed/crops/manifest.csv`) tiene 23.706 `candidate`, 7.477 `review` y 2.545 `excluded`. La auditoría visual revisó una muestra estratificada de 44 candidatos y registró diez referencias de exclusión de grupos; la revisión individual del total de candidatos todavía está pendiente. El split completo existe con 23.687 filas: 16.456 train, 3.557 val y 3.674 test; grupos disjuntos, archivos y máscaras presentes, y 19 candidatos removidos por las exclusiones visuales. El split piloto sigue disponible como diagnóstico y no sustituye al completo.

No están implementados todavía: extracción formal de características, modelos (regresión logística de línea base, SVM, Random Forest y CNN), entrenamiento/tuning, evaluación, resultados o gráficas finales, guardado de modelos, ni aplicación Streamlit (`src/features/`, `src/models/`, `src/training/`, `src/evaluation/` y `src/app/` contienen sólo inicializadores vacíos). El README describe pasos futuros como si fueran el flujo previsto, no evidencia de una app o modelos ya disponibles.

## 5. Guía del semestre anterior

La referencia es la copia local **AgroQuality Classifier** 2026-1 en `C:\Users\damar\Downloads\apo-final-project-agroquality-classifier-main\apo-final-project-agroquality-classifier-main`. Es una entrega académica completa: documentación separada por fases CRISP-DM, auditoría exacta de duplicados y calidad de imagen, manifiestos y particiones, features manuales, Random Forest y SVM, CNN, comparación en validación/prueba y aplicación OpenCV/Tkinter. Reporta 27.061 imágenes limpias de Kaggle y colección propia, y documenta limitaciones de `Regular`, de generalización y de tamaño relativo.

Lo útil como guía es la cadena de evidencia: **decisión → artefacto reproducible → métrica → interpretación → límite/mitigación**. También sirve como patrón para auditar duplicados/calidad, calcular pesos de clase sólo desde train, separar validación de prueba, usar métricas macro y matriz de confusión, visualizar errores y explicar que el tamaño es una salida geométrica auxiliar. Sus cifras (incluido su desempeño alto) pertenecen únicamente a 2026-1: no se deben copiar como resultados ni usarse como expectativa para los datos actuales. Las fuentes, cantidades, estructura de despliegue y anotaciones difieren; el código del proyecto actual debe sostener cada afirmación de 2026-2.

## 6. Pendientes principales y riesgos

1. Resolver manualmente las cuatro máscaras de mango que siguen marcadas como dos frutas. Los dos candidatos de esa sesión ya quedaron en train; no habrá evaluación de mango hasta obtener grupos independientes.
2. Completar la revisión visual de recortes candidatos fuera de la muestra estratificada. `ok` sólo significa que pasó los controles automáticos; no garantiza segmentación correcta. Priorizar muestras por producto/fuente y casos múltiples o dudosos.
3. La partición completa ya se generó. Se verificaron exclusiones, cobertura global, grupos disjuntos, archivos y máscaras. El único grupo de mango se asignó a train y no aporta evaluación independiente.
4. Analizar la fuga de atajos por fuente/fondo/calidad: `regular` está fuertemente asociado a AgriFreshNET. Reportar cobertura y rendimiento por fuente/producto cuando el soporte lo permita; no presentar un promedio global sin sus límites.
5. Validar el tamaño: los umbrales son provisionales y las fotos no tienen encuadre controlado. El EDA documenta que 23 de 32 recortes propios anotados `medium` quedaron bajo el umbral provisional de `small`. Revisar máscaras y protocolo antes de afirmar que la categoría de tamaño es fiable.
6. Implementar y comparar al menos dos modelos clásicos más una CNN; ajustar hiperparámetros con validación cruzada que respete `group`, nunca usar test para seleccionar modelo y ajustar cualquier escalado/pesos sólo con train.
7. Evaluar con F1 macro y métricas por clase, matriz de confusión, errores `bad → good`, resultados por fuente/producto con soporte, generalización y sobreajuste; registrar gráficas, hiperparámetros, semilla y manifiesto usado.
8. Desarrollar la interfaz con el mismo preprocesamiento de entrenamiento y mensajes claros para fotos inválidas, varias piezas y límites de uso.
9. Completar PI1, PI2 y PI3 con evidencia matemática, ética e impactos; actualizar cifras y redactar informe IEEE de hasta 7 páginas, referencias verificadas y video de hasta 10 minutos.

## 7. Punto de partida para próximas sesiones

Tomar este archivo como mapa, pero comprobar el estado de Git y los datos generados al iniciar cada sesión. La documentación principal de decisiones está en [CRISP-DM](crisp-dm.md), [criterios de calidad](criterios-calidad.md) y [revisión de anotaciones](revision-anotaciones.md). La siguiente etapa lógica es cerrar calidad/anotación y auditoría completa de recortes antes de producir las particiones finales y entrenar. No asumir que el proyecto guía, el README o un manifiesto local ignorado por Git demuestran resultados reproducibles del estado remoto.
