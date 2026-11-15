# Metodología CRISP-DM del clasificador de frutas y verduras

Universidad Icesi · Algoritmos y Programación III · 2026-2

El proyecto desarrolla un prototipo de inspección visual que recibe una foto de una pieza sobre fondo uniforme y estima su calidad externa (buena, regular o mala) y su tamaño relativo. La clasificación se plantea como apoyo a la revisión humana en mercados y comercios. La preparación de datos está en curso. El entrenamiento, la evaluación final y la interfaz corresponden a las fases siguientes.

## Flujo del proyecto

![Diagrama CRISP-DM del clasificador. Las seis fases conectan la definición de calidad, el análisis de las tres fuentes, la segmentación y partición, la comparación de modelos, la evaluación y la interfaz Streamlit. Los hallazgos retroalimentan las fases iniciales.](figures/crisp-dm.svg)

Las flechas punteadas indican cuándo volver a una fase anterior. Una concentración de errores en cierta fuente o producto obliga a revisar la cobertura y las etiquetas. Las dificultades observadas al tomar fotos en la app pueden cambiar el protocolo de captura y la definición del caso de uso. [Guía CRISP-DM de Chapman et al.](https://www.kde.cs.uni-kassel.de/lehre/ws2012-13/kdd/files/CRISPWP-0800.pdf)

## 1. Comprensión del negocio

En plazas de mercado y comercios, la clasificación manual por apariencia depende de la experiencia de quien inspecciona y de las condiciones de captura. Una inspección más consistente podría agilizar la separación de productos y evitar descartes por defectos cosméticos. También importa el error inverso, que ocurre cuando una pieza con deterioro visible recibe una buena valoración. Los beneficios son hipótesis que requieren medición.

La unidad de análisis es una pieza individual. La etiqueta describe la condición **externa visible**. La inocuidad y el estado interno requieren otros controles. La medida de tamaño expresa cuánto ocupa la pieza en la imagen y depende de la distancia, la perspectiva y la orientación. Para comparar capturas se necesita un encuadre consistente. Los criterios de éxito del prototipo son mejorar el F1 macro de la línea base en la evaluación final, revisar el reconocimiento de las clases `regular` y `bad`, y demostrar el flujo de captura y carga en la interfaz. [Criterios de calidad del equipo](criterios-calidad.md)

### Decisiones éticas y evidencia para PI1

| Situación | Principio del código IEEE | Medida y evidencia |
| --- | --- | --- |
| Capturas en comercios, hogares o desde la app | Privacidad y bienestar público | Encuadrar solo la pieza, revisar fotos y metadatos antes de compartir y obtener permiso si quedan personas identificables. La app se diseñará para procesar las capturas sin conservarlas por defecto. |
| Asociación entre la fuente de datos y la clase `regular` | Equidad y honestidad en las afirmaciones | El EDA cuantifica la concentración por fuente. En evaluación se reportarán errores y soporte por producto y procedencia, junto con los casos con cobertura insuficiente. |
| Deterioro visual interpretado como diagnóstico | Bienestar público y comunicación de los límites | La app indicará que la predicción describe la apariencia externa. La decisión de venta, consumo o descarte seguirá a cargo de quien inspecciona. |
| Uso de datos y herramientas de terceros | Reconocimiento de las contribuciones | El README acredita las fuentes y sus licencias. El inventario guarda la procedencia de cada foto y separa los conflictos de etiqueta para su revisión. |

[Código de ética del IEEE](https://ieee-cas.org/about/ieee-code-ethics) · [Créditos del proyecto](../README.md#créditos)

### Impactos iniciales y mitigación para PI2

| Dimensión | Beneficio potencial y riesgo | Mitigación y forma de contrastarlo |
| --- | --- | --- |
| Social | Una revisión más consistente podría reducir rechazos injustificados. La automatización puede trasladar errores a decisiones que afectan a vendedores y compradores. | Mantener revisión humana y evaluar confusiones entre deterioro leve y grave en fotos independientes. |
| Económica | El prototipo podría reducir el tiempo de inspección. Repetir capturas o corregir errores puede introducir costes. | Comparar el tiempo por pieza y registrar cuántos casos necesitan nueva captura o corrección en una prueba de uso. |
| Ambiental | Evitar descartes innecesarios puede reducir desperdicio. El entrenamiento y la inferencia tienen un coste computacional. | Comparar tiempo de entrenamiento e inferencia entre modelos y documentar el equipo utilizado. Validar la reducción de descartes con una prueba operativa. |
| Global | Un flujo de captura con celular facilita su réplica. Variedades, cámaras y prácticas de clasificación locales pueden cambiar la distribución de las imágenes y las etiquetas. | Validar por producto y origen antes de extender el uso a otras regiones o variedades y documentar los límites de transferencia. |

## 2. Comprensión de los datos

El inventario local registra **33.728 archivos en 17.481 grupos**. La versión de Kaggle de Fruit Quality Classification aporta 19.526 archivos, la descarga procesada de AgriFreshNET aporta 14.160 y el equipo aporta 42 fotos. La etiqueta de Kaggle se toma inicialmente de `Good` o `Bad`. AgriFreshNET aporta `Fresh`, `Semi-Fresh` y `Rotten`, que se traducen a `good`, `regular` y `bad` según los criterios del equipo. Las escenas de `Mixed` requieren una anotación por pieza si se incorporan. [Fuente de Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification) · [Ficha de AgriFreshNET](https://data.mendeley.com/datasets/42m5tb7yv9/1)

| Calidad | Archivos con etiqueta disponible | Grupos con esa etiqueta |
| --- | ---: | ---: |
| Buena | 16.364 | 8.265 |
| Regular | 4.705 | 1.642 |
| Mala | 11.510 | 6.678 |

El EDA contabiliza 32.579 archivos con etiqueta disponible en 16.581 grupos únicos. Un grupo puede contener más de una calidad, como ocurre con fotos propias de productos agrupadas por sesión cuando su identidad es incierta. AgriFreshNET aporta 4.695 de los 4.705 archivos regulares. Para manzana, guayaba, lima y granada faltan ejemplos etiquetados de esa clase. Si una cámara, un fondo o una forma de captura identifica la fuente, el modelo puede aprender esa asociación en lugar de las señales de deterioro. [Notebook de análisis exploratorio](../notebooks/01_eda.ipynb) · [Generación del inventario](../src/data/inventory.py)

Las fotos propias con calidad y tamaño anotados incluyen 33 medianas y dos pequeñas. El tamaño grande carece de ejemplos anotados en ese conjunto. La inspección de las fotos propias también encontró fondos con textura, otras piezas en el encuadre y tomas de una misma sesión. La auditoría de segmentación sobre fotos propias y una muestra externa registró imágenes en escala de grises, máscaras dudosas y escenas con múltiples piezas. [Anotaciones propias](../data/own/labels.csv) · [Revisión de anotaciones y segmentación](revision-anotaciones.md)

## 3. Preparación de los datos

1. **Control de etiquetas.** Mantener fuera del conjunto supervisado las escenas con etiqueta por pieza pendiente, los mangos aún sin anotación definitiva y los 17 grupos con conflicto confirmado. Si se aprovechan fotos `Mixed`, aislar y etiquetar cada pieza elegida y conservar el grupo de la foto original en todos los recortes.
2. **Segmentación y recorte.** Estimar el fondo a partir del borde de la imagen en Lab, ajustar el contorno con GrabCut y usar componentes conectados y watershed para detectar múltiples piezas. Registrar fotos ilegibles, grises, múltiples o con máscara dudosa. La preparación piloto guarda los recortes candidatos de 128 × 128 píxeles y sus máscaras alineadas. El estado de segmentación `ok` indica que el algoritmo pasó sus controles automáticos. La revisión visual comprueba que la máscara corresponde a la pieza. [Segmentación](../src/data/segmentation.py) · [Preparación por lotes](../src/data/prepare.py)
3. **Tamaño relativo.** Calcular el diámetro equivalente a partir del área del contorno en la imagen original y normalizarlo por el lado mayor de esa imagen. El cálculo precede al recorte y redimensionamiento. Con los umbrales provisionales, 23 de los 32 recortes propios candidatos con etiqueta mediana recibieron tamaño pequeño. Para validar los rangos se requieren fotos anotadas, máscaras revisadas y un protocolo de captura que controle distancia y encuadre. [Configuración](../src/config.py) · [Pruebas de segmentación](../tests/test_segmentation.py)
4. **Particiones y desbalance.** Crear un manifiesto de entrenamiento, validación y prueba con proporciones objetivo de 70/15/15 y semilla 42. Asignar cada `group` completo a una partición, conservar las variantes de cada toma juntas y estratificar por calidad cuando sea viable. Verificar cobertura por clase, fuente y producto y comprobar que no haya intersección de grupos entre particiones. Ajustar escaladores, pesos de clase y aumento de datos con el conjunto de entrenamiento, incluida la separación por grupos durante la validación cruzada.

**Evidencia actual.** Inventario, controles de segmentación y manifiesto piloto con decisiones por archivo. La revisión completa de los recortes y la creación de particiones están pendientes. [Revisión de anotaciones y segmentación](revision-anotaciones.md)

## 4. Modelado

La línea base será una regresión logística sobre histogramas de color en la región de la pieza. SVM y Random Forest compararán la utilidad de atributos de color, textura y forma extraídos con la máscara. La CNN pequeña entrenada desde cero recibirá los recortes RGB de 128 × 128 para aprender patrones espaciales de manchas, bordes y textura. La estimación del tamaño relativo utilizará la geometría de la máscara de forma común a los modelos.

Se ajustarán hiperparámetros de SVM y Random Forest mediante búsqueda por rejilla con validación cruzada por grupos. La CNN utilizará la validación para ajustar la arquitectura e hiperparámetros y controlar el sobreajuste. Se registrarán semilla, versión del manifiesto, representación de entrada, hiperparámetros, curvas y métricas de cada ejecución. El F1 macro guiará la selección en validación con revisión de los errores graves por clase.

**Evidencia por producir.** Configuración y resultados de cada entrenamiento en `experiments/`, con las decisiones de selección respaldadas por métricas de validación. El planteamiento matemático del equipo complementa esta fase en PI3 con la formulación, la justificación de las estrategias y sus alternativas, los cálculos implementados y la interpretación cuantitativa.

## 5. Evaluación

Con los modelos y las reglas de decisión congelados, evaluar la línea base, SVM, Random Forest y CNN sobre el mismo conjunto de prueba. Reportar F1 macro como métrica principal, precisión y exhaustividad por clase, y matrices de confusión con sus soportes. Revisar en particular los casos `bad` predichos como `good` y los errores entre clases contiguas. Comparar entrenamiento, validación y prueba para interpretar la generalización y el posible sobreajuste. Cuando haya soporte suficiente, desglosar por producto y fuente e inspeccionar ejemplos de error para distinguir fallos de máscara, etiqueta o predicción.

Para tamaño, comprobar el diámetro normalizado frente a contornos reales anotados en una muestra y evaluar el acuerdo de categorías cuando haya referencias con condiciones de captura comparables. Registrar la cobertura y los casos con máscara dudosa. Si faltan ejemplos grandes, reportar el alcance de esa limitación junto con las métricas que sí se puedan estimar.

**Evidencia por producir.** Comparación con la línea base, tablas y figuras vectoriales, análisis de errores y justificación de la decisión de despliegue. El análisis final de impactos se actualizará con los resultados del prototipo.

## 6. Despliegue

La app de Streamlit permitirá cargar una foto o capturarla con la cámara. Usará el mismo preprocesamiento que el modelo seleccionado y mostrará la calidad visual, el diámetro normalizado y la categoría de tamaño si los rangos han sido validados. Ante un encuadre con varias piezas, una lectura fallida o una máscara dudosa, pedirá una nueva foto con una sola pieza y fondo uniforme.

**Evidencia por producir.** Comando de ejecución, versión del modelo y pruebas de carga y cámara con entradas válidas y rechazadas. Los casos que fallen en la demostración se incorporarán a la revisión de datos y del protocolo de captura.

## Seguimiento de la entrega

- [x] Documentar las seis fases con evidencia disponible y publicar el diagrama personalizado.
- [x] Auditar los 17 grupos en conflicto y registrar la revisión visual de los seis mangos y el piloto de recortes.
- [ ] Confirmar las propuestas de calidad de los mangos sobre las piezas aisladas, recuperar sus tamaños mediante medidas o fotos de referencia, y verificar los IDs con quien tomó las fotos cuando la identidad sea dudosa.
- [x] Rehacer y verificar los contornos de las tres incidencias propias detectadas en el piloto. Implementar y probar las particiones por grupo.
- [ ] Completar la preparación del inventario y verificar la cobertura de las particiones del conjunto completo.
- [ ] Entrenar, evaluar y desplegar. Incorporar el planteamiento matemático, las conclusiones técnicas y el análisis final de impactos.
- [ ] Integrar la metodología y el diagrama en el informe de máximo siete páginas, preparar el video de hasta diez minutos y cotejar la rúbrica de INTU.

Las secciones 4, 6 y 7 de la consigna *Proyecto Final 26-2* guían el flujo, sus evidencias y el formato del informe. El documento *Competencias que se deben evaluar en el curso* detalla PI1, PI2 y PI3. La [guía de tareas del 13 de noviembre](../output/pdf/guia-proyecto-y-tareas.pdf) conserva el corte anterior del avance.

La entrega **AgroQuality Classifier** de un semestre anterior se consultó como referencia para organizar las evidencias de CRISP-DM, la auditoría de duplicados y el análisis de errores. La copia local está en `Downloads/apo-final-project-agroquality-classifier-main/apo-final-project-agroquality-classifier-main`. Al retomar el trabajo, sus capítulos de preparación y evaluación sirven como guía para documentar los artefactos y resultados de este proyecto.
