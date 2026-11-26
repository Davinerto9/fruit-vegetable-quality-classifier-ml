# 2. Comprensión de los datos

## Fuentes y alcance observado

El índice local contiene **33.728 archivos en 17.481 grupos**. Estos conteos describen el inventario, no el conjunto final de entrenamiento.

| Fuente | Archivos | Etiquetas disponibles y restricciones |
| --- | ---: | --- |
| Fruit Quality Classification (ryandpark) | 19.526 | Good y Bad se mapean inicialmente a good y bad. Hay 1.074 imágenes Mixed que requieren etiqueta por pieza. |
| AgriFreshNET | 14.160 | Fresh, Semi-Fresh y Rotten se mapean a good, regular y bad. Las copias aumentadas deben permanecer junto con su original. |
| Fotos del equipo | 42 | 41 tienen calidad asignada; cuatro siguen en revisión de segmentación y una foto requiere etiquetas por pieza. |
| **Total** | **33.728** | 32.585 filas labeled, 68 en conflicto y 1.075 requieren etiqueta por pieza; ya no hay calidad pendiente. |

## Composición por calidad

Los conteos reflejan las etiquetas registradas en el índice e incluyen 68 registros de grupos conflictivos. Esos grupos no deben entrar en el aprendizaje mientras no se resuelvan.

| Etiqueta | Archivos |
| --- | ---: |
| good | 16.408 |
| regular | 4.735 |
| bad | 11.510 |

De las 4.735 filas regular, 4.720 proceden de AgriFreshNET. Esta asociación entre clase y fuente puede confundir el análisis: el modelo podría reconocer el estilo de captura de la fuente. Se deben revisar fondos, iluminación, producto y origen, e informar métricas por fuente cuando la muestra lo permita.

En las fotos propias hay 34 tamaños medium y dos small anotados; no hay ejemplos large. Estos datos no bastan para validar categorías de tamaño.

## Exploración y visualizaciones

El notebook [01_eda.ipynb](../notebooks/01_eda.ipynb) incluye conteos por fuente, producto y clase, desbalance y ejemplos. La exploración debe ampliarse con:

- imágenes y grupos por etiqueta, fuente y producto;
- resolución, brillo, contraste, nitidez y escala de grises;
- ejemplos de las tres clases y comparación visual entre fuentes;
- número de piezas por imagen y calidad de máscaras en muestras por fuente y producto;
- distribución del diámetro normalizado y comparación con las etiquetas de tamaño disponibles.

Las gráficas deben mostrar conteos y soporte, para que las diferencias se interpreten considerando cuántos ejemplos las producen.

## Calidad, conflictos y hallazgos

La auditoría registró **17 grupos de AgriFreshNET en conflicto**, con 68 archivos y 51 contenidos distintos. En cada grupo, una imagen idéntica aparece con dos etiquetas incompatibles. Se excluyen los grupos completos hasta adjudicar sus etiquetas.

Las seis fotos de mango **no son del equipo**: son las mismas seis de la carpeta compartida de la clase (*Photos of Fruits and Vegetables | APO III*, Google Drive) y deben citarse como aporte del pool de la clase. Tienen etiquetas confirmadas para la pieza del primer plano. Dos recortes pasan segmentación automática; cuatro siguen en revisión porque se detectan dos frutas en la escena. Las seis fotos permanecen en un mismo grupo. Las escenas con varias piezas no deben tratarse como una observación única. Las copias de una misma captura o sus aumentos se agrupan para evitar que aparezcan en distintos splits.

## Limitaciones y conclusión

La clase regular está concentrada en una fuente, faltan etiquetas por pieza para escenas Mixed y algunas imágenes requieren revisión de calidad o segmentación. El inventario y el EDA inicial orientan la preparación, pero no prueban que todos los recortes automáticos sean válidos ni que el conjunto final sea representativo.

**Estado:** inventario, conteos básicos y notebook EDA disponibles; la auditoría visual de candidatos fue estratificada y encontró escenas adicionales con varias piezas. Revisar más recortes, sobre todo manos, sombras y contornos incompletos, antes de presentar el conjunto como validado. Consultar los [criterios de calidad](criterios-calidad.md).

## Referencias de esta fase

- [Inventario de imágenes](../src/data/inventory.py).
- [Notebook de análisis exploratorio](../notebooks/01_eda.ipynb).
- [Fruit Quality Classification en Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification).
- [AgriFreshNET en Mendeley Data](https://data.mendeley.com/datasets/42m5tb7yv9/1).
