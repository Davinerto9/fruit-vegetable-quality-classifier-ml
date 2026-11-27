# 2. Comprensión de los datos

## Fuentes y alcance observado

El índice local contiene **36.265 archivos en 6.068 grupos** (conteos del [notebook 01](../notebooks/01_eda.ipynb)). Estos conteos describen el inventario, no el conjunto final de entrenamiento.

| Fuente | Archivos | Etiquetas disponibles y restricciones |
| --- | ---: | --- |
| Fruit Quality Classification (ryandpark) | 19.526 | Good y Bad se mapean inicialmente a good y bad. Hay 1.074 imágenes Mixed que requieren etiqueta por pieza. |
| AgriFreshNET | 14.160 | Fresh, Semi-Fresh y Rotten se mapean a good, regular y bad. Las copias aumentadas deben permanecer junto con su original. |
| AFruitDB | 2.537 | Grados 1st, 2nd y 3rd se mapean a good, regular y bad (ver abajo). |
| Fotos del equipo | 42 | 41 tienen calidad asignada; cuatro siguen en revisión de segmentación y una foto requiere etiquetas por pieza. |
| **Total** | **36.265** | 34.943 filas labeled, 247 en grupos conflictivos y 1.075 requieren etiqueta por pieza; ya no hay calidad pendiente. |

## Composición por calidad

Los conteos son de las 34.943 filas labeled (sin los grupos conflictivos, que no entran en el aprendizaje).

| Etiqueta | Archivos |
| --- | ---: |
| good | 17.294 |
| regular | 5.745 |
| bad | 11.904 |

De las 5.745 filas regular, 4.695 (82 %) proceden de AgriFreshNET. Esta asociación entre clase y fuente puede confundir el análisis: el modelo podría reconocer el estilo de captura de la fuente. Se deben revisar fondos, iluminación, producto y origen, e informar métricas por fuente cuando la muestra lo permita.

### Cuarta fuente: AFruitDB, para que `regular` no dependa de una sola fuente

Para atacar esa asociación se agregó [AFruitDB](https://data.mendeley.com/datasets/bz65dz2pbj/1) (Mojumdar et al., 2025, CC BY 4.0): fotos de celular tomadas en mercados de Bangladesh, con tres grados que coinciden con nuestras clases (1st = good, 2nd = regular, 3rd = bad). Se usan las frutas que el modelo ya conocía y además el **mango**, que pasa a ser el producto 13 (la uva birmana queda fuera):

| Fruta | good | regular | bad | Días | Sesiones (pausa > 10 min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Manzana | 198 | 172 | 112 | 7 | 9 |
| Banano | 243 | 239 | 2 | 5 | 4 |
| Mango | 251 | 250 | 117 | 9 | 11 |
| Papaya | 152 | 147 | 152 | — | — |
| Tomate | 145 | 231 | 126 | 3 | 4 |

Lo que se revisó y cómo se maneja:

- **Son ráfagas.** Las fotos de una misma fruta van cada ~5 segundos. Por eso el grupo de AFruitDB es la **sesión** (fotos con menos de 10 minutos entre sí, con todas sus calidades juntas), no la foto: si no, la fuga volvería. En la práctica son pocas sesiones independientes, así que AFruitDB aporta **variedad de fuente**, no miles de ejemplos independientes.
- **Son fotos originales de 12 MP con fondo** (papel blanco, cartón, superficies oscuras y a veces una mano con guante), no las de 224×224 del artículo. La segmentación aceptó las 24 de una muestra revisada a mano.
- **Hay un atajo posible dentro de la fuente.** Todas las manzanas `good` (de una variedad amarilla) y todos los tomates `good` se tomaron el 30 de septiembre de 2023; sus `regular` y `bad` son de otros días y, en la manzana, de variedades rojas. El banano es el caso limpio: todos sus días tienen `good` y `regular`. El mango también está bastante mezclado: `good` y `regular` comparten 4 de sus 9 días.
- **El mango solo tiene esta fuente** (más seis fotos de `data/own/`). Las seis de mango del pool de celular son copias de `data/own/`, así que no queda ninguna foto de mango para la prueba externa: su nota sale de val y test de AFruitDB (mangos de Bangladesh) y debe leerse así. Por eso se reporta el acierto en `regular` por fuente (`recall_regular_by_source`, notebook 04).


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

La auditoría registró 68 archivos de AgriFreshNET en los que una imagen idéntica aparece con dos etiquetas incompatibles. Al agrupar AgriFreshNET por producto y semana, esos conflictos quedan en **18 grupos con 247 archivos**. Se excluyen los grupos completos hasta adjudicar sus etiquetas.

Las seis fotos de mango **no son del equipo**: son las mismas seis de la carpeta compartida de la clase (*Photos of Fruits and Vegetables | APO III*, Google Drive) y deben citarse como aporte del pool de la clase. Tienen etiquetas confirmadas para la pieza del primer plano. Dos recortes pasan segmentación automática; cuatro siguen en revisión porque se detectan dos frutas en la escena. Las seis fotos permanecen en un mismo grupo. Las escenas con varias piezas no deben tratarse como una observación única. Las copias de una misma captura o sus aumentos se agrupan para evitar que aparezcan en distintos splits.

## Limitaciones y conclusión

La clase regular está concentrada en una fuente, faltan etiquetas por pieza para escenas Mixed y algunas imágenes requieren revisión de calidad o segmentación. El inventario y el EDA inicial orientan la preparación, pero no prueban que todos los recortes automáticos sean válidos ni que el conjunto final sea representativo.

**Estado:** inventario, conteos básicos y notebook EDA disponibles; la auditoría visual de candidatos fue estratificada y encontró escenas adicionales con varias piezas. Revisar más recortes, sobre todo manos, sombras y contornos incompletos, antes de presentar el conjunto como validado. Consultar los [criterios de calidad](criterios-calidad.md).

## Referencias de esta fase

- [Inventario de imágenes](../src/data/inventory.py).
- [Notebook de análisis exploratorio](../notebooks/01_eda.ipynb).
- [Fruit Quality Classification en Kaggle](https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification).
- [AgriFreshNET en Mendeley Data](https://data.mendeley.com/datasets/42m5tb7yv9/1).
- [AFruitDB en Mendeley Data](https://data.mendeley.com/datasets/bz65dz2pbj/1) y su artículo en *Data in Brief* ([DOI 10.1016/j.dib.2025.111380](https://doi.org/10.1016/j.dib.2025.111380)).
