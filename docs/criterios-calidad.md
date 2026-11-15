# Criterios de calidad visual

La etiqueta se asigna a una pieza individual según lo que se ve en la foto. Para juzgar el color, se toma como referencia la coloración habitual del producto y de su etapa de maduración.

| Calidad | Criterio |
| --- | --- |
| Buena (`good`) | La pieza está completa, conserva su coloración habitual y carece de zonas oscuras asociadas a podredumbre. |
| Regular (`regular`) | La pieza muestra deterioro inicial, como marcas, golpes, manchas localizadas o una abertura leve. Los cambios de color son pequeños o se concentran alrededor de la marca. |
| Mala (`bad`) | La pieza presenta hongos visibles, pérdida marcada de su coloración habitual o señales claras de podredumbre. Una abertura con tejido visiblemente decolorado y deteriorado también corresponde a esta clase. Por ejemplo, una manzana con una zona marrón extensa y tejido visiblemente deteriorado. |

Si una pieza cumple criterios de dos clases, se asigna la de mayor deterioro visible. Una abertura leve puede ser `regular`. Si hay hongos o podredumbre clara, se etiqueta como `bad`. Una zona oscura propia de la variedad o de su maduración habitual se evalúa dentro de ese contexto. Si la luz, el encuadre o el desenfoque impiden distinguir la causa de una marca, la foto queda pendiente de revisión.

## Correspondencia con las fuentes

- En **Kaggle**, `Good` y `Bad` corresponden inicialmente a `good` y `bad`. La carpeta `Mixed` agrupa fotos con piezas de distintas calidades. Si se aprovechan, se etiqueta cada pieza después de separarla.
- En **AgriFreshNET**, `Fresh`, `Semi-Fresh` y `Rotten` se registran como `good`, `regular` y `bad`. El equipo revisó muestras de `Semi-Fresh` y confirmó su correspondencia con `regular`.
- En las **fotos propias**, se aplican directamente los criterios de la tabla al etiquetar o revisar cada foto.

## Inventario de imágenes

`data/processed/index.csv` registra la ruta (`filename`), producto, calidad, fuente, tamaño, `id_fruta`, grupo y estado. Los tamaños vienen de la anotación del equipo cuando están disponibles. El identificador `orange` conserva la convención del proyecto para las mandarinas registradas en las notas de las fotos propias.

- `id_fruta` identifica una pieza propia cuando hay evidencia visual suficiente. La forma, el tallo y la posición de las marcas ayudan a reconocerla. Si la identificación es dudosa, el campo queda vacío.
- `group` es la unidad que debe mantenerse junta al dividir los datos. En las fotos propias agrupa por producto y sesión cuando varias imágenes podrían mostrar la misma pieza. En ryandpark agrupa las ráfagas de captura: las fotos tomadas en el mismo minuto, según la fecha y hora del nombre, o con números consecutivos en bloques de diez cuando el nombre no tiene fecha. Así, las tomas de la misma escena con segundos de diferencia no quedan en particiones distintas. En AgriFreshNET agrupa la foto original con sus copias aumentadas y sus variantes. En ambas fuentes une las copias idénticas por contenido. Un nombre genérico repetido en distintas etiquetas de AgriFreshNET se conserva en grupos distintos si corresponde a imágenes diferentes.
- `status` distingue registros con etiqueta disponible (`labeled`), escenas que requieren recortar y etiquetar cada pieza (`needs_instance_labels`), mangos por anotar (`pending_quality`) y grupos con una copia idéntica etiquetada con dos calidades distintas (`conflicting_quality`). Una ráfaga que contiene fotos diferentes de calidades distintas no es un conflicto. El valor `labeled` indica que hay una etiqueta de origen. La revisión de cuántas piezas contiene cada foto corresponde a la segmentación.

El agrupamiento externo identifica variantes de la misma toma. Dos fotos distintas de la misma pieza pueden requerir una revisión adicional cuando la fuente carece de un identificador físico.
