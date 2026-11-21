# Revisión de anotaciones y segmentación

La preparación parte del inventario local de 33.728 archivos. Esta revisión confirma el origen de los conflictos de etiqueta, registra observaciones sobre las fotografías de mango y comprueba qué ocurre al segmentar fotos reales.

## Conflictos de AgriFreshNET

Los 17 grupos marcados como `conflicting_quality` corresponden a naranja. Al calcular SHA-256 sobre los 68 archivos de esos grupos se obtuvieron 51 contenidos distintos. En cada grupo se confirmó una imagen con contenido idéntico en las carpetas `Fresh` y `Semi-Fresh`. Dieciséis grupos corresponden a los nombres `IMG-20251122-WA0000` a `IMG-20251122-WA0015` y el último a `IMG_20251116_220609831_HDR`.

Los 17 grupos, incluidas sus variantes, quedan fuera del conjunto supervisado mientras no exista una adjudicación de la calidad. Se conserva la etiqueta original y el estado de conflicto en el inventario para poder revisar la decisión. [Código de agrupación](../src/data/inventory.py)

## Fotos propias

Las etiquetas quedaron confirmadas para la pieza de primer plano: `good` para mango_002 y `regular` para mango_001 y mango_003 a mango_006. Las seis fotos permanecen en un solo grupo. Mango_002 y mango_005 pasan segmentación automática; las otras cuatro quedan en revisión porque el algoritmo detecta dos frutas. El tamaño y el ID de pieza siguen vacíos; el tamaño no se usa para el clasificador de calidad.

Se mantiene `mango_session_001` como un solo grupo. Las tomas de banano y mandarina con identidad incierta también conservan su agrupación por sesión. Se debe contrastar la identidad con quien tomó las fotos antes de asignar IDs adicionales. [Anotaciones propias](../data/own/labels.csv)

## Piloto de preparación

El piloto se ejecuta desde la raíz del proyecto después de generar el inventario.

```bash
python -m src.data.prepare --pilot
```

El piloto procesa las 42 fotos propias y toma una de cada 300 filas entre las imágenes externas con etiqueta disponible, sobre el orden del inventario. Es una comprobación del flujo. La muestra sistemática puede sobrerrepresentar algunas carpetas o productos.

| Fuente | Procesadas | Estado automático `ok` | Revisión automática | Descartadas por escala de grises | Recortes candidatos |
| --- | ---: | ---: | ---: | ---: | ---: |
| Propias | 42 | 37 | 5 | 0 | 34 |
| Externas | 109 | 71 | 36 | 2 | 71 |
| Total | 151 | 108 | 41 | 2 | 105 |

- `banana_good_007.jpg` muestra un racimo. Aunque la segmentación devolvió `ok`, la anotación de varias piezas impidió generar un recorte candidato.
- `banana_good_001.jpg` se marcó como dos piezas aunque visualmente se aprecia un solo banano.
- Durante el piloto, cuatro de seis fotos de mango se marcaron como múltiples. Tras confirmar las calidades, mango_002 y mango_005 pasan los controles automáticos; mango_001, mango_003, mango_004 y mango_006 siguen en revisión de segmentación.
- En la revisión visual de los recortes propios, las máscaras de `orange_good_003.jpg` y `orange_good_004.jpg` perdieron parte del contorno superior pese al estado `ok`.

Como diagnóstico de tamaño, se compararon las etiquetas propias con los umbrales provisionales. Entre los 34 recortes propios candidatos, los dos anotados como pequeños también dieron pequeño. De los 32 anotados como medianos, 23 dieron pequeño y nueve mediano. Estos conteos se calcularon antes de corregir las máscaras detectadas y con fotos de encuadres distintos. Los rangos actuales requieren validación con capturas controladas.

El comando genera `data/processed/pilot-crops/manifest.csv` con el estado de anotación, el resultado de segmentación, la decisión de preparación, el diámetro normalizado de los candidatos y las rutas relativas de sus imágenes y máscaras. Las salidas se guardan localmente. Las máscaras de 128 × 128 permiten extraer características de la pieza usando el mismo recorte que recibe la CNN. [Código de preparación](../src/data/prepare.py)

Para aplicar el flujo al inventario completo, ejecutar `python -m src.data.prepare`. La salida queda en `data/processed/crops/`. El manifiesto completo conserva las filas excluidas y omite segmentar fotos con anotación pendiente o en conflicto. Para generar otra corrida, archivar la salida anterior antes de ejecutar el comando. Después se usa la revisión de recortes para decidir qué registros pasan a la partición por grupos.

## Correcciones y particiones

Se trazaron y verificaron los contornos de `banana_good_001.jpg`, `orange_good_003.jpg` y `orange_good_004.jpg` sobre sus fotos originales. La preparación aplica estos contornos antes de calcular el diámetro y guarda la procedencia de cada máscara como `manual` o `automatic`. [Contornos revisados](../data/own/contours.json)

La revisión visual por estratos del manifiesto completo encontró tres escenas adicionales con varias frutas o productos dentro del mismo recorte: una lima clasificada como bad, una guayaba good y una granada good. Se registró una imagen por escena y el split aparta el grupo completo correspondiente. Las siete exclusiones iniciales del piloto se asociaron a siete grupos y retiraban 15 candidatos del manifiesto completo (no 430); al sumar estas tres escenas, la lista contiene diez referencias de exclusión. También se observaron manos, sombras y contornos incompletos en la muestra. [Grupos excluidos tras la revisión visual](../data/annotations/segmentation_exclusions.csv)

El comando `python -m src.data.split` filtra los grupos excluidos tras la revisión visual y asigna cada grupo restante a entrenamiento, validación o prueba. Busca proporciones cercanas a 70/15/15 por grupo y prioriza que train tenga las tres calidades y todos los productos disponibles; después favorece la cobertura de clases en val/test y el balance por calidad, fuente y producto. El único grupo con mango quedó en train; no permite evaluar mango de forma independiente. [Código de partición](../src/data/split.py)


La partición completa queda en `data/processed/crops/splits.csv`: 23.687 recortes, con 16.456 en train, 3.557 en val y 3.674 en test. Los 13.192 grupos no se comparten entre particiones; las diez referencias de exclusión se resolvieron y ningún grupo excluido aparece en el split. Todos los registros son candidatos únicos y tienen recorte y máscara. El grupo de mango (dos candidatos) quedó en train; no hay datos de mango en val/test para medir generalización.
La prueba con el manifiesto del piloto dejó 98 recortes en 70 grupos. Se verificó que los conjuntos de entrenamiento, validación y prueba tienen 50, 10 y 10 grupos, respectivamente, sin intersecciones y con ejemplos de las tres calidades.
