# Revisión de anotaciones y segmentación

La preparación parte del inventario local de 33.728 archivos. Esta revisión confirma el origen de los conflictos de etiqueta, registra observaciones sobre las fotografías de mango y comprueba qué ocurre al segmentar fotos reales.

## Conflictos de AgriFreshNET

Los 17 grupos marcados como `conflicting_quality` corresponden a naranja. Al calcular SHA-256 sobre los 68 archivos de esos grupos se obtuvieron 51 contenidos distintos. En cada grupo se confirmó una imagen con contenido idéntico en las carpetas `Fresh` y `Semi-Fresh`. Dieciséis grupos corresponden a los nombres `IMG-20251122-WA0000` a `IMG-20251122-WA0015` y el último a `IMG_20251116_220609831_HDR`.

Los 17 grupos, incluidas sus variantes, quedan fuera del conjunto supervisado mientras no exista una adjudicación de la calidad. Se conserva la etiqueta original y el estado de conflicto en el inventario para poder revisar la decisión. [Código de agrupación](../src/data/inventory.py)

## Fotos propias

Las observaciones de los seis mangos quedaron en la columna `notes` de las anotaciones. La propuesta es `good` para el mango 002 y `regular` para los mangos 001 y 003 a 006, con base en la pieza de primer plano. Las seis fotos tienen otros productos en el encuadre. Las de los mangos 003 y 006 requieren especial revisión del roce y de la lesión oscura, respectivamente. Quedan pendientes la confirmación de la etiqueta sobre la pieza aislada y una referencia de tamaño. El encuadre de las fotos solo permite estimar una medida relativa a la imagen.

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
- Cuatro de seis fotos de mango se marcaron como múltiples. Las otras dos pasaron el control automático y quedaron excluidas por su anotación pendiente.
- En la revisión visual de los recortes propios, las máscaras de `orange_good_003.jpg` y `orange_good_004.jpg` perdieron parte del contorno superior pese al estado `ok`.

Como diagnóstico de tamaño, se compararon las etiquetas propias con los umbrales provisionales. Entre los 34 recortes propios candidatos, los dos anotados como pequeños también dieron pequeño. De los 32 anotados como medianos, 23 dieron pequeño y nueve mediano. Estos conteos se calcularon antes de corregir las máscaras detectadas y con fotos de encuadres distintos. Los rangos actuales requieren validación con capturas controladas.

El comando genera `data/processed/pilot-crops/manifest.csv` con el estado de anotación, el resultado de segmentación, la decisión de preparación, el diámetro normalizado de los candidatos y las rutas relativas de sus imágenes y máscaras. Las salidas se guardan localmente. Las máscaras de 128 × 128 permiten extraer características de la pieza usando el mismo recorte que recibe la CNN. [Código de preparación](../src/data/prepare.py)

Para aplicar el flujo al inventario completo, ejecutar `python -m src.data.prepare`. La salida queda en `data/processed/crops/`. El manifiesto completo conserva las filas excluidas y omite segmentar fotos con anotación pendiente o en conflicto. Para generar otra corrida, archivar la salida anterior antes de ejecutar el comando. Después se usa la revisión de recortes para decidir qué registros pasan a la partición por grupos.

## Correcciones y particiones

Se trazaron y verificaron los contornos de `banana_good_001.jpg`, `orange_good_003.jpg` y `orange_good_004.jpg` sobre sus fotos originales. La preparación aplica estos contornos antes de calcular el diámetro y guarda la procedencia de cada máscara como `manual` o `automatic`. [Contornos revisados](../data/own/contours.json)

El comando `python -m src.data.split` toma los recortes candidatos y asigna cada grupo completo a entrenamiento, validación o prueba. Busca proporciones cercanas a 70/15/15 por grupo y prioriza la cobertura de las tres calidades al comparar asignaciones. El resumen en consola muestra la distribución por calidad, fuente y producto. La prueba con el manifiesto histórico del piloto separó sus 77 grupos en 53/12/12 y conservó las tres calidades en cada partición. [Código de partición](../src/data/split.py)
