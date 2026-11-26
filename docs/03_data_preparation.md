# 3. Preparación de los datos

## Objetivo de preparación

Convertir las fuentes heterogéneas en observaciones individuales, etiquetadas y trazables; revisar su segmentación; y construir particiones reproducibles que separen piezas relacionadas. La guía CRISP-DM recomienda documentar inclusiones, exclusiones, correcciones y transformaciones, y volver a explorar el resultado de operaciones importantes.

## Entrada y salida

**Entrada:** inventario de Kaggle, AgriFreshNET y fotos propias con rutas, fuente, producto, calidad disponible, grupo y estado.  
**Salida prevista:** manifiesto de revisión, recortes y máscaras alineados, atributos o tensores de entrada, y particiones train/validation/test sin fuga entre grupos.

## Selección y limpieza

1. Excluir las escenas Mixed sin etiquetas por pieza, los cuatro mangos que siguen con segmentación dudosa y los 17 grupos conflictivos.
2. Revisar el número de piezas por imagen. Cada fila de entrenamiento debe representar una pieza, no un racimo ni una escena con varios productos.
3. Conservar fuente, producto, grupo, decisión y motivo de revisión para auditoría. No usar metadatos de procedencia como atributos predictivos.
4. Detectar imágenes ilegibles, en escala de grises, desenfocadas o con contraste insuficiente; documentar el motivo de cada exclusión.
5. Revisar visualmente los recortes candidatos. El estado automático ok no demuestra que la máscara sea correcta.

El manifiesto completo local actualizado tiene 33.728 filas: **23.706 candidate, 7.477 review y 2.545 excluded**. El estado candidate incluye controles automáticos y revisión visual estratificada; no significa que cada uno de los 23.706 recortes haya recibido inspección visual individual.

## Segmentación y construcción de ejemplos

El código estima el fondo en espacio Lab, crea una máscara inicial, la refina con GrabCut y aplica controles de área y componentes. Produce un recorte para CNN y una versión con máscara para atributos clásicos. Hay contornos manuales para corregir tres incidencias propias del piloto.

Los recortes se llevan a **128 × 128 píxeles**. La máscara debe conservar alineación con el recorte. Se revisan casos con varias piezas, máscara parcial, sombras, fondo mezclado y producto cortado por el borde.

## Medida relativa de tamaño

Con el área \(A\) de la máscara en la imagen original, se calcula el diámetro equivalente \(d=2\sqrt{A/\pi}\) y luego \(d_n=d/\max(H,W)\), donde \(H,W\) son la altura y el ancho originales. \(d_n\) mide ocupación relativa de la fotografía, no tamaño real. Los umbrales vigentes (`small` < 0,33 ≤ `medium` < 0,55 ≤ `large`) se calibraron con las 35 fotos propias de una pieza con tamaño anotado y coinciden con 34 (97%). Siguen siendo umbrales de un protocolo de captura, no categorías validadas: solo hay dos ejemplos pequeños y ninguno grande. La justificación está en los [fundamentos matemáticos, §4](supporting_docs/mathematical_and_theoretical_foundations.md#4-tamaño).

## Integración y organización por grupos

Las fuentes se normalizan a las clases good, regular y bad, conservando procedencia y regla aplicada. El mapeo de AgriFreshNET es una aproximación que debe declararse. Las copias aumentadas se agrupan con su imagen original; las capturas relacionadas de Kaggle y las fotos propias de identidad incierta se mantienen juntas usando el campo group.

## Partición y prevención de fuga

El split busca proporciones aproximadas de 70/15/15 para entrenamiento, validación y prueba, con semilla 42. Cada grupo se asigna completo a una sola partición. Se verifican intersecciones, cobertura por calidad, fuente y producto, exclusiones y distribución obtenida. La validación cruzada también respeta grupos. Cualquier escalado, selección de atributos, cálculo de pesos de clase y aumento de datos se ajusta exclusivamente con entrenamiento.

### Versiones del reparto: v1, v2 y v3

Las tres versiones usan **los mismos recortes**; solo cambia qué fotos se consideran "la misma" y, por tanto, deben quedar juntas en *train*, validación o prueba.

| Versión | Qué se agrupa | Problema que corrige | Archivo |
| --- | --- | --- | --- |
| v1 | Una foto con sus copias aumentadas | — | `splits.csv` (`src/data/split.py`) |
| v2 | Además, las ráfagas de Kaggle (fotos tomadas segundos seguidos) | En v1, el 99% de las fotos de Kaggle en validación tenía una gemela en entrenamiento; el F1 de Kaggle subía unos 3,5 puntos (0,968 frente a 0,932). | `splits_v2_burst.csv` (`src/data/split_v2.py`) |
| **v3** | Además, las sesiones de AgriFreshNET (misma fruta y semana) | AgriFreshNET fotografía las mismas piezas durante semanas (carpetas `Fresh X(1-9)`, `Semi fresh X(9-20)`, `Rotten X(24-35)`: días desde la compra). En validación v2, 727 grupos de AgriFreshNET eran en realidad 82 sesiones de semana. | `splits_v3_<nivel>.csv` (`src/data/split_v3.py`) |

| | v1 | v2 |
| --- | ---: | ---: |
| Recortes | 23.687 | 23.543 |
| Grupos | 13.192 | 5.677 |
| train / val / test (recortes) | 16.456 / 3.557 / 3.674 | 16.272 / 3.795 / 3.476 |
| Ráfagas de Kaggle repartidas | 645 | 0 |

v3 se construye a partir de v2 (por eso se conservan los tres scripts). Elige el agrupamiento más amplio (semana, luego día, luego minuto) que siga dejando las tres calidades en cada partición y las proporciones 70/15/15 con un desvío de hasta 5 puntos. El F1 no interviene en esa elección. Las cifras de la tabla de v3 se agregan al generarlo ([plan simple](plan_simple.md), paso 1).

## Artefactos y estado

- data/processed/index.csv: inventario con 33.728 filas.
- data/processed/crops/manifest.csv: manifiesto completo con estado y motivo por registro.
- data/processed/pilot-crops/manifest.csv: piloto de 151 filas (105 candidatas, 37 por revisar y nueve excluidas).
- data/processed/pilot-crops/splits.csv: piloto de 98 recortes en 70 grupos (77 train, 10 val y 11 test), sin grupos compartidos y con las tres clases.
- data/processed/crops/splits.csv: split v1, conservado solo para comparar (tiene la fuga por ráfagas).
- data/processed/crops/splits_v2_burst.csv: split vigente.
- Código relacionado: [inventario](../src/data/inventory.py), [segmentación](../src/data/segmentation.py), [preparación](../src/data/prepare.py), [partición](../src/data/split.py) y [split v2](../src/data/split_v2.py).

**Pendiente:** completar la revisión visual de candidatos fuera de la muestra estratificada y resolver las máscaras de cuatro mangos. El grupo de mango con dos candidatos quedó completo en train; no hay soporte de mango en validación ni prueba, así que no se reportarán métricas para ese producto. Usar el split v2 para todo entrenamiento y evaluación; v1 y el piloto quedan solo como diagnóstico.

## Conclusión de la fase

El pipeline de preparación generó el manifiesto y la partición por grupos. La primera versión tenía una fuga por ráfagas en Kaggle, corregida en el split v2, que no tiene ninguna ráfaga repartida. Queda por comprobar el agrupamiento por fecha y hora en AgriFreshNET. La auditoría visual fue estratificada, no individual para todos los candidatos, y cuatro fotos de mango siguen en revisión de segmentación. El modelado inicial puede comenzar con los candidatos aceptados, dejando documentado que los resultados son provisionales hasta completar esa revisión.
