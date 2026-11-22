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

Con el área \(A\) de la máscara en la imagen original, se calcula el diámetro equivalente \(d=2\sqrt{A/\pi}\) y luego \(d_n=d/\max(H,W)\), donde \(H,W\) son la altura y el ancho originales. \(d_n\) mide ocupación relativa de la fotografía, no tamaño real. Los umbrales de small, medium y large son provisionales. No se deben presentar como categorías validadas: el conjunto propio tiene 34 medianas y dos pequeñas, sin ejemplos grandes, y las condiciones de captura varían.

## Integración y organización por grupos

Las fuentes se normalizan a las clases good, regular y bad, conservando procedencia y regla aplicada. El mapeo de AgriFreshNET es una aproximación que debe declararse. Las copias aumentadas se agrupan con su imagen original; las capturas relacionadas de Kaggle y las fotos propias de identidad incierta se mantienen juntas usando el campo group.

## Partición y prevención de fuga

El split final busca proporciones aproximadas de 70/15/15 para entrenamiento, validación y prueba, con semilla 42. Cada grupo se asigna completo a una sola partición. Se verifican intersecciones, cobertura por calidad, fuente y producto, exclusiones y distribución obtenida. La validación cruzada también debe respetar grupos. Cualquier escalado, selección de atributos, cálculo de pesos de clase y aumento de datos se ajusta exclusivamente con entrenamiento.

## Artefactos y estado

- data/processed/index.csv: inventario con 33.728 filas.
- data/processed/crops/manifest.csv: manifiesto completo con estado y motivo por registro.
- data/processed/pilot-crops/manifest.csv: piloto de 151 filas (105 candidatas, 37 por revisar y nueve excluidas).
- data/processed/pilot-crops/splits.csv: piloto de 98 recortes en 70 grupos (77 train, 10 val y 11 test), sin grupos compartidos y con las tres clases.
- Código relacionado: [inventario](../src/data/inventory.py), [segmentación](../src/data/segmentation.py), [preparación](../src/data/prepare.py) y [partición](../src/data/split.py).

**Pendiente:** completar la revisión visual de candidatos fuera de la muestra estratificada y resolver las máscaras de cuatro mangos. El grupo de mango con dos candidatos quedó completo en train; no hay soporte de mango en validación ni prueba, así que no se reportarán métricas para ese producto. La partición completa ya está generada y verificada estructuralmente; no sustituirla por el split piloto.

## Conclusión de la fase

El pipeline de preparación generó el manifiesto y la partición completa por grupos; la separación se verificó sin grupos compartidos y con archivos/máscaras presentes. La auditoría visual fue estratificada, no individual para todos los candidatos, y cuatro fotos de mango siguen en revisión de segmentación. El modelado inicial puede comenzar con los candidatos aceptados, dejando documentado que los resultados son provisionales hasta completar esa revisión.
