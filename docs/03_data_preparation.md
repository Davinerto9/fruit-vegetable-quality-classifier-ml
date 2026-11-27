# 3. Preparación de los datos

## Objetivo de preparación

Convertir las fuentes heterogéneas en observaciones individuales, etiquetadas y trazables; revisar su segmentación; y construir particiones reproducibles que separen piezas relacionadas. La guía CRISP-DM recomienda documentar inclusiones, exclusiones, correcciones y transformaciones, y volver a explorar el resultado de operaciones importantes.

## Entrada y salida

**Entrada:** inventario de Kaggle, AgriFreshNET y fotos propias con rutas, fuente, producto, calidad disponible, grupo y estado.  
**Salida prevista:** manifiesto de revisión, recortes y máscaras alineados, atributos o tensores de entrada, y particiones train/validation/test sin fuga entre grupos.

## Selección y limpieza

1. Excluir las escenas Mixed sin etiquetas por pieza, los cuatro mangos que siguen con segmentación dudosa y los 18 grupos conflictivos (247 archivos).
2. Revisar el número de piezas por imagen. Cada fila de entrenamiento debe representar una pieza, no un racimo ni una escena con varios productos.
3. Conservar fuente, producto, grupo, decisión y motivo de revisión para auditoría. No usar metadatos de procedencia como atributos predictivos.
4. Detectar imágenes ilegibles, en escala de grises, desenfocadas o con contraste insuficiente; documentar el motivo de cada exclusión.
5. Revisar visualmente los recortes candidatos. El estado automático ok no demuestra que la máscara sea correcta.

El manifiesto completo tiene 34.943 filas (las fotos labeled): **26.015 candidate, 7.524 review y 1.404 excluded**. El estado candidate incluye controles automáticos y revisión visual estratificada; no significa que cada uno de los 23.706 recortes haya recibido inspección visual individual.

## Segmentación y construcción de ejemplos

El código estima el fondo en espacio Lab, crea una máscara inicial, la refina con GrabCut y aplica controles de área y componentes. Si la máscara queda con forma de media luna (podredumbre muy oscura sobre un fondo también oscuro, que se confunde con el fondo), se repite con un umbral de oscuridad más permisivo y se acepta solo si la forma se llena claramente (la solidez sube al menos 0,15) sin contar más frutas. En 156 fotos revisadas (AFruitDB, propias, Drive y mangos nuevos) esto cambió solo los dos mangos podridos que lo necesitaban; si un lado entero de la fruta es negro sobre un fondo oscuro, sigue sin poder separarse por color. Produce un recorte para CNN y una versión con máscara para atributos clásicos. Hay contornos manuales para corregir tres incidencias propias del piloto.

Los recortes se llevan a **128 × 128 píxeles**. La máscara debe conservar alineación con el recorte. Se revisan casos con varias piezas, máscara parcial, sombras, fondo mezclado y producto cortado por el borde.

## Medida relativa de tamaño

Con el área \(A\) de la máscara en la imagen original, se calcula el diámetro equivalente \(d=2\sqrt{A/\pi}\) y luego \(d_n=d/\max(H,W)\), donde \(H,W\) son la altura y el ancho originales. \(d_n\) mide ocupación relativa de la fotografía, no tamaño real. Los umbrales vigentes (`small` < 0,33 ≤ `medium` < 0,55 ≤ `large`) se calibraron con las 35 fotos propias de una pieza con tamaño anotado y coinciden con 34 (97%). Siguen siendo umbrales de un protocolo de captura, no categorías validadas: solo hay dos ejemplos pequeños y ninguno grande. La justificación está en los [fundamentos matemáticos, §4](supporting_docs/mathematical_and_theoretical_foundations.md#4-tamaño).

## Integración y organización por grupos

Las fuentes se normalizan a las clases good, regular y bad, conservando procedencia y regla aplicada. El mapeo de AgriFreshNET es una aproximación que debe declararse. Las copias aumentadas se agrupan con su imagen original; las capturas relacionadas de Kaggle y las fotos propias de identidad incierta se mantienen juntas usando el campo group.

## Partición y prevención de fuga

El split busca proporciones aproximadas de 70/15/15 para entrenamiento, validación y prueba, con semilla 42. Cada grupo se asigna completo a una sola partición. Se verifican intersecciones, cobertura por calidad, fuente y producto, exclusiones y distribución obtenida. La validación cruzada también respeta grupos. Cualquier escalado, selección de atributos, cálculo de pesos de clase y aumento de datos se ajusta exclusivamente con entrenamiento.

### Cómo llegamos al reparto final

Los tres intentos usaron **los mismos recortes**; solo cambió qué fotos se consideran "la misma" y, por tanto, deben quedar juntas en *train*, val o test. Cada vez que encontramos fotos "gemelas" repartidas, agrupamos más grueso.

| Intento | Qué se agrupa | Qué problema encontramos |
| --- | --- | --- |
| 1 | Una foto con sus copias aumentadas | El 99 % de las fotos de Kaggle en val tenía una gemela (misma ráfaga, segundos de diferencia) en train; el F1 de Kaggle subía unos 3,5 puntos (0,968 frente a 0,932). |
| 2 | Además, las ráfagas de Kaggle | AgriFreshNET fotografía las mismas piezas durante semanas (carpetas `Fresh X(1-9)`, `Semi fresh X(9-20)`, `Rotten X(24-35)`: días desde la compra). En val, 727 grupos de AgriFreshNET eran en realidad 82 semanas de fotos. |
| **Final** | Además, las fotos de AgriFreshNET del mismo producto y la misma **semana** | — |

| | Intento 1 | Intento 2 |
| --- | ---: | ---: |
| Recortes | 23.687 | 23.543 |
| Grupos | 13.192 | 5.677 |
| train / val / test (recortes) | 16.456 / 3.557 / 3.674 | 16.272 / 3.795 / 3.476 |
| Ráfagas de Kaggle repartidas | 645 | 0 |

La semana es el agrupamiento más grueso (frente a día o minuto) que todavía deja las tres calidades en cada partición y las proporciones 70/15/15 con un desvío de hasta 5 puntos; `src/data/split.py` verifica esas reglas y se detiene si no se cumplen. El F1 no intervino en esa elección. `python -m src.main fuga` mide cuánto se inflaba la nota con cada agrupamiento; las cifras del reparto final se agregan al generarlo.

## Artefactos y estado

- `data/processed/index.csv`: inventario con 36.265 filas (`python -m src.main inventario`).
- `data/processed/crops/manifest.csv`: decisión y motivo para cada foto (`python -m src.main recortar`).
- `data/processed/crops/splits.csv`: el reparto final (`python -m src.main dividir`).
- `data/processed/features.npz`: los 359 atributos de cada recorte (`python -m src.main atributos`).
- `data/processed/pilot-crops/`: piloto de 151 fotos para revisar la segmentación (`python -m src.data.prepare --pilot`).
- Código: [inventario](../src/data/inventory.py), [segmentación](../src/data/segmentation.py), [recortes](../src/data/prepare.py), [reparto](../src/data/split.py) y [atributos](../src/data/features.py).

**Limitación:** la revisión visual fue por muestra estratificada, no foto por foto. De las 41 fotos propias con estado, 37 pasan la segmentación y entran al reparto; las demás quedan fuera. El mango entra como producto 13 gracias a AFruitDB (618 fotos, 11 sesiones).

## Conclusión de la fase

El pipeline de preparación genera el manifiesto, el reparto por grupos y los atributos con un comando por paso. Las fugas por ráfagas de Kaggle y por semanas de AgriFreshNET quedan corregidas en el reparto final. La auditoría visual fue estratificada, no individual para todos los candidatos.
