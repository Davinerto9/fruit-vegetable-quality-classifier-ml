# 1. Comprensión del negocio

## Contexto del problema

En plazas de mercado y comercios, la clasificación de frutas y verduras por apariencia depende de la experiencia de cada persona y de las condiciones de observación. Una revisión asistida por visión por computadora podría aportar consistencia a una primera inspección. También puede cometer errores que afecten a vendedores y compradores, por lo que se plantea como apoyo y no como autoridad de decisión.

## Objetivo del proyecto

Construir y evaluar un prototipo reproducible que, a partir de la foto de una pieza individual sobre fondo sencillo:

- identifique qué fruta o verdura es, entre los 12 productos con datos de entrenamiento, y diga cuándo no la reconoce;
- clasifique su apariencia externa como good, regular o bad;
- calcule el diámetro equivalente normalizado de la pieza en la imagen;
- acepte una imagen cargada o capturada con cámara desde una interfaz sencilla.

La clasificación no evalúa inocuidad ni estado interno. La medida de tamaño no representa centímetros porque no se controla distancia, perspectiva ni escala física.

## Objetivo analítico y criterios de éxito

El objetivo analítico es comparar una línea base, dos modelos clásicos de aprendizaje automático y una CNN pequeña usando grupos de imágenes separados entre entrenamiento, validación y prueba.

| Criterio | Evidencia |
| --- | --- |
| El modelo seleccionado supera la línea base en F1 macro sobre el test reservado. | Tabla de métricas, matriz de confusión y soporte por clase bajo el mismo protocolo. |
| Los errores importantes no quedan ocultos por el promedio. | Exhaustividad por clase y análisis de bad predicha como good; errores por fuente y producto cuando haya soporte. |
| No hay fuga entre particiones. | Verificación de que ningún grupo aparezca en más de un split. |
| La interfaz demuestra el flujo previsto. | Prueba de carga y cámara, entradas válidas e inválidas y versión del modelo. |
| Se demuestran beneficios operativos. | No se afirmarán sin un estudio de uso que mida tiempo, desacuerdos o desperdicio. |

Después de medir la línea base y antes de comparar candidatos, el equipo debe registrar qué diferencia en validación considera útil para elegir un modelo. El conjunto de prueba queda reservado para la evaluación final.

## Alcance, recursos y restricciones

El alcance académico cubre los datos, la clasificación visual, la estimación geométrica relativa y una demostración local. Las personas interesadas son el equipo, el profesorado y, en una prueba posterior, quienes realizan inspección manual. El sistema no automatiza decisiones de consumo, descarte, compra o comercialización.

- **Recursos:** Python, bibliotecas de visión por computadora y ML, tres fuentes descargadas localmente, 42 fotos propias y tiempo del equipo para revisar anotaciones.
- **Requisitos:** foto individual, trazabilidad de fuente, etiquetas consistentes, grupos sin fuga entre particiones, ajuste con validación y comunicación de límites.
- **Supuestos:** el mapeo de categorías de cada fuente es una aproximación; el modelo debe aprender señales de deterioro y no solo fondo, cámara o fuente.
- **Restricciones:** fuentes con criterios y capturas distintas; pocas etiquetas propias de tamaño; no hay escala física; datos descargados y procesados no se incluyen en Git.

## Riesgos y contingencias

| Riesgo | Consecuencia | Respuesta |
| --- | --- | --- |
| La clase regular se concentra en AgriFreshNET. | El modelo podría reconocer la fuente en lugar del deterioro. | Analizar cobertura y errores por fuente; restringir conclusiones si fuente y clase no pueden distinguirse. |
| Etiquetas o máscaras dudosas. | Ruido en entrenamiento y evaluación. | Mantener el motivo en el manifiesto, revisar muestras y excluir casos no recuperables. |
| Falta de ejemplos grandes o escala física. | No se puede sostener una categoría fiable de tamaño físico. | Reportar diámetro relativo; mantener categorías como experimentales hasta validarlas. |
| Falta de tiempo o capacidad de cómputo. | Comparación incompleta. | Priorizar modelos pequeños y dejar constancia de lo que no se logró implementar. |
| Privacidad en imágenes propias o de uso. | Exposición de personas o información identificable. | Encuadrar el producto, revisar metadatos y pedir autorización cuando haya personas identificables. |

## Costos, beneficios, ética e impactos

Los costos que deben documentarse son captura y anotación, almacenamiento, procesamiento, entrenamiento y desarrollo de la interfaz. Los beneficios potenciales incluyen mayor consistencia y menor tiempo de inspección, pero ninguno está medido. Una prueba de uso futura deberá compararlos con errores y reintentos.

Las decisiones éticas y la matriz de impactos para PI1 y PI2 se desarrollan en [Ética e impactos](supporting_docs/ethics_considerations.md). El sistema debe mantener revisión humana, describir la condición externa visible y no inferir inocuidad.

## Referencias de esta fase

- Universidad Icesi, Proyecto Final 26-2: Algoritmos y Programación III.
- IBM, IBM SPSS Modeler CRISP-DM Guide, versión 18.3.
- [Criterios de calidad visual del equipo](criterios-calidad.md).
