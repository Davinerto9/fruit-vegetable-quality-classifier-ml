# Fundamentos matemáticos y teóricos

## 1. Definición del problema

Para cada imagen segmentada \(x_i\), la tarea principal es clasificación supervisada multiclase:

\[
f_\theta(x_i)=\hat y_i,\qquad y_i\in\{\mathrm{good},\mathrm{regular},\mathrm{bad}\}.
\]

El modelo estima una clase de apariencia externa visible. La predicción no representa inocuidad ni estado interno. La estimación geométrica se calcula de forma separada desde la máscara.

## 2. Por qué es un problema complejo de ingeniería

Las fuentes tienen formatos, iluminación, fondos y criterios de etiqueta diferentes; algunas imágenes contienen varias piezas, hay duplicados y etiquetas en conflicto, y las clases están desbalanceadas. El equipo debe resolver identificación de la pieza, calidad de anotación, fuga por capturas relacionadas, evaluación imparcial y uso responsable. Una métrica alta no basta si el modelo reconoce la fuente o falla de forma grave en una clase.

## 3. Representación de los datos

Los modelos clásicos reciben atributos de la región segmentada: histogramas de color, medidas de textura y descriptores de forma. La CNN recibe recortes RGB de tamaño 128 × 128. Los atributos numéricos se escalan dentro del entrenamiento de cada fold; los parámetros de preprocesamiento no se ajustan con validación o test.

Si \(A\) es el área de la máscara de la pieza en la imagen original y \(H,W\) son alto y ancho de esa imagen, el diámetro de un círculo de área equivalente es:

\[
d=2\sqrt{\frac{A}{\pi}},\qquad d_n=\frac{d}{\max(H,W)}.
\]

El valor \(d_n\) expresa ocupación relativa de la imagen. Sin calibración de distancia, perspectiva y escala, no es una medida de centímetros. Las categorías small, medium y large dependen de umbrales todavía provisionales y de pocas etiquetas propias.

## 4. Selección de modelos y alternativas

- **Regresión logística:** línea base sencilla sobre atributos manuales. Sirve para saber si los modelos más complejos agregan valor.
- **SVM:** busca una frontera de separación con margen amplio; los kernels permiten fronteras no lineales en el espacio de atributos.
- **Random Forest:** promedia decisiones de varios árboles y puede capturar interacciones entre atributos.
- **CNN:** combina filtros convolucionales y capas de clasificación para aprender patrones espaciales directamente de los pixeles.

SVM y Random Forest cumplen la comparación de dos métodos clásicos; la CNN satisface la comparación de deep learning. KNN, árbol individual y transfer learning son alternativas; su selección o descarte se justificará según los datos, el costo de entrenamiento, interpretabilidad y tiempo.

## 5. Entrenamiento y pérdida

Para la CNN, una salida softmax produce probabilidades por clase. La pérdida multiclase habitual es entropía cruzada:

\[
L=-\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{ik}\log(p_{ik}),\qquad K=3.
\]

Los modelos se entrenan solo con train. La búsqueda de hiperparámetros y la selección se hacen en validación cruzada por grupos; el test se conserva para una evaluación final. La semilla, las particiones, transformaciones e hiperparámetros se registran para reproducibilidad.

## 6. Desbalance y métricas

La exactitud puede ocultar un desempeño bajo en la clase menos representada. Para la clase \(k\):

\[
P_k=\frac{TP_k}{TP_k+FP_k},\qquad R_k=\frac{TP_k}{TP_k+FN_k},\qquad
F1_k=\frac{2P_kR_k}{P_k+R_k}.
\]

El F1 macro da el mismo peso a las tres clases:

\[
F1_{\mathrm{macro}}=\frac{1}{K}\sum_{k=1}^{K}F1_k,\qquad K=3.
\]

Se reportarán también precisión y exhaustividad por clase, matriz de confusión, exactitud balanceada y soporte. El error bad→good se revisa explícitamente por sus posibles consecuencias. Los pesos de clase, si se usan, se calculan solo con train.

## 7. Generalización y control de fuga

Imágenes de la misma pieza, ráfaga, original o aumento permanecen en un solo grupo y en una única partición. Los folds de validación respetan grupos. El conjunto de prueba no participa en la selección. Se comparan métricas de entrenamiento, validación y prueba, y se inspeccionan errores por fuente/producto con soporte reportado. Una diferencia grande entre train y val/test puede indicar sobreajuste; resultados concentrados por fuente pueden indicar atajo o confusión de etiquetas.

## 8. Estado de la fundamentación

Estas ecuaciones y fundamentos describen la estrategia planeada. No son evidencia de modelos ejecutados ni de resultados medidos. Se completarán con la configuración real, cálculos implementados, alternativas consideradas, gráficas e interpretación cuando finalicen modelado y evaluación.

## Referencias

- Universidad Icesi, *Competencias que se deben evaluar en el curso*, indicador PI3.
- IBM, *IBM SPSS Modeler CRISP-DM Guide*, versión 18.3.
- [Criterios de calidad visual](../criterios-calidad.md).
