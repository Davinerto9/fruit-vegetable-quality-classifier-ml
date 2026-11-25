# 6. Despliegue

## Objetivo

Una interfaz local de Streamlit permite probar el modelo seleccionado con una fotografía cargada o tomada con la cámara. Es una demostración académica: no significa que el sistema esté instalado en una operación comercial.

## Cómo ejecutarla

Desde la raíz del repositorio, con el ambiente activado y los checkpoints en `experiments/checkpoints/`:

```bash
streamlit run src/app/app.py
```

| Archivo | Necesario para | Cómo se obtiene |
| --- | --- | --- |
| `experiments/checkpoints/random_forest.joblib` | Clasificación (modelo principal) | `python -m src.training.classical_models` |
| `experiments/checkpoints/cnn_rgb.keras` | Comparación opcional con la CNN | `python -m src.training.train_cnn` |
| `experiments/results/*.json` | Mostrar el F1 de validation de cada modelo | Se generan al entrenar |

Los checkpoints no están en Git por tamaño: cada integrante debe tenerlos localmente. La app avisa si falta alguno o si la versión de scikit-learn instalada no es la 1.7.2 con la que se guardaron.

## Arquitectura

| Módulo | Responsabilidad |
| --- | --- |
| `src/app/app.py` | Solo interfaz: entrada (carga o cámara), resultados, avisos. |
| `src/app/inference.py` | Decodificar, segmentar, preprocesar, predecir y decidir si la foto se rechaza. No depende de Streamlit. |
| `src/data/segmentation.py` | `process()` y `model_inputs()`: los mismos que usa `src/data/prepare.py` al generar los datos de entrenamiento. |
| `src/features/classic.py` | `extract_classic_features_from_arrays()`: los mismos 359 atributos que el entrenamiento, desde memoria. |

Una prueba (`tests/test_inference.py`) verifica que los atributos calculados en memoria por la app son idénticos, valor por valor, a los calculados desde los PNG del entrenamiento.

## Flujo de predicción

1. El usuario carga una foto o la toma con la cámara (`st.camera_input`).
2. `segmentation.process()` estima el fondo, segmenta la pieza y cuenta las frutas.
3. Si la foto está en escala de grises, tiene varias piezas o la máscara es dudosa, la app **no clasifica**: explica el motivo y pide otra foto.
4. `segmentation.model_inputs()` produce el recorte y la máscara de 128×128.
5. Random Forest recibe los 359 atributos; la CNN (opcional) recibe el recorte con el fondo anulado.
6. La interfaz muestra la clase, las probabilidades, el contorno detectado, lo que recibe el modelo, la categoría de tamaño y $d_n$.
7. Si la probabilidad máxima es menor que `config.APP_MIN_CONFIDENCE` (0,55, provisional), o si RF y la CNN no coinciden, pide revisión manual.

## Decisiones de diseño

- **Random Forest como modelo principal:** mayor F1 macro en validation (0,842). La CNN se ofrece como comparación porque comete menos errores `bad→good` (6,2% frente a 7,9%).
- **Rechazar antes que adivinar:** una foto con varias piezas o con mala segmentación produciría atributos sin sentido; la app lo dice en lugar de mostrar una clase con falsa seguridad.
- **Lógica separada de la interfaz:** `inference.py` se prueba con pytest y se puede reutilizar en una simulación de línea de empaque.
- **Sin almacenamiento:** las fotos se procesan en memoria y no se escriben en disco.
- **Comunicación de límites:** la pantalla indica que la clase es apariencia externa, que no evalúa inocuidad y que el tamaño es relativo al encuadre.

## Pruebas de uso pendientes

- [ ] Demostrar carga y cámara con al menos una pieza buena, una regular y una mala.
- [ ] Mostrar dos entradas rechazadas (dos frutas; fondo parecido a la fruta).
- [ ] Medir la latencia por foto en el computador de la demo (para PI2, dimensión ambiental).
- [ ] Elegir `APP_MIN_CONFIDENCE` en validation y documentar el valor.
- [ ] Probar en celular a través de la red local (`streamlit run src/app/app.py --server.address 0.0.0.0`).

## Monitoreo previsto

Si se recopilan nuevas imágenes, se revisa su distribución por calidad, producto, cámara e iluminación y el desempeño por clase cuando haya etiquetas fiables. Nuevos productos, cambios de captura o caída del desempeño deben activar revisión y posible reentrenamiento. No se extiende el uso a otros contextos sin validación.

## Estado

Interfaz implementada y probada con imágenes sintéticas y fotos propias (segmentación y rechazo). Falta la demostración con los checkpoints reales y las pruebas de uso de la lista anterior. El cierre de la entrega incluirá el informe IEEE de máximo siete páginas, un video de hasta diez minutos y las referencias realmente utilizadas.
