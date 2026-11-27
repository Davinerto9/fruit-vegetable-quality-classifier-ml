# 6. Despliegue

## Objetivo

Una interfaz local de Streamlit permite probar el modelo seleccionado con una fotografía cargada o tomada con la cámara. Es una demostración académica: no significa que el sistema esté instalado en una operación comercial.

## Cómo ejecutarla

Desde la raíz del repositorio, con el ambiente activado:

```bash
streamlit run app/app.py
```

La app usa tres archivos de `models/`, que crea `python -m src.main entrenar` y que se suben a Git:

| Archivo | Para qué |
| --- | --- |
| `models/fruta_random_forest.joblib` | Qué fruta es |
| `models/calidad_random_forest.joblib` | Estado (Random Forest) |
| `models/calidad_cnn.keras` | Estado (CNN) y el promedio con el Random Forest; necesita TensorFlow |

El F1 de val que muestra la barra lateral sale de `results/entrenamiento_*.json` y `results/evaluacion_val.json`.

La app muestra primero **qué fruta es**; si la probabilidad de la fruta es menor que `APP_MIN_PRODUCT_CONFIDENCE` (0,5, en `src/config.py`), dice que no la reconoce. `APP_MODEL` decide quién da el estado: `"random_forest"` (el elegido), `"ensemble"` o `"cnn"`. Si el modelo pedido no está disponible (por ejemplo, sin TensorFlow), la app usa el otro y lo avisa en pantalla. También avisa si la versión de scikit-learn instalada no es la 1.7.2 con la que se guardaron los modelos.

## Arquitectura

| Módulo | Responsabilidad |
| --- | --- |
| `app/app.py` | Solo interfaz: entrada (carga o cámara), resultados, avisos. |
| `src/prediction/predictor.py` | Decodificar, segmentar, preprocesar, predecir y decidir si la foto se rechaza. No depende de Streamlit. |
| `src/data/segmentation.py` | `process()` y `model_inputs()`: los mismos que usa `src/data/prepare.py` al generar los datos de entrenamiento. |
| `src/data/features.py` | `extract_classic_features_from_arrays()`: los mismos 359 atributos que el entrenamiento, desde memoria. |

Una prueba (`tests/test_predictor.py`) verifica que los atributos calculados en memoria por la app son idénticos, valor por valor, a los calculados desde los PNG del entrenamiento.

## Flujo de predicción

1. El usuario carga una foto o la toma con la cámara (`st.camera_input`).
2. `segmentation.process()` estima el fondo, segmenta la pieza y cuenta las frutas.
3. Si la foto está en escala de grises, tiene varias piezas o la máscara es dudosa, la app **no clasifica**: explica el motivo y pide otra foto.
4. `segmentation.model_inputs()` produce el recorte y la máscara de 128×128.
5. Random Forest recibe los 359 atributos; la CNN recibe el recorte con el fondo anulado.
6. Con `APP_MODEL = "random_forest"`, la clase es la de mayor probabilidad del RF. Si se cambia a `"ensemble"`, la clase es el argmax de $0{,}5\,p_\text{RF}+0{,}5\,p_\text{CNN}$, el mismo voto que mide `python -m src.main evaluar`. Las clases de RF se reordenan antes de sumar.
7. La interfaz muestra la clase, las probabilidades, el contorno detectado, lo que recibe el modelo, la categoría de tamaño, $d_n$ y la predicción de cada modelo por separado.
8. Si la probabilidad máxima es menor que `config.APP_MIN_CONFIDENCE` (0,55, provisional), o si RF y la CNN no coinciden, pide revisión manual.

## Decisiones de diseño

- **Random Forest como decisión final:** se eligió en val, antes de abrir test. RF, la CNN y el ensamble empatan en F1 (0,762, 0,738 y 0,769, con intervalos solapados). Pero RF deja pasar como buenas 41 frutas malas, frente a 82 del ensamble y 120 de la CNN, y encuentra más regulares. Con F1 empatado decide el error más caro (fruta mala vendida como buena).
- **Sin TensorFlow obligatorio:** al decidir RF, la app funciona sin la CNN. Con el interruptor "Usar la CNN" se puede ver su opinión como segunda referencia.
- **Aviso de dominio:** la pantalla indica que el desempeño con fotos de celular se mide aparte. En val, RF y la CNN fallaban 3 de las 4 fotos propias; por eso la mitad de las fotos de celular del curso entra al entrenamiento y la otra mitad lo mide.
- **Rechazar antes que adivinar:** una foto con varias piezas o con mala segmentación produciría atributos sin sentido; la app lo dice en lugar de mostrar una clase con falsa seguridad.
- **Lógica separada de la interfaz:** `predictor.py` se prueba con pytest y se puede reutilizar en una simulación de línea de empaque.
- **Sin almacenamiento:** las fotos se procesan en memoria y no se escriben en disco.
- **Comunicación de límites:** la pantalla indica que la clase es apariencia externa, que no evalúa inocuidad y que el tamaño es relativo al encuadre.

## Pruebas de uso pendientes

- [ ] Demostrar carga y cámara con al menos una pieza buena, una regular y una mala.
- [ ] Mostrar dos entradas rechazadas (dos frutas; fondo parecido a la fruta).
- [ ] Medir la latencia por foto en el computador de la demo (para PI2, dimensión ambiental).
- [ ] Elegir `APP_MIN_CONFIDENCE` en validation y documentar el valor.
- [ ] Probar en celular a través de la red local (`streamlit run app/app.py --server.address 0.0.0.0`).

## Monitoreo previsto

Si se recopilan nuevas imágenes, se revisa su distribución por calidad, producto, cámara e iluminación y el desempeño por clase cuando haya etiquetas fiables. Nuevos productos, cambios de captura o caída del desempeño deben activar revisión y posible reentrenamiento. No se extiende el uso a otros contextos sin validación.

## Estado

Interfaz implementada con los modelos entrenados con todos los datos (`models/`); 91 pruebas automáticas pasan. Entregados: el [informe IEEE](../report/informe.pdf) (6 páginas, fuente en `report/`) y el [video del proyecto](https://youtu.be/gwlolsmqH44).
