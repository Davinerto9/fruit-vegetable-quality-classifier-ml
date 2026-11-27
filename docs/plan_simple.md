# Plan simple: fruta + estado, sin trampa

**Objetivo.** Con una foto, la app dice **qué fruta es** y **en qué estado está** (buena, regular o mala), con cifras que se puedan defender: medidas sin trampa y con modelos sencillos.

## Qué hace el sistema

1. **Separa la fruta del fondo** con OpenCV (`src/data/segmentation.py`). Si hay varias frutas o el fondo se confunde con la fruta, pide otra foto en lugar de adivinar.
2. **Describe la fruta** con 359 números de color, textura y forma (`src/data/features.py`).
3. **Fruta:** un Random Forest con esos números elige entre los 13 productos que conoce. Si su probabilidad es menor al 50%, la app dice que no la reconoce.
4. **Estado:** un Random Forest y una red convolucional (4 bloques) dan cada uno una probabilidad por estado; la app promedia las dos, mitad y mitad.

| Pieza | Modelo | Por qué |
| --- | --- | --- |
| Fruta | Random Forest (200 árboles) | Simple, rápido y explicable; usa los mismos atributos. |
| Estado (clásicos) | Regresión logística, SVM lineal, Random Forest | La consigna pide al menos dos modelos clásicos; la logística es la línea base. |
| Estado (profundo) | CNN de 4 bloques Conv-BatchNorm-ReLU-MaxPool, hecha desde cero | La consigna pide un modelo de deep learning. |
| Estado final | Promedio RF + CNN | Una línea de código y menos errores de vender fruta mala como buena. |

## Sin trampa: tres reglas

1. **Fotos parecidas, del mismo lado.** AgriFreshNET fotografió las mismas frutas durante semanas. Por eso todas las fotos de una misma fruta y semana quedan juntas en *train*, en *validación* o en *prueba* (`src/data/split.py`). Lo mismo con las ráfagas de Kaggle.
2. **Hiperparámetros con validación cruzada.** Se eligen con `GridSearchCV` y `StratifiedGroupKFold(3)` de scikit-learn, solo con *train* y respetando los grupos de la regla 1. La CNN decide cuándo parar con un 10% de *train*, no con validación.
3. **Validación y prueba, una vez.** Ninguna se usa para elegir nada. La prueba tiene un candado: si se intenta abrir otra vez, el programa se niega.

## Métricas (y cómo explicarlas)

- **Accuracy (fruta):** de cada 100 fotos, en cuántas acertó la fruta.
- **F1 macro (estado):** el promedio del F1 de buena, regular y mala, cada una con el mismo peso, así que no se puede inflar acertando solo la clase más común.
- **Matriz de confusión:** qué se confunde con qué.
- **Errores mala→buena:** cuántas frutas malas pasarían como buenas. Es el error más caro para un comprador.
- **Intervalo de confianza:** el rango donde probablemente está la cifra real, calculado remuestreando grupos de fotos.

## Cómo se reproduce

Cada paso es un comando de `src/main.py` y lee lo que dejó el anterior:

```
fotos (data/raw/)
  │  1. inventario   ¿qué fruta, qué estado, qué grupo?         -> index.csv
  │  2. recortar     fruta sin fondo, 128x128 (o se rechaza)     -> crops/
  │  3. dividir      70/15/15 por grupos                         -> splits.csv
  │  4. atributos    359 números por recorte                     -> features.npz
  │  5. entrenar     GridSearchCV por grupos + CNN               -> models/
  │  6. evaluar      métricas en val                             -> results/
  ▼
app/app.py usa models/ para una foto nueva
```

Antes, las fotos de celular del curso se parten por sesión: la mitad se etiqueta y entra a train (`celular ingerir`, `hojas`, `finalizar`), la otra mitad se guarda para el final. `python -m src.main todo` corre los seis pasos seguidos. Al final, una sola vez: `congelar`, `evaluar --test --confirmo` y `celular evaluar --confirmo`. Los comandos exactos están en el [README](../README.md#cómo-usar).

## Límites que hay que decir en la sustentación

- **"Cualquier foto" tiene condiciones:** una sola fruta, fondo liso y distinto a la fruta, y que la fruta esté completa. La app rechaza las fotos que no cumplen, en lugar de inventar.
- **Solo conoce 13 productos.** Una fruta distinta (kiwi, fresa…) debe recibir "no la reconozco".
- **El mango solo se midió dentro de AFruitDB** (mangos de Bangladesh). Con mangos colombianos la fruta o el estado pueden fallar; para medirlo hacen falta fotos propias nuevas de mango en la mitad de prueba.
- **El tamaño es relativo al encuadre,** no centímetros.
- **El estado es la apariencia externa,** no la inocuidad ni el interior de la fruta.

## Qué quedó fuera, a propósito

Para mantenerlo simple no se usa transfer learning, búsqueda de variantes de la CNN, calibración del ensamble ni adaptación de dominio. Se pueden mencionar como trabajo futuro.
