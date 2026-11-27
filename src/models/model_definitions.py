"""
Los modelos del proyecto y los valores de hiperparámetros que se prueban.

Fase CRISP-DM: Modelado
Uso:
    from src.models.model_definitions import classic_models, build_cnn, CNN_SETTINGS

Este archivo solo DEFINE los modelos; los entrena src/training/train_models.py.

Modelos clásicos (usan los 359 atributos de src/data/features.py)
    Estado (good / regular / bad):
        regresión logística  - una frontera lineal; es la línea base más simple.
        SVM lineal           - también lineal, pero busca el margen más ancho entre clases.
        Random Forest        - muchos árboles de decisión que votan; capta combinaciones de
                               color y textura que una recta no separa.
    Fruta (13 productos):
        Random Forest        - el mismo tipo de modelo, pero la etiqueta es la fruta.

    Para cada uno, GridSearchCV prueba todas las combinaciones de su rejilla con validación
    cruzada por grupos y se queda con la de mejor F1 macro (ver train_models.py).

    Qué significa cada hiperparámetro
        C                 cuánto castiga equivocarse en train: C alto = frontera más ajustada
                          (riesgo de memorizar); C bajo = frontera más simple.
        min_samples_leaf  mínimo de fotos en cada hoja del árbol: más alto = árboles menos
                          detallados, que memorizan menos.
        max_depth         profundidad máxima del árbol (None = sin límite).
        max_features      cuántos atributos mira cada árbol en cada pregunta ("sqrt" = 19 de
                          359; 0.1 = 36): menos atributos = árboles más distintos entre sí.
        class_weight      "balanced" da más peso a las clases con menos fotos (regular).

Red convolucional (CNN, usa el recorte de 128x128 sin fondo)
    4 bloques iguales: Conv 3x3 -> BatchNorm -> ReLU -> MaxPool. Los filtros se duplican en
    cada bloque (32, 64, 128, 256) mientras la imagen se reduce a la mitad (128 -> 8 píxeles):
    los primeros bloques ven bordes y manchas pequeñas; los últimos, zonas grandes de la fruta.
    Luego GlobalAveragePooling (un número por filtro), Dropout 0.35 y una capa softmax con
    3 salidas (probabilidad de good, regular y bad). Unos 390 mil parámetros: es una red
    pequeña, hecha desde cero, sin modelos preentrenados.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src import config

# Ajustes fijos de la CNN (no se buscan con rejilla: cada entrenamiento tarda minutos en GPU).
CNN_SETTINGS = {
    "filters_per_block": (32, 64, 128, 256),
    "dropout": 0.35,
    "learning_rate": 1e-3,     # Adam; se divide entre 2 si la pérdida no mejora en 4 épocas
    "batch_size": 64,
    "max_epochs": 100,
    "patience": 12,            # se detiene si la pérdida no mejora en 12 épocas
    "reduce_lr_patience": 4,
    "min_lr": 1e-5,
    "early_stop_fraction": 0.1,  # 10 % de los GRUPOS de train para decidir cuándo parar
}


def _random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=config.SEED)


def classic_models() -> dict[str, tuple[str, object, dict]]:
    """
    Modelos clásicos y sus rejillas.

    Retorna:
        dict nombre -> (etiqueta que predice: "quality" o "product", modelo, rejilla).
        Los nombres de la rejilla siguen la convención de scikit-learn ("paso__parámetro"
        cuando el modelo va dentro de un Pipeline con StandardScaler).
    """
    return {
        "logistic_regression": ("quality", make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
            {"logisticregression__C": [0.1, 1, 10]}),
        "svm_linear": ("quality", make_pipeline(
            StandardScaler(), LinearSVC(class_weight="balanced", max_iter=5000)),
            {"linearsvc__C": [0.1, 1, 10]}),
        "random_forest": ("quality", _random_forest(),
                          {"min_samples_leaf": [1, 5, 20], "max_depth": [None, 20],
                           "max_features": ["sqrt", 0.1]}),
        "fruit_rf": ("product", _random_forest(),
                     {"min_samples_leaf": [1, 5], "max_features": ["sqrt", 0.1]}),
    }


def build_cnn(tf, settings: dict = CNN_SETTINGS):
    """CNN de 4 bloques descrita arriba, compilada con Adam y entropía cruzada."""
    inputs = tf.keras.Input(shape=(*config.IMG_SIZE, 3))
    x = inputs
    for filters in settings["filters_per_block"]:
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(settings["dropout"])(x)
    outputs = tf.keras.layers.Dense(len(config.QUALITY_CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="cnn_calidad")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
