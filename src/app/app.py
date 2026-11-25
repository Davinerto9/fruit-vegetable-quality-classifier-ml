"""
Aplicación Streamlit: clasifica la calidad visible y el tamaño relativo de una fruta o verdura
a partir de una foto cargada o tomada con la cámara.

Fase CRISP-DM: Despliegue
Uso (desde la raíz del repositorio, con el ambiente activado):
    streamlit run src/app/app.py

Toda la lógica de segmentación y predicción vive en src/app/inference.py; este archivo solo
arma la interfaz. Las fotos no se guardan: se procesan en memoria y se descartan.
"""
import sys
from pathlib import Path

# `streamlit run src/app/app.py` pone src/app en sys.path, no la raíz: se agrega para
# poder importar el paquete src igual que en el resto del proyecto.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.app.inference import (  # noqa: E402
    QUALITY_LABELS_ES,
    SIZE_LABELS_ES,
    analyze_image,
    decode_image,
    ensemble_validation_f1,
    load_models,
)

MODE_NAMES = {"ensemble": "Ensamble RF + CNN", "random_forest": "Random Forest", "cnn": "CNN"}

QUALITY_COLORS = {"good": "#2e7d32", "regular": "#b26a00", "bad": "#c62828"}
QUALITY_HELP = {
    "good": "Pieza completa, con su color habitual y sin zonas oscuras de podredumbre.",
    "regular": "Deterioro inicial: marcas, golpes, manchas localizadas o una abertura leve.",
    "bad": "Hongos visibles, pérdida marcada del color o señales claras de podredumbre.",
}


@st.cache_resource(show_spinner="Cargando modelos…")
def get_models(include_cnn: bool):
    return load_models(include_cnn=include_cnn)


def render_sidebar() -> bool:
    st.sidebar.header("Modelos")
    include_cnn = st.sidebar.toggle(
        "Usar la CNN", value=config.APP_MODEL != "random_forest",
        help=f"Necesaria para el ensamble. Requiere TensorFlow y {config.APP_CNN_CHECKPOINT}.",
    )
    models, notices = get_models(include_cnn)
    st.sidebar.markdown(f"Decide: **{MODE_NAMES.get(config.APP_MODEL, config.APP_MODEL)}**")
    if config.APP_MODEL == "ensemble":
        f1 = ensemble_validation_f1()
        st.sidebar.caption(
            f"Voto suave con peso fijo {1 - config.APP_ENSEMBLE_CNN_WEIGHT:.1f} RF / "
            f"{config.APP_ENSEMBLE_CNN_WEIGHT:.1f} CNN"
            + ("" if f1 is None else f". F1 macro en validation: {f1:.3f}.")
        )
    for model in models:
        f1 = "s/d" if model.validation_f1 is None else f"{model.validation_f1:.3f}"
        st.sidebar.markdown(
            f"**{model.name}**  \nF1 macro en validation: {f1}  \n"
            f"Checkpoint `{model.sha256[:12]}`"
        )
        for warning in model.warnings:
            st.sidebar.warning(warning)
    for notice in notices:
        st.sidebar.info(notice)

    st.sidebar.header("Cómo tomar la foto")
    st.sidebar.markdown(
        "- Una sola fruta o verdura.\n"
        "- Fondo liso y de un color distinto al de la pieza.\n"
        "- La pieza completa, sin tocar los bordes.\n"
        "- Buena luz, sin sombras fuertes.\n"
        "- Distancia parecida entre fotos: el tamaño es relativo al encuadre."
    )
    st.sidebar.caption(
        "Productos con datos de entrenamiento: " + ", ".join(config.PRODUCTS) + "."
    )
    return include_cnn


def render_probabilities(probabilities: dict[str, float]) -> None:
    for label in config.QUALITY_CLASSES:
        value = probabilities[label]
        st.markdown(f"{QUALITY_LABELS_ES[label]}: **{value:.0%}**")
        st.progress(min(max(value, 0.0), 1.0))


def render_result(analysis) -> None:
    left, right = st.columns([3, 2], gap="large")
    with left:
        if analysis.overlay_rgb is not None:
            st.image(analysis.overlay_rgb, caption="Contorno detectado")
        if analysis.crop_masked_rgb is not None:
            st.image(analysis.crop_masked_rgb, width=192,
                     caption="Lo que recibe el modelo (128×128, sin fondo)")

    with right:
        if not analysis.accepted:
            st.error(analysis.message)
            st.caption(f"Estado de la segmentación: `{analysis.status}`")
            return
        if not analysis.predictions:
            st.warning("La foto es válida, pero no hay modelos cargados para clasificarla.")
            return

        main = analysis.final
        if analysis.final_note:
            st.warning(analysis.final_note)
        color = QUALITY_COLORS[main.label]
        st.markdown(
            f"<div style='font-size:0.9rem;opacity:0.7'>Calidad visible ({main.model_name})</div>"
            f"<div style='font-size:2.4rem;font-weight:700;color:{color}'>"
            f"{QUALITY_LABELS_ES[main.label]}</div>",
            unsafe_allow_html=True,
        )
        st.caption(QUALITY_HELP[main.label])
        if not main.confident:
            st.warning(
                f"Confianza baja ({main.probabilities[main.label]:.0%}). "
                "Revisa la pieza manualmente o toma otra foto."
            )
        render_probabilities(main.probabilities)

        st.divider()
        size_text = SIZE_LABELS_ES.get(analysis.size, "s/d")
        st.metric("Tamaño relativo", size_text, help=(
            "Categoría a partir del diámetro normalizado. Depende de la distancia de la "
            "cámara: no es una medida en centímetros."
        ))
        st.caption(f"Diámetro normalizado d_n = {analysis.d_norm:.3f}")

        if len(analysis.predictions) > 1:
            st.divider()
            st.markdown("**Cada modelo por separado**")
            for prediction in analysis.predictions:
                st.markdown(
                    f"- {prediction.model_name}: {QUALITY_LABELS_ES[prediction.label]} "
                    f"({prediction.probabilities[prediction.label]:.0%})"
                )
            if len({p.label for p in analysis.predictions}) > 1:
                st.warning("Los modelos no coinciden. Conviene revisar la pieza manualmente.")


def main() -> None:
    st.set_page_config(page_title="Calidad de frutas y verduras", layout="wide")
    st.title("Clasificador de calidad de frutas y verduras")
    st.markdown(
        "Sube o toma una foto de **una** pieza sobre un fondo sencillo. El sistema estima su "
        "calidad visible (buena, regular o mala) y su tamaño relativo."
    )
    st.info(
        "La predicción describe la apariencia externa y apoya la inspección humana. No evalúa "
        "inocuidad ni estado interno, y no reemplaza la decisión de quien inspecciona. "
        "Las fotos no se guardan."
    )
    st.caption(
        "Los modelos se entrenaron con fotos de dos datasets públicos. Con fotos de celular, "
        "en otras condiciones de luz y fondo, el desempeño todavía no está medido y puede ser "
        "menor que el reportado."
    )

    include_cnn = render_sidebar()
    models, _ = get_models(include_cnn)

    upload_tab, camera_tab = st.tabs(["Subir foto", "Usar cámara"])
    with upload_tab:
        uploaded = st.file_uploader("Foto de la fruta o verdura", type=["jpg", "jpeg", "png"])
    with camera_tab:
        captured = st.camera_input("Toma una foto")

    source = captured or uploaded
    if source is None:
        return
    with st.spinner("Segmentando y clasificando…"):
        analysis = analyze_image(decode_image(source.getvalue()), models)
    render_result(analysis)


if __name__ == "__main__":
    main()
