# 6. Despliegue

## Objetivo

La entrega prevé una interfaz local de Streamlit para probar el modelo seleccionado con una fotografía cargada o tomada con cámara. El despliegue descrito aquí es un plan académico; no significa que el sistema esté instalado en una operación comercial.

## Flujo de predicción

1. El usuario carga o captura una imagen de una sola pieza sobre fondo sencillo.
2. La aplicación verifica que el archivo se pueda leer y reutiliza el preprocesamiento del entrenamiento.
3. El sistema segmenta la pieza y revisa controles básicos de plausibilidad.
4. La interfaz muestra la clase de apariencia externa, el diámetro normalizado y, solo si se valida, una categoría de tamaño.
5. Si la imagen tiene varias piezas, no se lee, está demasiado borrosa o la máscara es dudosa, solicita una nueva captura.

La interfaz debe dejar claro que la predicción apoya la inspección visual y no evalúa inocuidad ni decide consumo, descarte o comercialización. Las imágenes no deben conservarse por defecto.

## Pruebas de uso y monitoreo previsto

La demostración debe cubrir carga, cámara, entradas válidas, entradas rechazadas y consistencia del preprocesamiento. Se registrarán versión del modelo, dependencias y transformaciones para reproducir la inferencia. Durante la prueba se pueden medir latencia, errores y solicitudes de recaptura; esto no equivale a medir beneficios comerciales.

Si se recopilan nuevas imágenes, debe revisarse la distribución por calidad, fuente, producto, cámara e iluminación, junto con el desempeño por clase si se obtienen etiquetas fiables. La aparición de nuevos productos, cambios de captura o caída del desempeño debe activar revisión y posible reentrenamiento. No se extenderá el uso a otros contextos sin validación.

## Estado y conclusión

src/app contiene solo un inicializador: Streamlit, carga/cámara e inferencia todavía no están implementados. El despliegue se completará cuando exista la aplicación, use el mismo preprocesamiento del modelo, comunique límites y haya sido demostrada con casos correctos e incorrectos.

El cierre de la entrega incluirá el informe IEEE de máximo siete páginas, un video de hasta diez minutos, referencias realmente utilizadas y una revisión de las lecciones del ciclo CRISP-DM.
