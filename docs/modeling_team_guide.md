# Guía del equipo: reproducir validation y auditar errores

## Situación actual

Los cuatro modelos fueron entrenados sobre el mismo split grupal 70/15/15: regresión logística, SVM lineal, Random Forest y CNN RGB. Los checkpoints locales se guardan en `experiments/checkpoints/`; están ignorados por Git. Las métricas registradas en validation son iniciales y test sigue intacto.

| Modelo | F1 macro | Balanced accuracy | Entrada |
| --- | ---: | ---: | --- |
| Regresión logística | 0.6845 | 0.7010 | 359 atributos clásicos |
| SVM lineal | 0.6978 | 0.7022 | 359 atributos clásicos |
| Random Forest | 0.8424 | 0.8363 | 359 atributos clásicos |
| CNN RGB | 0.7988 | 0.8139 | RGB 128 × 128 |

## Preparar el entorno reproducible

Ejecutar desde la raíz en WSL/Linux, con el entorno Python 3.11 del proyecto:

```bash
source ~/.venvs/fruit-quality/bin/activate
python -m pip install "scikit-learn==1.7.2"
python -c "import sklearn; print(sklearn.__version__)"
```

Debe imprimir `1.7.2`. Si se creó el ambiente desde cero, instalar dependencias según `requirements.txt`. No actualizar scikit-learn a una versión más reciente en el entorno que carga los `.joblib` clásicos.

## Reproducir el análisis de validation

Primero comprobar que los artefactos locales requeridos existen: `data/processed/crops/splits.csv`, `data/processed/classic_features.npz`, cuatro checkpoints y los archivos de datos necesarios para la CNN. No regenerar el split como parte de esta evaluación.

```bash
python -m src.evaluation.model_audit --partition val --batch-size 64
```

El proceso solo lee validation y debe indicar `test_used=False`. Escribe/actualiza:

- `experiments/results/validation_audit.json`
- `experiments/results/validation_predictions.csv`
- `experiments/results/validation_error_cases.csv`

La primera salida debe reportar 3.557 filas y 1.979 grupos si el split y los datos son los mismos. Confirmar que no aparecen warnings de versión al cargar los checkpoints clásicos. La advertencia de TensorFlow sobre GPU puede aparecer si WSL no ve CUDA; la evaluación continúa en CPU y no implica que el checkpoint esté dañado.

No entrenar la CNN para volver a obtener métricas. Para puntuar su checkpoint por separado, el modo de evaluación es:

```bash
python -m src.training.train_cnn --evaluate-checkpoint
```

Ese modo carga `experiments/checkpoints/cnn_rgb.keras` y solo evalúa validation. No ejecutar el entrenamiento normal de CNN salvo que una modificación aprobada a los datos o al pipeline justifique rehacer el experimento; guardar entonces un checkpoint nuevo y documentar por qué.

## Auditar los 871 casos

El audit inicial encontró 871 filas donde Random Forest o CNN se equivoca: RF solo 210, CNN solo 378, ambos 283. No significa que 871 etiquetas estén mal; es la unión de errores de ese par de modelos. El archivo `validation_error_cases.csv` tiene 1.426 filas, ya que agrega cualquier error de los cuatro modelos; `validation_error_cases_reviewed.csv` contiene la misma cola y no tiene anotaciones de revisión. Si el equipo completa auditoría de todos los clasificadores, revisar las 1.426 filas. Antes de usar cualquiera de los CSV, regenerarlo después de alinear scikit-learn 1.7.2. Revisar cada fila en este orden:

1. Abrir la imagen original y el recorte/máscara usados; verificar que sea una sola pieza y que el fondo no altere la apariencia.
2. Confirmar etiqueta contra los criterios del proyecto, registrar clase predicha de cada modelo, fuente, producto y grupo.
3. Registrar decisión en una copia de trabajo con columnas nuevas: `review_status`, `review_decision`, `review_reason`, `reviewer`, `review_date`. Usar estados controlados: `confirmed_label`, `label_change_proposed`, `mask_issue`, `ambiguous`, `model_error`, `excluded_candidate`.
4. No editar silenciosamente el CSV fuente ni cambiar solo la etiqueta de una fila cuando forma parte de un grupo. Una propuesta de cambio debe rastrearse hasta el manifiesto/anotación canónica y revisarse por el equipo.
5. Resumir conteos por etiqueta real, predicción, fuente, producto y tipo de hallazgo. Distinguir etiqueta errónea de fallo de modelo.

Los casos no deben describirse como “resueltos” hasta que cada fila de la cola que el equipo decidió auditar tenga una decisión, motivo y revisor, y se hayan aplicado formalmente las correcciones autorizadas. El archivo `validation_error_cases_reviewed.csv` existente es una copia sin columnas de decisión; todavía no documenta revisión.

## Riesgos de datos que condicionan el cierre

- **Sesgo de fuente:** las 705 filas `regular` de validation (251 grupos) provienen 100% de AgriFreshNET. No existe soporte de esa clase de otras fuentes en validation; rendimiento por fuente no demuestra generalización de `regular`.
- **Mango:** solo dos candidatos aceptados de una misma sesión se movieron a train. No hay mango en validation/test y quedan cuatro máscaras por resolver; no afirmar rendimiento de mango. Resolver esas máscaras o excluirlo explícitamente del alcance.
- **Muestra visual:** la auditoría previa de recortes fue estratificada (44 candidatos) frente a decenas de miles de candidatos. La aprobación automática no garantiza máscara correcta.
- **Versión serializada:** los modelos clásicos se guardaron con scikit-learn 1.7.2. La ejecución anterior en 1.9.1 emitió `InconsistentVersionWarning`; tratar desgloses por imagen como provisionales hasta repetir.

No balancear moviendo filas entre particiones, no corregir `test`, no entrenar con validation/test y no abrir test para investigar errores.

## Cuándo congelar y abrir test

Se puede proponer congelamiento cuando: (1) audit de validation regenerado sin mismatch de sklearn; (2) los 871 casos revisados y resumidos; (3) cambios de etiqueta/máscara aprobados y aplicados a las fuentes canónicas; (4) mango y la falta de cobertura multifuente de `regular` tienen una decisión explícita de alcance; (5) equipo acepta modelos, features, umbrales y pesos; (6) split y checkpoints tienen hashes registrados.

Entonces generar y revisar `experiments/results/pretest_freeze.sha256`, que protege el comando de test. La evaluación final se corre una sola vez, únicamente mediante el modo protegido del auditor:

```bash
python -m src.evaluation.model_audit --partition test --confirm-final-test --batch-size 64
```

No usar test para escoger entre modelos ni corregir parámetros. Si el resultado revela un problema, registrarlo como limitación y planear una nueva versión del experimento con un protocolo de evaluación nuevo.

## Artefactos para compartir

Compartir el código, esta guía, los documentos CRISP-DM, los JSON y reportes derivados aprobados por el equipo. Mantener fuera de Git los datasets y checkpoints según `.gitignore`. Antes de subir CSV/JSON, revisar que no contengan rutas personales, nombres sensibles ni datos que el grupo no quiera publicar. En el resumen de equipo citar siempre el soporte y las limitaciones, no solo el F1 agregado.
