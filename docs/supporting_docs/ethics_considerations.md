# Ética e impactos

## Principios y código profesional

El desarrollo debe priorizar el bienestar público, la honestidad sobre capacidades y límites, la privacidad de las personas, la equidad en el análisis y el reconocimiento del trabajo y los datos de terceros. La relación con el [Código de Ética del IEEE](https://ieee-cas.org/about/ieee-code-ethics) se concreta en las decisiones de abajo y en las medidas que se incorporen al sistema y al informe.

## Decisiones éticas y mitigaciones

| Situación | Principio | Medida y evidencia |
| --- | --- | --- |
| Fotos tomadas en hogares, mercados o mediante cámara | Privacidad, consentimiento y bienestar público | Encuadrar solo la pieza; pedir autorización cuando aparezcan personas identificables; revisar metadatos antes de compartir y no conservar capturas de la app por defecto. |
| Fuentes con criterios y cobertura diferentes | Equidad, honestidad y rigor | Conservar procedencia, documentar el mapeo de clases y reportar cobertura/errores por fuente y producto cuando sea posible. |
| Error grave: pieza deteriorada predicha como buena | Bienestar público | Mantener inspección humana, reportar explícitamente estos errores y evitar que una puntuación se convierta en decisión automática. |
| Predicción visual interpretada como inocuidad | Comunicación responsable | Describir la salida como apariencia externa; indicar que el sistema no detecta contaminantes ni evalúa el interior. |
| Uso de datos, código o bibliotecas de terceros | Reconocimiento y cumplimiento | Registrar autores, licencias, versiones y transformaciones en el README y el informe. |

La implementación y los datos disponibles deben respaldar cada medida. Un compromiso escrito no prueba que la app ya procese imágenes sin almacenarlas; ese comportamiento se comprobará cuando la interfaz exista.

## Uso responsable

El resultado es una sugerencia para orientar una revisión, no un certificado de calidad o inocuidad. La persona usuaria conserva la decisión. Si una imagen no permite reconocer una pieza o la máscara es dudosa, la app debe pedir otra foto en vez de afirmar una clase con falsa seguridad.

## Impactos iniciales

| Dimensión | Beneficio potencial | Riesgo y mitigación |
| --- | --- | --- |
| Social | Puede apoyar una revisión más consistente. | Un falso good puede afectar compradores o vendedores. Mantener revisión humana y analizar falsos good para piezas bad. |
| Económica | Podría reducir tiempo o pérdidas evitables. | Errores, recapturas y mantenimiento también cuestan. Medir tiempo y tasa de reintento en una prueba de uso antes de afirmar ahorro. |
| Ambiental | Una clasificación más informada podría evitar algunos descartes. | Entrenar y ejecutar modelos consume energía. Registrar equipo y tiempos; comprobar cualquier reducción de desperdicio en campo. |
| Global | Una interfaz con celular puede facilitar la réplica. | Variedades, cámaras, prácticas y normas cambian por lugar. Validar cada población y contexto antes de extender conclusiones. |

Estos son impactos posibles, no efectos medidos del prototipo. PI2 se completa al contrastarlos con resultados y pruebas de uso.

## Criterios de cierre ético

- La procedencia y licencia de cada fuente están documentadas.
- Las fotos propias se revisan para evitar publicar información identificable.
- La evaluación informa diferencias de cobertura y errores relevantes.
- La interfaz comunica que predice apariencia externa y mantiene control humano.
- El informe separa resultados observados de beneficios hipotéticos.

## Referencias

- IEEE, [Code of Ethics](https://ieee-cas.org/about/ieee-code-ethics).
- Universidad Icesi, *Competencias que se deben evaluar en el curso*, indicador PI1 y análisis PI2.
- [Fuentes y licencias en README](../../README.md#créditos).
