# Informe IEEE (fuente LaTeX)

Reconstruido a partir del PDF entregado, con dos agregados pedidos por la rúbrica:
- Tabla VI: filas de consentimiento (fotos del curso) y sesgo de los datos.
- Sección III-D: párrafo "Alternativas descartadas".

## Compilar en Overleaf
1. New Project → Upload Project → sube este .zip.
2. Compilador: pdfLaTeX (por defecto). Archivo principal: main.tex.

## Figuras
- Fig. 1 (flujo) está dibujada con TikZ dentro de main.tex.
- Figs. 2, 3 y 4 son PDF en figuras/. Para regenerarlas desde los resultados del proyecto:
  `python informe/generar_figuras.py` (desde la raíz del repositorio).
