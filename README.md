# AirSpline — Caso 3: Predicción de calidad del aire con splines cúbicos

| Archivo | Qué es |
|---|---|
| `AirSpline_Caso3.ipynb` | Cuaderno completo (entregable técnico): teoría, spline propia vs SciPy, comparación Spline/Hermite/PCHIP/Lagrange, LOOCV, malla de 91 puntos, derivada, integral, alertas, Monte Carlo, diagrama de bloques, KPIs, **panel interactivo** y guion del pitch. |
| `airspline_app.py` | App de escritorio (ventana con sliders) para la demo en vivo sin Jupyter. |
| `requirements.txt` | Dependencias. |
| `modelado_validacion.py` | Módulo de **Modelado y Validación**: spline propio (natural, sujeta, not-a-knot) verificado contra SciPy, LOOCV con RMSE y error máximo, fenómeno de Runge (Lebesgue), orden de convergencia, robustez ante ruido; genera figuras y tablas del informe. |
| `tests/` | Pruebas automáticas (`pytest`) de verificación y validación. |
| `docs/MODELADO_Y_VALIDACION.md` | Documentación del rol de Modelado y Validación (resultados, tablas, respuestas para el jurado). |
| `docs/informe/` | Informe IEEE del rol (`.tex` + `.pdf`), figuras y tablas generadas. |

## VS Code
1. Instalar extensiones **Python** y **Jupyter**.
2. En una terminal:
   ```bash
   pip install -r requirements.txt
   ```
3. Abrir `AirSpline_Caso3.ipynb`, elegir el kernel de Python y pulsar **Run All**. El panel está en la sección 11.
4. App de escritorio:
   ```bash
   python airspline_app.py
   ```

## Modelado y Validación (reproducir resultados del informe)
```bash
python modelado_validacion.py
```
```bash
python -m pytest -q tests
```
El primero imprime el reporte y regenera `docs/informe/figuras/` y `docs/informe/generado/`; el informe se compila con `pdflatex informe_modelado_validacion.tex` dentro de `docs/informe/` (o en Overleaf). Ver [`docs/MODELADO_Y_VALIDACION.md`](docs/MODELADO_Y_VALIDACION.md).

## Google Colab
*Archivo → Subir cuaderno* → `AirSpline_Caso3.ipynb` → *Entorno de ejecución → Ejecutar todas*. La primera celda instala lo que falte y activa los widgets.

## Archivos generados
- `airspline_malla_91_puntos.csv` — concentración, derivada y alerta en los 91 puntos.
- `airspline_panel_export.csv` — exportación desde el panel (botón “Exportar malla CSV”; en Colab se descarga).
