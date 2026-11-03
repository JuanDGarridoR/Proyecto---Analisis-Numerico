# AirSpline — Caso 3: Predicción de calidad del aire con splines cúbicos

| Archivo | Qué es |
|---|---|
| `AirSpline_Caso3.ipynb` | Cuaderno completo (entregable técnico): teoría, spline propia vs SciPy, comparación Spline/Hermite/PCHIP/Lagrange, LOOCV, malla de 91 puntos, derivada, integral, alertas, Monte Carlo, diagrama de bloques, KPIs, **panel interactivo** y guion del pitch. |
| `airspline_app.py` | App de escritorio (ventana con sliders) para la demo en vivo sin Jupyter. |
| `requirements.txt` | Dependencias. |

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

## Google Colab
*Archivo → Subir cuaderno* → `AirSpline_Caso3.ipynb` → *Entorno de ejecución → Ejecutar todas*. La primera celda instala lo que falte y activa los widgets.

## Archivos generados
- `airspline_malla_91_puntos.csv` — concentración, derivada y alerta en los 91 puntos.
- `airspline_panel_export.csv` — exportación desde el panel (botón “Exportar malla CSV”; en Colab se descarga).
