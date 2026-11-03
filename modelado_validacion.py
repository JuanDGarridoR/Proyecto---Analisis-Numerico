# -*- coding: utf-8 -*-
"""
AirSpline · Módulo de Modelado y Validación (Estudio de Caso 3)
================================================================

Rol      : Ingeniero de Modelado y Validación — Miguel Angel Laiton
Equipo   : Juan David Garrido (Algoritmos y Simulación) · Edwin Quiñones (Integración y
           Estrategia) · Miguel Angel Laiton (Modelado y Validación)
Proyecto : AirSpline — sensores virtuales de PM₂.₅ con splines cúbicos (Air Health Analytics)

Este módulo es el "motor de verificación y validación" (V&V) del proyecto. Complementa las
secciones §2, §3 y §5 del cuaderno ``AirSpline_Caso3.ipynb`` y genera, desde una sola fuente,
todos los números, tablas y figuras del informe del rol (``docs/informe``).

    1. Formulación    — sistema tridiagonal del spline cúbico con frontera natural, sujeta y
                        not-a-knot; implementación propia (algoritmo de Thomas).
    2. Verificación   — coeficientes propios vs. SciPy, continuidad C² en los nodos y orden de
                        convergencia observado (≈4 not-a-knot/sujeta, ≈2 natural y lineal).
    3. Runge          — polinomio nodal ω(x), constante de Lebesgue, localidad del spline y
                        oscilaciones espurias (variación total, extremos).
    4. Validación     — Leave-One-Out en los 8 nodos internos: error absoluto, RMSE, MAE,
                        error máximo y estación peor.
    5. Robustez       — amplificación analítica del ruido en la LOOCV, LOOCV bajo ruido
                        σ = 2.5 µg/m³ (Monte Carlo) y sensibilidad de los tramos de alerta a la
                        frontera elegida.

Uso
---
    python modelado_validacion.py                  # reporte en consola + figuras + tablas
    python modelado_validacion.py --sin-figuras    # sólo números y tablas
    python -m pytest -q tests                      # pruebas automáticas de verificación

Los nombres de los modelos y las funciones ``construir_modelos``, ``loocv`` y
``tramos_sobre_umbral`` coinciden con los del cuaderno para que los resultados sean
directamente comparables.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import (BarycentricInterpolator, CubicHermiteSpline, CubicSpline,
                               PchipInterpolator)

# =============================================================================================
# 0. Datos del caso (idénticos a la §1 del cuaderno)
# =============================================================================================
X_KM = np.arange(0.0, 10.0, 1.0)                                         # km 0 … 9, h = 1 km
PM25 = np.array([45, 52, 68, 85, 92, 88, 79, 63, 48, 42], dtype=float)   # µg/m³
ESTACIONES = [f"E{i + 1}" for i in range(len(X_KM))]
L_KM = float(X_KM[-1] - X_KM[0])                                         # 9 km
DX = 0.1
XF = np.round(np.linspace(X_KM[0], X_KM[-1], int(round(L_KM / DX)) + 1), 1)   # 91 puntos
UMBRAL_AMARILLA, UMBRAL_ROJA = 70.0, 90.0
SIGMA, SEMILLA = 2.5, 2024
N_MC_VALIDACION = 2000            # realizaciones para la LOOCV bajo ruido (más que las 100 de la §8)

# Nombres EXACTOS del cuaderno (§4) y orden de dibujo validado para daltonismo.
NAK, NAT, SUJ = "Spline not-a-knot", "Spline natural", "Spline sujeta"
HER, PCH, LIN, LAG = "Hermite (np.gradient)", "PCHIP", "Lineal", "Lagrange (grado n-1)"
ORDEN_MODELOS = [NAK, HER, NAT, SUJ, LAG, PCH, LIN]
MODELOS_LINEALES = [NAK, NAT, SUJ, HER, LIN, LAG]      # PCHIP es no lineal en los datos

# Paleta del cuaderno (color = identidad del método) + codificación secundaria (línea/marcador)
COLOR = {NAK: "#2563eb", NAT: "#7c3aed", SUJ: "#0891b2", HER: "#ea580c", PCH: "#059669",
         LIN: "#6b7280", LAG: "#dc2626"}
LINEA = {NAK: "-", NAT: "--", SUJ: ":", HER: "-.", PCH: (0, (5, 1, 1, 1)), LIN: "-", LAG: ":"}
MARCA = {NAK: "o", NAT: "s", SUJ: "v", HER: "^", PCH: "P", LIN: "x", LAG: "D"}
TINTA, TINTA_2, REJILLA = "#1f2937", "#475569", "#e5e7eb"


# =============================================================================================
# 1. Formulación: spline cúbico propio (algoritmo de Thomas) para tres fronteras
# =============================================================================================
@dataclass
class SplineCubico:
    """Spline cúbico por tramos  S_i(t) = a_i + b_i (t-x_i) + c_i (t-x_i)² + d_i (t-x_i)³."""
    x: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray        # c_i = S''(x_i)/2, longitud n+1 (incluye el nodo final)
    d: np.ndarray
    frontera: str

    def _tramo(self, t):
        t = np.asarray(t, dtype=float)
        i = np.clip(np.searchsorted(self.x, t, side="right") - 1, 0, len(self.x) - 2)
        return t, i, t - self.x[i]

    def __call__(self, t, nu: int = 0):
        t, i, s = self._tramo(t)
        a, b, c, d = self.a[i], self.b[i], self.c[i], self.d[i]
        if nu == 0:
            return a + s * (b + s * (c + s * d))
        if nu == 1:
            return b + s * (2 * c + 3 * d * s)
        if nu == 2:
            return 2 * c + 6 * d * s
        if nu == 3:
            return 6 * d + 0 * s
        raise ValueError("nu debe ser 0, 1, 2 o 3")

    def integral(self) -> float:
        """∫_{x0}^{xn} S(t) dt exacta (suma de las integrales de cada cúbica)."""
        h = np.diff(self.x)
        return float(np.sum(self.a * h + self.b * h**2 / 2 + self.c[:-1] * h**3 / 3 + self.d * h**4 / 4))

    def coeficientes(self) -> pd.DataFrame:
        return pd.DataFrame({"Tramo [km]": [f"[{p:g}, {q:g}]" for p, q in zip(self.x[:-1], self.x[1:])],
                             "a_i": self.a, "b_i": self.b, "c_i": self.c[:-1], "d_i": self.d})


def thomas(sub, diag, sup, rhs):
    """Resuelve un sistema tridiagonal (algoritmo de Thomas, O(n)). ``sub[0]`` y ``sup[-1]`` no se usan."""
    n = len(diag)
    cp, dp = np.zeros(n), np.zeros(n)
    cp[0], dp[0] = sup[0] / diag[0] if n > 1 else 0.0, rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - sub[i] * cp[i - 1]
        cp[i] = sup[i] / m if i < n - 1 else 0.0
        dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / m
    sol = np.zeros(n)
    sol[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        sol[i] = dp[i] - cp[i] * sol[i + 1]
    return sol


def sistema_tridiagonal(x, y, frontera="not-a-knot", pendientes=None):
    """Arma el sistema tridiagonal en las incógnitas c_i = S''(x_i)/2.

    Ecuación de cada nodo interno i = 1, …, n-1 (continuidad de S' y S''):
        h_{i-1} c_{i-1} + 2(h_{i-1}+h_i) c_i + h_i c_{i+1} = 3(a_{i+1}-a_i)/h_i - 3(a_i-a_{i-1})/h_{i-1}

    * natural     : c_0 = c_n = 0                         → incógnitas c_1 … c_{n-1}
    * sujeta      : S'(x_0) = f'_0, S'(x_n) = f'_n        → incógnitas c_0 … c_n
    * not-a-knot  : S''' continua en x_1 y x_{n-1}  (d_0 = d_1, d_{n-2} = d_{n-1});
                    se eliminan c_0 y c_n                 → incógnitas c_1 … c_{n-1}
    * periódica   : sólo si y_0 = y_n (no aplica a un corredor abierto)

    Devuelve (sub, diag, sup, rhs, indices) donde ``indices`` son los c_i que se resuelven.
    """
    x, a = np.asarray(x, float), np.asarray(y, float)
    n = len(x) - 1
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("Los nodos deben ser estrictamente crecientes")
    r = np.zeros(n + 1)
    r[1:n] = 3 * (a[2:] - a[1:-1]) / h[1:] - 3 * (a[1:-1] - a[:-2]) / h[:-1]

    if frontera == "periodica":
        raise ValueError(f"Frontera periódica no aplica: el corredor es abierto y y_0 = {a[0]:g} ≠ "
                         f"y_n = {a[-1]:g}; además S'(0) = S'(9) no tiene sentido físico.")

    if frontera == "natural":
        idx = np.arange(1, n)
        sub, diag, sup = h[:-1].copy(), 2 * (h[:-1] + h[1:]), h[1:].copy()
        return sub, diag, sup, r[1:n].copy(), idx

    if frontera == "sujeta":
        if pendientes is None:
            raise ValueError("La frontera sujeta requiere pendientes (f'_0, f'_n)")
        fp0, fpn = pendientes
        idx = np.arange(0, n + 1)
        sub, diag, sup, rhs = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1), r.copy()
        diag[0], sup[0], rhs[0] = 2 * h[0], h[0], 3 * (a[1] - a[0]) / h[0] - 3 * fp0
        sub[1:n], diag[1:n], sup[1:n] = h[:-1], 2 * (h[:-1] + h[1:]), h[1:]
        sub[n], diag[n], rhs[n] = h[-1], 2 * h[-1], 3 * fpn - 3 * (a[-1] - a[-2]) / h[-1]
        return sub, diag, sup, rhs, idx

    if frontera == "not-a-knot":
        if n < 3:
            raise ValueError("not-a-knot requiere al menos 4 nodos")
        idx = np.arange(1, n)
        sub, diag, sup = h[:-1].copy(), 2 * (h[:-1] + h[1:]), h[1:].copy()
        # c_0 = (1 + h0/h1) c_1 − (h0/h1) c_2   sustituido en la primera ecuación
        diag[0] += h[0] * (1 + h[0] / h[1])
        sup[0] -= h[0] ** 2 / h[1]
        # c_n = (1 + h_{n-1}/h_{n-2}) c_{n-1} − (h_{n-1}/h_{n-2}) c_{n-2}   en la última
        diag[-1] += h[-1] * (1 + h[-1] / h[-2])
        sub[-1] -= h[-1] ** 2 / h[-2]
        return sub, diag, sup, r[1:n].copy(), idx

    raise ValueError(f"Frontera desconocida: {frontera!r}")


def matriz_densa(sub, diag, sup) -> np.ndarray:
    """Matriz tridiagonal densa (sólo para mostrarla en el informe y calcular su condición)."""
    return np.diag(diag) + np.diag(sub[1:], -1) + np.diag(sup[:-1], 1)


def spline_propio(x, y, frontera="not-a-knot", pendientes=None) -> SplineCubico:
    """Spline cúbico interpolante construido con nuestra propia formulación + Thomas."""
    x, a = np.asarray(x, float), np.asarray(y, float)
    n = len(x) - 1
    h = np.diff(x)
    sub, diag, sup, rhs, idx = sistema_tridiagonal(x, a, frontera, pendientes)
    c = np.zeros(n + 1)
    c[idx] = thomas(sub, diag, sup, rhs)
    if frontera == "not-a-knot":                                  # recuperar c_0 y c_n
        c[0] = (1 + h[0] / h[1]) * c[1] - (h[0] / h[1]) * c[2]
        c[n] = (1 + h[-1] / h[-2]) * c[n - 1] - (h[-1] / h[-2]) * c[n - 2]
    b = (a[1:] - a[:-1]) / h - h * (2 * c[:-1] + c[1:]) / 3
    d = (c[1:] - c[:-1]) / (3 * h)
    return SplineCubico(x, a[:-1].copy(), b, c, d, frontera)


def es_diagonal_dominante(sub, diag, sup) -> bool:
    fuera = np.abs(sub) + np.abs(sup)
    fuera[0], fuera[-1] = abs(sup[0]), abs(sub[-1])
    return bool(np.all(np.abs(diag) > fuera))


# =============================================================================================
# 2. Modelos de comparación (mismos nombres y configuración que la §4 del cuaderno)
# =============================================================================================
def construir_modelos(x, y):
    """Devuelve {nombre: interpolante} entrenado con (x, y). Igual que en el cuaderno."""
    pend = np.gradient(y, x)
    return {
        NAK: CubicSpline(x, y, bc_type="not-a-knot"),
        NAT: CubicSpline(x, y, bc_type="natural"),
        SUJ: CubicSpline(x, y, bc_type=((1, pend[0]), (1, pend[-1]))),
        HER: CubicHermiteSpline(x, y, pend),
        PCH: PchipInterpolator(x, y),
        LIN: (lambda t, x=x, y=y: np.interp(t, x, y)),
        LAG: BarycentricInterpolator(x, y),
    }


def fabrica(nombre):
    """Función (x, y) -> interpolante para un único método."""
    return lambda xs, ys: construir_modelos(xs, ys)[nombre]


# =============================================================================================
# 3. Verificación
# =============================================================================================
def verificar_contra_scipy(x=X_KM, y=PM25) -> pd.DataFrame:
    """Compara la implementación propia con SciPy para las 3 fronteras, en la malla uniforme y en
    las 8 mallas no uniformes que aparecen en la LOOCV (al retirar cada estación interna)."""
    filas = []
    mallas = [("uniforme (10 nodos)", np.ones(len(x), bool))]
    for i in range(1, len(x) - 1):
        m = np.ones(len(x), bool)
        m[i] = False
        mallas.append((f"LOOCV sin {ESTACIONES[i]}", m))
    t = np.linspace(x[0], x[-1], 2001)
    for frontera in ["natural", "sujeta", "not-a-knot"]:
        dif_coef, dif_val, dominante = 0.0, 0.0, True
        for _, m in mallas:
            xs, ys = x[m], y[m]
            if frontera == "sujeta":
                g = np.gradient(ys, xs)
                propio, ref = spline_propio(xs, ys, "sujeta", (g[0], g[-1])), CubicSpline(xs, ys, bc_type=((1, g[0]), (1, g[-1])))
                sist = sistema_tridiagonal(xs, ys, "sujeta", (g[0], g[-1]))
            else:
                propio, ref = spline_propio(xs, ys, frontera), CubicSpline(xs, ys, bc_type=frontera)
                sist = sistema_tridiagonal(xs, ys, frontera)
            mio = np.column_stack([propio.d, propio.c[:-1], propio.b, propio.a]).T
            dif_coef = max(dif_coef, float(np.max(np.abs(mio - ref.c))))
            dif_val = max(dif_val, float(np.max(np.abs(propio(t) - ref(t)))))
            dominante &= es_diagonal_dominante(*sist[:3])
        filas.append({"Frontera": frontera, "Mallas probadas": len(mallas),
                      "máx |Δ coeficientes|": dif_coef, "máx |Δ S(x)|": dif_val,
                      "Diagonal estrictamente dominante": "sí" if dominante else "no"})
    return pd.DataFrame(filas)


def saltos_en_nodos(s: SplineCubico) -> dict:
    """Saltos |S⁽ᵏ⁾(x_i⁺) − S⁽ᵏ⁾(x_i⁻)| en los nodos internos, k = 0…3."""
    h = np.diff(s.x)[:-1]
    a, b, c, d = s.a[:-1], s.b[:-1], s.c[:-2], s.d[:-1]
    izq = [a + b * h + c * h**2 + d * h**3, b + 2 * c * h + 3 * d * h**2, 2 * c + 6 * d * h, 6 * d]
    der = [s.a[1:], s.b[1:], 2 * s.c[1:-1], 6 * s.d[1:]]
    return {f"S{'′' * k}" if k else "S": np.abs(der[k] - izq[k]) for k in range(4)}


def tabla_continuidad(x=X_KM, y=PM25) -> pd.DataFrame:
    filas = []
    for frontera in ["natural", "not-a-knot"]:
        sj = saltos_en_nodos(spline_propio(x, y, frontera))
        s3 = sj["S′′′"]
        filas.append({"Frontera": frontera, "máx salto S": sj["S"].max(), "máx salto S′": sj["S′"].max(),
                      "máx salto S″": sj["S′′"].max(), "salto S‴ en x₁": s3[0], "salto S‴ en x₈": s3[-1],
                      "máx salto S‴ (otros nodos)": s3[1:-1].max()})
    return pd.DataFrame(filas)


def perfil_referencia(t):
    """Pluma gaussiana suave que imita el perfil del corredor (sólo para verificar convergencia)."""
    return 40.0 + 52.0 * np.exp(-((np.asarray(t) - 4.1) / 2.4) ** 2)


def perfil_referencia_derivada(t):
    t = np.asarray(t)
    return 52.0 * np.exp(-((t - 4.1) / 2.4) ** 2) * (-2.0 * (t - 4.1) / 2.4**2)


def estudio_convergencia(n_tramos=(9, 18, 36, 72, 144)) -> pd.DataFrame:
    """Error máximo ‖f − S‖∞ al refinar la red (h = 9/n) y orden observado p = log2(E_h / E_{h/2})."""
    tt = np.linspace(0.0, 9.0, 20001)
    f, fp = perfil_referencia, perfil_referencia_derivada
    metodos = {
        NAK: lambda xs, ys: CubicSpline(xs, ys, bc_type="not-a-knot"),
        NAT: lambda xs, ys: CubicSpline(xs, ys, bc_type="natural"),
        "Spline sujeta (f′ exacta)": lambda xs, ys: CubicSpline(xs, ys, bc_type=((1, fp(xs[0])), (1, fp(xs[-1])))),
        SUJ: fabrica(SUJ),
        LIN: fabrica(LIN),
    }
    filas = []
    for nombre, fit in metodos.items():
        errores = []
        for n in n_tramos:
            xs = np.linspace(0.0, 9.0, n + 1)
            errores.append(float(np.max(np.abs(fit(xs, f(xs))(tt) - f(tt)))))
        for k, n in enumerate(n_tramos):
            orden = np.log2(errores[k - 1] / errores[k]) if k else np.nan
            filas.append({"Método": nombre, "n tramos": n, "h (km)": 9.0 / n, "Error máx.": errores[k],
                          "Orden observado": orden})
    return pd.DataFrame(filas)


def estudio_runge(n_nodos=(5, 7, 9, 11, 13, 15, 17, 19, 21)) -> pd.DataFrame:
    """Función de Runge f(t) = 1/(1+25t²) en [−1, 1]: Lagrange equiespaciado vs spline not-a-knot."""
    f = lambda t: 1.0 / (1.0 + 25.0 * t**2)
    tt = np.linspace(-1, 1, 4001)
    filas = []
    for m in n_nodos:
        xs = np.linspace(-1, 1, m)
        filas.append({"Nodos": m,
                      "Error máx. Lagrange": float(np.max(np.abs(BarycentricInterpolator(xs, f(xs))(tt) - f(tt)))),
                      "Error máx. spline": float(np.max(np.abs(CubicSpline(xs, f(xs))(tt) - f(tt))))})
    return pd.DataFrame(filas)


# =============================================================================================
# 4. Fenómeno de Runge cuantificado en los nodos del corredor
# =============================================================================================
def funciones_cardinales(fit, x, t) -> np.ndarray:
    """ℓ_j(t): interpolante de los datos e_j (1 en el nodo j, 0 en los demás). Fila j = ℓ_j."""
    return np.array([fit(x, np.eye(len(x))[j])(t) for j in range(len(x))])


def funcion_lebesgue(fit, x, t) -> np.ndarray:
    """λ(t) = Σ_j |ℓ_j(t)|.  Para todo dato con error |δ_j| ≤ δ, el error inducido es ≤ λ(t)·δ."""
    return np.abs(funciones_cardinales(fit, x, t)).sum(axis=0)


def tabla_lebesgue(x=X_KM) -> pd.DataFrame:
    t = np.linspace(x[0], x[-1], 9001)
    filas = []
    for nombre in [LAG, NAK, NAT, SUJ, HER, LIN]:
        lam = funcion_lebesgue(fabrica(nombre), x, t)
        filas.append({"Método": nombre, "Constante de Lebesgue Λ": lam.max(), "en km": t[lam.argmax()]})
    return pd.DataFrame(filas)


def crecimiento_lebesgue(n_nodos=(4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30)) -> pd.DataFrame:
    """Λ_n para nodos equiespaciados: exponencial en Lagrange, acotada en el spline."""
    filas = []
    for m in n_nodos:
        xs = np.linspace(0.0, 9.0, m)
        t = np.linspace(0.0, 9.0, 300 * m + 1)
        fila = {"Nodos": m, "Asintótica 2^n/(e·(n−1)·ln(n−1))": 2.0**m / (np.e * (m - 1) * np.log(m - 1))}
        for nombre in [LAG, NAK, NAT]:
            fit = (lambda xs_, ys_: BarycentricInterpolator(xs_, ys_)) if nombre == LAG else \
                  (lambda xs_, ys_, bc="not-a-knot" if nombre == NAK else "natural": CubicSpline(xs_, ys_, bc_type=bc))
            fila[nombre] = funcion_lebesgue(fit, xs, t).max()
        filas.append(fila)
    return pd.DataFrame(filas)


def polinomio_nodal(x, t):
    """ω(t) = Π (t − x_i), factor geométrico del error de Lagrange."""
    return np.prod([t - xi for xi in x], axis=0)


def razon_nodal(x=X_KM) -> dict:
    t = np.linspace(x[0], x[-1], 9001)
    w = np.abs(polinomio_nodal(x, t))
    borde = w[t <= x[1]].max()
    centro = w[(t >= x[4]) & (t <= x[5])].max()
    return {"max|ω| tramo de borde [0,1]": borde, "max|ω| tramo central [4,5]": centro, "razón": borde / centro}


def localidad_spline(x=X_KM, j=4) -> pd.DataFrame:
    """Influencia de UNA estación (j) sobre cada tramo: max|ℓ_j| por tramo, para spline y Lagrange.
    En la malla uniforme el spline decae como (2 − √3) ≈ 0.268 por tramo."""
    t = np.linspace(x[0], x[-1], 9001)
    filas = []
    ls = funciones_cardinales(fabrica(NAK), x, t)[j]
    lg = funciones_cardinales(fabrica(LAG), x, t)[j]
    for i in range(len(x) - 1):
        m = (t >= x[i]) & (t <= x[i + 1])
        filas.append({"Tramo [km]": f"[{x[i]:g}, {x[i + 1]:g}]", "Distancia a la estación (tramos)": min(abs(i - j), abs(i + 1 - j)),
                      "max|ℓ| spline not-a-knot": np.abs(ls[m]).max(), "max|ℓ| Lagrange": np.abs(lg[m]).max()})
    return pd.DataFrame(filas)


def oscilaciones(x=X_KM, y=PM25) -> pd.DataFrame:
    """Variación total y extremos locales del interpolante vs. los datos (detección de oscilaciones espurias)."""
    t = np.linspace(x[0], x[-1], 18001)
    tv_datos = float(np.abs(np.diff(y)).sum())
    ext_datos = int(np.sum(np.sign(np.diff(y))[:-1] != np.sign(np.diff(y))[1:]))
    filas = []
    modelos = construir_modelos(x, y)
    for nombre in ORDEN_MODELOS:
        v = modelos[nombre](t)
        dv = np.diff(v)
        dv = dv[np.abs(dv) > 1e-12]
        filas.append({"Método": nombre, "Variación total": float(np.abs(np.diff(v)).sum()),
                      "Exceso VT vs datos": float(np.abs(np.diff(v)).sum()) - tv_datos,
                      "Extremos locales": int(np.sum(np.sign(dv[:-1]) != np.sign(dv[1:]))),
                      "Extremos en los datos": ext_datos,
                      "Mínimo": float(v.min()), "Máximo": float(v.max())})
    return pd.DataFrame(filas)


# =============================================================================================
# 5. Validación cruzada Leave-One-Out en los 8 nodos internos
# =============================================================================================
def loocv(x=X_KM, y=PM25, nombres=None):
    """LOOCV en los nodos internos (no se extrapola: E1 y E10 nunca se retiran).

    Devuelve (pred, err): predicción ŷ_{-i}(x_i) y error con signo e_i = ŷ_{-i}(x_i) − y_i.
    """
    nombres = nombres or ORDEN_MODELOS
    pred, err = [], []
    for i in range(1, len(x) - 1):
        m = np.ones(len(x), dtype=bool)
        m[i] = False
        mods = construir_modelos(x[m], y[m])
        p = {n: float(mods[n](x[i])) for n in nombres}
        pred.append({"Estación": ESTACIONES[i], "km": x[i], "Real": y[i], **p})
        err.append({"Estación": ESTACIONES[i], "km": x[i], "Real": y[i], **{n: p[n] - y[i] for n in nombres}})
    return pd.DataFrame(pred).set_index("Estación"), pd.DataFrame(err).set_index("Estación")


def metricas(err: pd.DataFrame, nombres=None) -> pd.DataFrame:
    """RMSE = √(Σe²/8), MAE, error máximo absoluto, estación peor y sesgo medio."""
    nombres = nombres or [c for c in err.columns if c not in ("km", "Real")]
    e = err[nombres]
    ea = e.abs()
    return pd.DataFrame({"RMSE": np.sqrt((e**2).mean()), "MAE": ea.mean(), "Error máx.": ea.max(),
                         "Estación peor": ea.idxmax(), "Sesgo medio": e.mean()}).sort_values("RMSE")


def seleccionar_frontera(resumen: pd.DataFrame) -> str:
    """Frontera con menor RMSE LOOCV entre los splines C² (criterio de la §5 del cuaderno)."""
    splines = resumen.loc[[n for n in resumen.index if n.startswith("Spline")]]
    return splines["RMSE"].idxmin()


# =============================================================================================
# 6. Robustez de la validación frente al ruido instrumental
# =============================================================================================
def matriz_loocv(nombre, x=X_KM) -> np.ndarray:
    """Para un método LINEAL en los datos, el vector de errores LOOCV es e = M·y (M de 8×10).
    Fila i: pesos w_ij de la predicción en x_i y −1 en la columna i."""
    if nombre not in MODELOS_LINEALES:
        raise ValueError(f"{nombre} no es lineal en los datos")
    n = len(x)
    M = np.zeros((n - 2, n))
    for k, i in enumerate(range(1, n - 1)):
        m = np.ones(n, bool)
        m[i] = False
        for j_local, j in enumerate(np.where(m)[0]):
            M[k, j] = float(construir_modelos(x[m], np.eye(n - 1)[j_local])[nombre](x[i]))
        M[k, i] = -1.0
    return M


def factores_ruido(x=X_KM) -> pd.DataFrame:
    """Factor de amplificación ‖M_i‖₂ = √(1 + Σ_j w_ij²): si cada lectura tiene ruido independiente
    N(0, σ²), la desviación típica del error LOOCV en la estación i es σ·‖M_i‖₂."""
    filas = {nombre: np.linalg.norm(matriz_loocv(nombre, x), axis=1) for nombre in MODELOS_LINEALES}
    df = pd.DataFrame(filas, index=ESTACIONES[1:-1])
    df.loc["RMS"] = np.sqrt((df**2).mean())
    return df


def loocv_bajo_ruido(x=X_KM, y=PM25, sigma=SIGMA, n=N_MC_VALIDACION, semilla=SEMILLA):
    """LOOCV repetida sobre n realizaciones y + ε, ε ~ N(0, σ²). Devuelve RMSE por realización."""
    rng = np.random.default_rng(semilla)
    Y = y + rng.normal(0.0, sigma, size=(n, len(y)))
    rmse = {}
    for nombre in ORDEN_MODELOS:
        if nombre in MODELOS_LINEALES:
            E = Y @ matriz_loocv(nombre, x).T                      # (n, 8) en una sola operación
        else:                                                      # PCHIP: no lineal → ajuste explícito
            E = np.array([[float(PchipInterpolator(np.delete(x, i), np.delete(yk, i))(x[i])) - yk[i]
                           for i in range(1, len(x) - 1)] for yk in Y])
        rmse[nombre] = np.sqrt((E**2).mean(axis=1))
    return pd.DataFrame(rmse)


def resumen_ruido(rmse_mc: pd.DataFrame, resumen_nominal: pd.DataFrame) -> pd.DataFrame:
    gana = rmse_mc.idxmin(axis=1).value_counts(normalize=True)
    df = pd.DataFrame({
        "RMSE nominal": resumen_nominal.loc[rmse_mc.columns, "RMSE"],
        "RMSE medio (ruido)": rmse_mc.mean(),
        "P2.5": rmse_mc.quantile(0.025), "P97.5": rmse_mc.quantile(0.975),
        "% realizaciones donde es el mejor": 100 * gana.reindex(rmse_mc.columns).fillna(0.0),
        "Percentil del RMSE nominal": [100 * (rmse_mc[c] <= resumen_nominal.loc[c, "RMSE"]).mean() for c in rmse_mc.columns],
    })
    return df.sort_values("RMSE medio (ruido)")


def banda_mc_cuaderno(x=X_KM, y=PM25, xf=XF, sigma=SIGMA, n=100, semilla=SEMILLA) -> dict:
    """Réplica independiente de la banda del 95 % de la §8 del cuaderno (mismo generador y semilla)
    para contrastarla con la amplificación de ruido calculada aquí."""
    rng = np.random.default_rng(semilla)
    Yk = y + rng.normal(0.0, sigma, size=(n, len(y)))
    curvas = np.array([CubicSpline(x, yk, bc_type="not-a-knot")(xf) for yk in Yk])
    ancho = np.percentile(curvas, 97.5, axis=0) - np.percentile(curvas, 2.5, axis=0)
    interior = (xf >= 2) & (xf <= 7)
    return {"ancho_max": float(ancho.max()), "km_max": float(xf[ancho.argmax()]),
            "ancho_min": float(ancho.min()), "km_min": float(xf[ancho.argmin()]),
            "ancho_medio_borde": float(ancho[~interior].mean()), "ancho_medio_interior": float(ancho[interior].mean())}


def ruido_efectivo_estimado(resumen_nominal: pd.DataFrame, factores: pd.DataFrame) -> pd.DataFrame:
    """Si toda la discrepancia LOOCV fuese ruido: σ_ef ≈ RMSE / RMS(‖M_i‖) (cota aproximada)."""
    filas = [{"Método": n, "σ efectivo máx. (µg/m³)": resumen_nominal.loc[n, "RMSE"] / factores.loc["RMS", n]}
             for n in MODELOS_LINEALES]
    return pd.DataFrame(filas).sort_values("σ efectivo máx. (µg/m³)")


# =============================================================================================
# 7. Sensibilidad de las alertas a la frontera / método
# =============================================================================================
def tramos_sobre_umbral(f, umbral, a=0.0, b=9.0):
    """Intervalos [x_ini, x_fin] donde f(x) ≥ umbral (raíces exactas si f es polinomio por tramos)."""
    if hasattr(f, "solve"):
        r = np.asarray(f.solve(umbral, extrapolate=False), dtype=float)
    else:
        xx = np.linspace(a, b, 90001)
        g = f(xx) - umbral
        idx = np.where(g[:-1] * g[1:] < 0)[0]                  # cambio de signo estricto
        r = np.concatenate([xx[g == 0],                         # raíces que caen justo en la malla
                            xx[idx] - g[idx] * (xx[idx + 1] - xx[idx]) / (g[idx + 1] - g[idx])])
    r = r[np.isfinite(r) & (r > a) & (r < b)]
    pts = np.unique(np.concatenate([[a], r, [b]]))
    return [(p, q) for p, q in zip(pts[:-1], pts[1:]) if f(0.5 * (p + q)) >= umbral]


def sensibilidad_alertas(x=X_KM, y=PM25) -> pd.DataFrame:
    modelos = construir_modelos(x, y)
    ref = {u: tramos_sobre_umbral(modelos[NAK], u) for u in (UMBRAL_AMARILLA, UMBRAL_ROJA)}
    filas = []
    for nombre in ORDEN_MODELOS:
        fila = {"Método": nombre}
        for etiqueta, u in (("Amarilla", UMBRAL_AMARILLA), ("Roja", UMBRAL_ROJA)):
            tr = tramos_sobre_umbral(modelos[nombre], u)
            p, q = (tr[0][0], tr[-1][1]) if tr else (np.nan, np.nan)
            fila[f"{etiqueta} desde km"], fila[f"{etiqueta} hasta km"] = p, q
            fila[f"{etiqueta}: máx. desfase vs not-a-knot (m)"] = 1000 * max(abs(p - ref[u][0][0]), abs(q - ref[u][-1][1]))
        filas.append(fila)
    return pd.DataFrame(filas)


def diferencia_fronteras(x=X_KM, y=PM25) -> pd.DataFrame:
    """Dónde difieren las fronteras: |S_nak − S_otra| en borde (km 0–1 y 8–9) vs interior."""
    m = construir_modelos(x, y)
    t = np.linspace(0, 9, 9001)
    borde = (t <= 1) | (t >= 8)
    filas = []
    for nombre in [NAT, SUJ]:
        d = np.abs(m[NAK](t) - m[nombre](t))
        filas.append({"Comparación": f"{NAK} vs {nombre}", "máx |ΔS| en tramos de borde": d[borde].max(),
                      "máx |ΔS| en el interior (km 1–8)": d[~borde].max(),
                      "S″(0) / S″(9) not-a-knot": f"{float(m[NAK](0.0, 2)):.2f} / {float(m[NAK](9.0, 2)):.2f}",
                      f"S″(0) / S″(9) {nombre.split()[-1]}": f"{float(m[nombre](0.0, 2)):.2f} / {float(m[nombre](9.0, 2)):.2f}"})
    return pd.DataFrame(filas)


# =============================================================================================
# 8. Paquete de resultados (una sola fuente para consola, informe y documentación)
# =============================================================================================
def calcular_todo(n_mc=N_MC_VALIDACION, semilla=SEMILLA) -> dict:
    pred, err = loocv()
    res = metricas(err)
    elegido = seleccionar_frontera(res)
    fac = factores_ruido()
    rmse_mc = loocv_bajo_ruido(n=n_mc, semilla=semilla)
    sub, diag, sup, rhs, _ = sistema_tridiagonal(X_KM, PM25, "natural")
    A_nat = matriz_densa(sub, diag, sup)
    sub_k, diag_k, sup_k, rhs_k, _ = sistema_tridiagonal(X_KM, PM25, "not-a-knot")
    A_nak = matriz_densa(sub_k, diag_k, sup_k)
    return {
        "pred": pred, "err": err, "resumen": res, "elegido": elegido,
        "verificacion": verificar_contra_scipy(), "continuidad": tabla_continuidad(),
        "convergencia": estudio_convergencia(), "runge": estudio_runge(),
        "lebesgue": tabla_lebesgue(), "crecimiento": crecimiento_lebesgue(),
        "nodal": razon_nodal(), "localidad": localidad_spline(), "oscilaciones": oscilaciones(),
        "factores": fac, "rmse_mc": rmse_mc, "ruido": resumen_ruido(rmse_mc, res),
        "sigma_ef": ruido_efectivo_estimado(res, fac), "alertas": sensibilidad_alertas(),
        "fronteras": diferencia_fronteras(), "banda": banda_mc_cuaderno(),
        "A_nat": A_nat, "rhs_nat": rhs, "cond_nat": float(np.linalg.cond(A_nat)),
        "A_nak": A_nak, "rhs_nak": rhs_k, "cond_nak": float(np.linalg.cond(A_nak)),
        "coef_nak": spline_propio(X_KM, PM25, "not-a-knot").coeficientes(),
        "n_mc": n_mc, "semilla": semilla,
    }


def imprimir_reporte(R: dict) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    res = R["resumen"]
    linea = "=" * 96
    print(linea)
    print("AirSpline · Modelado y Validación — Estudio de Caso 3 (PM₂.₅, corredor de 9 km)")
    print(linea)
    print("\n[1] Sistema tridiagonal (frontera natural, h = 1 km) — A·c = r:")
    print(pd.DataFrame(R["A_nat"], index=[f"c{i}" for i in range(1, 9)], columns=[f"c{i}" for i in range(1, 9)]).astype(int))
    print("    r =", R["rhs_nat"].astype(int), f"   κ₂(A) = {R['cond_nat']:.3f}")
    print(f"    not-a-knot: κ₂(A) = {R['cond_nak']:.3f}  (primera/última fila: 6·c₁ = r₁, 6·c₈ = r₈)")
    print("\n[2] Verificación implementación propia (Thomas) vs SciPy:")
    print(R["verificacion"].to_string(index=False))
    print("\n    Continuidad en nodos internos:")
    print(R["continuidad"].to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    conv = R["convergencia"]
    print("\n[3] Orden de convergencia observado (pluma gaussiana de referencia, h: 1 → 1/16 km):")
    print(conv.pivot(index="Método", columns="n tramos", values="Orden observado").round(2).to_string())
    print("\n[4] Runge — constante de Lebesgue en los 10 nodos del corredor:")
    print(R["lebesgue"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    nod = R["nodal"]
    print(f"    |ω(x)| máximo en el tramo de borde es {nod['razón']:.1f}× el del tramo central.")
    print(R["crecimiento"].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\n    Localidad (influencia de la estación E5, km 4):")
    print(R["localidad"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n    Oscilaciones espurias:")
    print(R["oscilaciones"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n[5] LOOCV — error con signo ŷ₋ᵢ(xᵢ) − yᵢ (µg/m³):")
    print(R["err"].round(3).to_string())
    print("\n    Métricas (ordenadas por RMSE):")
    print(res.round(3).to_string())
    e = R["elegido"]
    print(f"\n    ⇒ Frontera elegida: {e}. RMSE = {res.loc[e, 'RMSE']:.3f} µg/m³, error máx. = "
          f"{res.loc[e, 'Error máx.']:.3f} µg/m³ ({res.loc[e, 'Estación peor']}).")
    print(f"      Reducción de RMSE: {100 * (1 - res.loc[e, 'RMSE'] / res.loc[LIN, 'RMSE']):.1f} % vs lineal · "
          f"{100 * (1 - res.loc[e, 'RMSE'] / res.loc[LAG, 'RMSE']):.1f} % vs Lagrange")
    print("\n[6] Robustez — factor de amplificación del ruido √(1+Σw²) por estación:")
    print(R["factores"].round(3).to_string())
    print(f"\n    LOOCV bajo ruido σ = {SIGMA} µg/m³ ({R['n_mc']} realizaciones, semilla {R['semilla']}):")
    print(R["ruido"].round(3).to_string())
    print("\n    Ruido efectivo compatible con la LOOCV nominal:")
    print(R["sigma_ef"].round(3).to_string(index=False))
    print("\n[7] Sensibilidad de las alertas al método / frontera:")
    print(R["alertas"].round(3).to_string(index=False))
    print(R["fronteras"].round(3).to_string(index=False))
    b = R["banda"]
    print(f"\n    Réplica banda 95 % de la §8 (N = 100): ancho máx. {b['ancho_max']:.2f} µg/m³ en km {b['km_max']:.1f}, "
          f"mín. {b['ancho_min']:.2f} en km {b['km_min']:.1f}; media borde {b['ancho_medio_borde']:.2f} vs "
          f"interior (km 2–7) {b['ancho_medio_interior']:.2f}")
    print(linea)


# =============================================================================================
# 9. Figuras (estilo sobrio, paleta del cuaderno + línea/marcador como codificación secundaria)
# =============================================================================================
def _estilo_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 220, "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": "#9ca3af", "axes.linewidth": 0.6, "axes.labelcolor": TINTA,
        "text.color": TINTA, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.markersize": 4.5, "font.family": "DejaVu Sans",
        "axes.titleweight": "bold", "axes.titlelocation": "left", "figure.facecolor": "white",
    })
    return plt


def generar_figuras(R: dict, carpeta: Path) -> list[Path]:
    plt = _estilo_mpl()
    from matplotlib.ticker import FixedLocator
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = []

    def guardar(fig, nombre):
        ruta = carpeta / nombre
        fig.savefig(ruta, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        rutas.append(ruta)

    # ---- Fig. 1: LOOCV --------------------------------------------------------------------
    res, err = R["resumen"], R["err"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.35), gridspec_kw={"width_ratios": [1.25, 1]})
    for n in [NAK, NAT, LIN, LAG]:
        a1.plot(err["km"], err[n].abs(), linestyle=LINEA[n], marker=MARCA[n], color=COLOR[n], label=n,
                markeredgecolor="white", markeredgewidth=0.6)
    a1.set_yscale("symlog", linthresh=2, linscale=1.2)
    a1.yaxis.set_major_locator(FixedLocator([0, 0.5, 1, 1.5, 2, 5, 10, 20]))
    a1.set_yticklabels(["0", "0.5", "1", "1.5", "2", "5", "10", "20"])
    a1.set_xticks(err["km"])
    a1.set_xticklabels([f"{e}\nkm {k:g}" for e, k in zip(err.index, err["km"])])
    a1.set(title="(a) Error absoluto LOOCV por estación retirada", ylabel="|ŷ₋ᵢ − yᵢ| (µg/m³)")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=4, fontsize=6.8)
    orden = res.index[::-1]
    yy = np.arange(len(orden))
    elegido = R["elegido"]
    a2.barh(yy, res.loc[orden, "RMSE"], height=0.62,
            color=[COLOR[NAK] if n == elegido else "#cbd5e1" for n in orden], label="RMSE")
    a2.plot(res.loc[orden, "Error máx."], yy, "|", color=TINTA, markersize=9, markeredgewidth=1.6, label="Error máx.")
    for k, n in enumerate(orden):
        a2.text(res.loc[n, "Error máx."] + 0.35, k, f"{res.loc[n, 'RMSE']:.2f} / {res.loc[n, 'Error máx.']:.2f}",
                va="center", fontsize=6.8, color=TINTA_2)
    a2.set_yticks(yy)
    a2.set_yticklabels([n.replace(" (np.gradient)", "").replace(" (grado n-1)", "") for n in orden])
    a2.set_xlim(0, 24)
    a2.set(title="(b) RMSE (barra) y error máx. (|)", xlabel="µg/m³  ·  etiqueta: RMSE / error máx.")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    guardar(fig, "fig1_loocv.png")

    # ---- Fig. 2: Runge / Lebesgue ------------------------------------------------------------
    t = np.linspace(0, 9, 3001)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.3))
    for n in [NAK, NAT, LAG]:
        a1.plot(t, funcion_lebesgue(fabrica(n), X_KM, t), linestyle=LINEA[n], color=COLOR[n], label=n)
    a1.plot(X_KM, np.ones_like(X_KM), "o", color=TINTA, markersize=3.5, label="estaciones (λ = 1)")
    a1.set_yscale("log")
    a1.set(title="(a) Función de Lebesgue λ(x) en el corredor", xlabel="km", ylabel="λ(x) = Σ|ℓⱼ(x)|")
    a1.legend(fontsize=6.8, loc="upper center", ncol=2)
    a1.set_ylim(0.8, 60)
    cr = R["crecimiento"]
    for n in [NAK, NAT, LAG]:
        a2.plot(cr["Nodos"], cr[n], linestyle=LINEA[n], marker=MARCA[n], color=COLOR[n], label=n,
                markeredgecolor="white", markeredgewidth=0.5)
    a2.plot(cr["Nodos"], cr["Asintótica 2^n/(e·(n−1)·ln(n−1))"], color="#9ca3af", lw=1, label="2ⁿ/(e(n−1)ln(n−1))")
    a2.axvline(10, color="#9ca3af", lw=0.8)
    a2.text(10.5, 4.0, "10 estaciones", fontsize=6.8, color=TINTA_2)
    a2.set_yscale("log")
    a2.set(title="(b) Constante de Lebesgue Λ vs. número de nodos", xlabel="nodos equiespaciados", ylabel="Λ")
    a2.legend(fontsize=6.8, loc="upper left")
    fig.tight_layout()
    guardar(fig, "fig2_runge_lebesgue.png")

    # ---- Fig. 3: convergencia ----------------------------------------------------------------
    conv, rg = R["convergencia"], R["runge"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.45))
    est = {NAK: (COLOR[NAK], "-", "o"), "Spline sujeta (f′ exacta)": (COLOR[SUJ], ":", "v"),
           NAT: (COLOR[NAT], "--", "s"), LIN: (COLOR[LIN], "-", "x")}
    for n, (col, ls, mk) in est.items():
        d = conv[conv["Método"] == n]
        a1.loglog(d["h (km)"], d["Error máx."], linestyle=ls, marker=mk, color=col, label=n,
                  markeredgecolor="white", markeredgewidth=0.5)
    h = np.array([1.0, 1 / 16])
    a1.loglog(h, 0.06 * h**2, color="#9ca3af", lw=0.9)
    a1.loglog(h, 0.008 * h**4, color="#9ca3af", lw=0.9)
    a1.text(0.25, 0.06 * 0.25**2 * 0.45, "O(h²)", fontsize=7, color=TINTA_2, ha="center", va="top")
    a1.text(0.25, 0.008 * 0.25**4 * 0.45, "O(h⁴)", fontsize=7, color=TINTA_2, ha="center", va="top")
    a1.invert_xaxis()
    a1.set(title="(a) Convergencia al refinar la red", xlabel="h (km)", ylabel="‖f − S‖∞ (µg/m³)")
    a1.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    a2.semilogy(rg["Nodos"], rg["Error máx. Lagrange"], linestyle=LINEA[LAG], marker=MARCA[LAG], color=COLOR[LAG],
                label="Lagrange equiespaciado", markeredgecolor="white", markeredgewidth=0.5)
    a2.semilogy(rg["Nodos"], rg["Error máx. spline"], linestyle=LINEA[NAK], marker=MARCA[NAK], color=COLOR[NAK],
                label="Spline cúbico not-a-knot", markeredgecolor="white", markeredgewidth=0.5)
    a2.set(title="(b) Función de Runge 1/(1+25x²)", xlabel="nodos", ylabel="error máx.")
    a2.set_xticks(rg["Nodos"])
    a2.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)
    fig.tight_layout()
    guardar(fig, "fig3_convergencia.png")

    # ---- Fig. 4: robustez ante ruido ---------------------------------------------------------
    fac, mc = R["factores"].drop(index="RMS"), R["rmse_mc"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1, 1.2]})
    km = X_KM[1:-1]
    for n in [NAK, NAT, LIN, LAG]:
        a1.plot(km, fac[n], linestyle=LINEA[n], marker=MARCA[n], color=COLOR[n], label=n,
                markeredgecolor="white", markeredgewidth=0.5)
    a1.set_yscale("log")
    a1.yaxis.set_major_locator(FixedLocator([1, 1.5, 2.5, 5, 10, 25]))
    a1.set_yticklabels(["1", "1.5", "2.5", "5", "10", "25"])
    a1.set_xticks(km)
    a1.set_xticklabels(ESTACIONES[1:-1])
    a1.set(title="(a) Amplificación del ruido √(1+Σw²)", ylabel="σ(eᵢ) / σ")
    a1.legend(fontsize=6.6, loc="upper center", ncol=2)
    a1.set_ylim(1, 60)
    orden = list(R["ruido"].index)
    datos = [mc[n].values for n in orden]
    bp = a2.boxplot(datos, orientation="horizontal", whis=(2.5, 97.5), showfliers=False, widths=0.55, patch_artist=True,
                    medianprops={"color": TINTA, "linewidth": 1.2})
    for caja, n in zip(bp["boxes"], orden):
        caja.set(facecolor=COLOR[n], alpha=0.35, edgecolor=COLOR[n])
    for elem in ("whiskers", "caps"):
        for linea, n in zip(bp[elem], np.repeat(orden, 2)):
            linea.set(color=COLOR[n])
    a2.plot(R["resumen"].loc[orden, "RMSE"], np.arange(1, len(orden) + 1), "D", color=TINTA, markersize=4,
            label="datos del caso")
    a2.set_yticks(np.arange(1, len(orden) + 1))
    a2.set_yticklabels([n.replace(" (np.gradient)", "").replace(" (grado n-1)", "") for n in orden])
    a2.set_xscale("log")
    a2.set_xlim(0.5, 100)
    a2.set(title=f"(b) RMSE LOOCV bajo ruido σ = {SIGMA} (N = {R['n_mc']})", xlabel="RMSE (µg/m³), P2.5–P97.5")
    a2.grid(axis="y", visible=False)
    a2.legend(fontsize=6.6, loc="lower right")
    fig.tight_layout()
    guardar(fig, "fig4_robustez_ruido.png")

    # ---- Fig. 5: efecto de la frontera ------------------------------------------------------
    mods = construir_modelos(X_KM, PM25)
    t = np.linspace(0, 9, 2001)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for n in [NAK, NAT, SUJ]:
        a1.plot(t, mods[n](t), linestyle=LINEA[n], color=COLOR[n], label=n)
        a2.plot(t, mods[n](t, 2), linestyle=LINEA[n], color=COLOR[n], label=n)
    a1.plot(X_KM, PM25, "o", color=TINTA, markersize=4, label="estaciones", zorder=5)
    a1.axhline(UMBRAL_AMARILLA, color="#a16207", lw=0.8)
    a1.axhline(UMBRAL_ROJA, color="#b91c1c", lw=0.8)
    a1.text(0.05, UMBRAL_AMARILLA + 1, "70 µg/m³", fontsize=6.6, color="#a16207")
    a1.text(0.05, UMBRAL_ROJA + 1, "90 µg/m³", fontsize=6.6, color="#b91c1c")
    a1.set(title="(a) Perfil interpolado S(x)", xlabel="km", ylabel="PM₂.₅ (µg/m³)")
    a1.legend(fontsize=6.6, loc="lower center")
    a2.axhline(0, color="#9ca3af", lw=0.8)
    for xi in X_KM:
        a2.axvline(xi, color=REJILLA, lw=0.6, zorder=0)
    a2.set(title="(b) Curvatura S″(x) (natural: S″ = 0 en km 0 y 9)", xlabel="km", ylabel="µg/m³/km²")
    a2.legend(fontsize=6.6, loc="upper center")
    a2.set_ylim(-15, 20)
    fig.tight_layout()
    guardar(fig, "fig5_fronteras.png")
    return rutas


# =============================================================================================
# 10. Exportación de tablas (CSV) y macros LaTeX para el informe
# =============================================================================================
def _fmt(v, nd=3):
    return f"{v:.{nd}f}"


def exportar(R: dict, carpeta: Path) -> None:
    gen = carpeta / "generado"
    gen.mkdir(parents=True, exist_ok=True)
    R["err"].to_csv(gen / "loocv_errores_con_signo.csv", float_format="%.6f")
    R["resumen"].to_csv(gen / "loocv_metricas.csv", float_format="%.6f")
    R["verificacion"].to_csv(gen / "verificacion_vs_scipy.csv", index=False)
    R["continuidad"].to_csv(gen / "continuidad_nodos.csv", index=False)
    R["convergencia"].to_csv(gen / "convergencia.csv", index=False)
    R["lebesgue"].to_csv(gen / "lebesgue_corredor.csv", index=False)
    R["crecimiento"].to_csv(gen / "lebesgue_crecimiento.csv", index=False)
    R["oscilaciones"].to_csv(gen / "oscilaciones.csv", index=False)
    R["factores"].to_csv(gen / "factores_ruido.csv", float_format="%.6f")
    R["ruido"].to_csv(gen / "loocv_bajo_ruido.csv", float_format="%.6f")
    R["alertas"].to_csv(gen / "sensibilidad_alertas.csv", index=False, float_format="%.4f")
    R["coef_nak"].to_csv(gen / "coeficientes_not_a_knot.csv", index=False, float_format="%.6f")

    res, e = R["resumen"], R["elegido"]
    leb = R["lebesgue"].set_index("Método")
    cr = R["crecimiento"].set_index("Nodos")
    conv = R["convergencia"]
    orden = lambda n: conv[(conv["Método"] == n) & (conv["n tramos"] == 144)]["Orden observado"].iloc[0]
    rz, fac, al = R["ruido"], R["factores"], R["alertas"].set_index("Método")
    loc = R["localidad"]
    osc = R["oscilaciones"].set_index("Método")
    ver = R["verificacion"]
    cont = R["continuidad"].set_index("Frontera")
    sig = R["sigma_ef"].set_index("Método")
    fr = R["fronteras"].set_index("Comparación")
    macros = {
        "RMSEnak": _fmt(res.loc[NAK, "RMSE"]), "EMAXnak": _fmt(res.loc[NAK, "Error máx."]),
        "MAEnak": _fmt(res.loc[NAK, "MAE"]), "PEORnak": res.loc[NAK, "Estación peor"],
        "RMSEnat": _fmt(res.loc[NAT, "RMSE"]), "EMAXnat": _fmt(res.loc[NAT, "Error máx."]),
        "RMSEsuj": _fmt(res.loc[SUJ, "RMSE"]), "RMSElin": _fmt(res.loc[LIN, "RMSE"]),
        "RMSElag": _fmt(res.loc[LAG, "RMSE"]), "EMAXlag": _fmt(res.loc[LAG, "Error máx."]),
        "RMSEher": _fmt(res.loc[HER, "RMSE"]), "RMSEpch": _fmt(res.loc[PCH, "RMSE"]),
        "REDlin": f"{100 * (1 - res.loc[e, 'RMSE'] / res.loc[LIN, 'RMSE']):.1f}",
        "REDlag": f"{100 * (1 - res.loc[e, 'RMSE'] / res.loc[LAG, 'RMSE']):.1f}",
        "REDnat": f"{100 * (1 - res.loc[e, 'RMSE'] / res.loc[NAT, 'RMSE']):.1f}",
        "LEBlag": _fmt(leb.loc[LAG, "Constante de Lebesgue Λ"], 2), "LEBlagkm": _fmt(leb.loc[LAG, "en km"], 2),
        "LEBnak": _fmt(leb.loc[NAK, "Constante de Lebesgue Λ"], 2), "LEBnat": _fmt(leb.loc[NAT, "Constante de Lebesgue Λ"], 2),
        "LEBlagTreinta": f"{cr.loc[30, LAG]:.1e}".replace("e+0", r"\times10^{").replace("e+", r"\times10^{") + "}",
        "LEBnakTreinta": _fmt(cr.loc[30, NAK], 2),
        "RAZONnodal": f"{R['nodal']['razón']:.0f}",
        # localidad medida hacia el interior (tramos [5,6], [6,7], [7,8]), lejos del efecto not-a-knot del borde
        "LOCuno": _fmt(loc.loc[5, "max|ℓ| spline not-a-knot"], 3),
        "LOCdos": _fmt(loc.loc[6, "max|ℓ| spline not-a-knot"], 3),
        "LOCtres": _fmt(loc.loc[7, "max|ℓ| spline not-a-knot"], 3),
        "LOCnakBorde": _fmt(loc.loc[[0, 8], "max|ℓ| spline not-a-knot"].max(), 3),
        "LOClagBorde": _fmt(loc.loc[[0, 8], "max|ℓ| Lagrange"].max(), 2),
        "EXTlag": str(osc.loc[LAG, "Extremos locales"]), "VTlag": _fmt(osc.loc[LAG, "Exceso VT vs datos"], 2),
        "VTnak": _fmt(osc.loc[NAK, "Exceso VT vs datos"], 2), "MINnak": _fmt(osc.loc[NAK, "Mínimo"], 2),
        "ORDnak": _fmt(orden(NAK), 2), "ORDnat": _fmt(orden(NAT), 2), "ORDsujex": _fmt(orden("Spline sujeta (f′ exacta)"), 2),
        "ORDsuj": _fmt(orden(SUJ), 2), "ORDlin": _fmt(orden(LIN), 2),
        "VERmax": f"{ver['máx |Δ coeficientes|'].max():.1e}".replace("e-", r"\times10^{-") + "}",
        "CONTmax": f"{cont[['máx salto S', 'máx salto S′', 'máx salto S″']].to_numpy().max():.1e}".replace("e-", r"\times10^{-") + "}",
        "CONDnat": _fmt(R["cond_nat"], 2), "CONDnak": _fmt(R["cond_nak"], 2),
        "FACnakBorde": _fmt(fac.loc["E2", NAK], 2), "FACnatBorde": _fmt(fac.loc["E2", NAT], 2),
        "FAClagBorde": _fmt(fac.loc["E2", LAG], 1), "FAClin": _fmt(fac.loc["E2", LIN], 3),
        "FACnakRMS": _fmt(fac.loc["RMS", NAK], 2), "FACnatRMS": _fmt(fac.loc["RMS", NAT], 2),
        "MCnak": _fmt(rz.loc[NAK, "RMSE medio (ruido)"], 2), "MCnat": _fmt(rz.loc[NAT, "RMSE medio (ruido)"], 2),
        "MClin": _fmt(rz.loc[LIN, "RMSE medio (ruido)"], 2), "MClag": _fmt(rz.loc[LAG, "RMSE medio (ruido)"], 1),
        "MCher": _fmt(rz.loc[HER, "RMSE medio (ruido)"], 2),
        "MCganaNak": f"{rz.loc[NAK, '% realizaciones donde es el mejor']:.0f}",
        "PCTnak": f"{rz.loc[NAK, 'Percentil del RMSE nominal']:.1f}",
        "SIGef": _fmt(sig.loc[NAK, "σ efectivo máx. (µg/m³)"], 2),
        "NMC": str(R["n_mc"]), "SEMILLA": str(R["semilla"]),
        "AMdesde": _fmt(al.loc[NAK, "Amarilla desde km"], 2), "AMhasta": _fmt(al.loc[NAK, "Amarilla hasta km"], 2),
        "ROdesde": _fmt(al.loc[NAK, "Roja desde km"], 2), "ROhasta": _fmt(al.loc[NAK, "Roja hasta km"], 2),
        "DESFASEsplines": f"{max(al.loc[n, c] for n in (NAT, SUJ) for c in al.columns if 'desfase' in c):.0f}",
        "DESFASElin": f"{max(al.loc[LIN, c] for c in al.columns if 'desfase' in c):.0f}",
        "RUNGElagIni": _fmt(R["runge"]["Error máx. Lagrange"].iloc[0], 2),
        "RUNGElagFin": _fmt(R["runge"]["Error máx. Lagrange"].iloc[-1], 1),
        "RUNGEsplFin": f"{R['runge']['Error máx. spline'].iloc[-1]:.4f}",
        "RUNGEnodosFin": str(int(R["runge"]["Nodos"].iloc[-1])),
        "SPPnakIni": _fmt(float(CubicSpline(X_KM, PM25)(0.0, 2)), 1), "SPPnakFin": _fmt(float(CubicSpline(X_KM, PM25)(9.0, 2)), 1),
        "SPPsujIni": _fmt(float(construir_modelos(X_KM, PM25)[SUJ](0.0, 2)), 1),
        "SPPsujFin": _fmt(float(construir_modelos(X_KM, PM25)[SUJ](9.0, 2)), 1),
        "BANDAmax": _fmt(R["banda"]["ancho_max"], 2), "BANDAmaxkm": _fmt(R["banda"]["km_max"], 1),
        "BANDAmin": _fmt(R["banda"]["ancho_min"], 2), "BANDAminkm": _fmt(R["banda"]["km_min"], 1),
        "BANDAborde": _fmt(R["banda"]["ancho_medio_borde"], 2), "BANDAinterior": _fmt(R["banda"]["ancho_medio_interior"], 2),
        "DESFASEherPch": f"{max(al.loc[n, c] for n in (HER, PCH) for c in al.columns if 'desfase' in c):.0f}",
        "ROlinDesde": _fmt(al.loc[LIN, "Roja desde km"], 2), "ROlinHasta": _fmt(al.loc[LIN, "Roja hasta km"], 2),
        "DIFbordeNat": _fmt(fr.loc[f"{NAK} vs {NAT}", "máx |ΔS| en tramos de borde"], 2),
        "DIFintNat": _fmt(fr.loc[f"{NAK} vs {NAT}", "máx |ΔS| en el interior (km 1–8)"], 2),
    }
    with open(gen / "valores.tex", "w", encoding="utf-8") as fh:
        fh.write("% Generado automáticamente por modelado_validacion.py — no editar a mano.\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    # Tabla LOOCV (errores con signo por estación + métricas)
    cols = [NAK, NAT, SUJ, HER, PCH, LIN, LAG]
    cab = ["NaK", "Nat.", "Suj.", "Herm.", "PCHIP", "Lin.", "Lagr."]
    err = R["err"]
    lineas = [r"\begin{tabular}{@{}l r *{7}{r}@{}}", r"\toprule",
              r"Est. & $y_i$ & " + " & ".join(cab) + r" \\", r"\midrule"]
    for est, fila in err.iterrows():
        lineas.append(f"{est} & {fila['Real']:.0f} & " + " & ".join(f"${fila[c]:+.2f}$" for c in cols) + r" \\")
    lineas.append(r"\midrule")
    for metrica, nd in (("RMSE", 2), ("MAE", 2), ("Error máx.", 2)):
        valores = [res.loc[c, metrica] for c in cols]
        mejor = min(valores)
        celdas = [(rf"\textbf{{{v:.{nd}f}}}" if np.isclose(v, mejor) else f"{v:.{nd}f}") for v in valores]
        nombre = {"RMSE": "RMSE", "MAE": "MAE", "Error máx.": r"$e_{\max}$"}[metrica]
        lineas.append(rf"\multicolumn{{2}}{{@{{}}l}}{{{nombre}}} & " + " & ".join(celdas) + r" \\")
    lineas.append(r"\multicolumn{2}{@{}l}{Peor} & " + " & ".join(res.loc[c, "Estación peor"] for c in cols) + r" \\")
    lineas += [r"\bottomrule", r"\end{tabular}"]
    (gen / "tabla_loocv.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # Tabla de robustez bajo ruido
    lineas = [r"\begin{tabular}{@{}l r r r r@{}}", r"\toprule",
              r"Método & RMSE & $\overline{\mathrm{RMSE}}_{\sigma}$ & P2.5--P97.5 & Mejor (\%) \\", r"\midrule"]
    nombre_corto = {NAK: "Spline not-a-knot", NAT: "Spline natural", SUJ: "Spline sujeta", HER: "Hermite",
                    PCH: "PCHIP", LIN: "Lineal", LAG: "Lagrange"}
    for n, fila in rz.iterrows():
        lineas.append(f"{nombre_corto[n]} & {fila['RMSE nominal']:.2f} & {fila['RMSE medio (ruido)']:.2f} & "
                      f"{fila['P2.5']:.2f}--{fila['P97.5']:.2f} & {fila['% realizaciones donde es el mejor']:.0f} \\\\")
    lineas += [r"\bottomrule", r"\end{tabular}"]
    (gen / "tabla_ruido.tex").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    # Matriz del sistema natural (8×8) para el informe
    A, r = R["A_nat"], R["rhs_nat"]
    filas = [" & ".join(f"{int(v)}" if v else "0" for v in fila) for fila in A]
    cuerpo = r" \\ ".join(filas)
    rr = r" \\ ".join(f"{int(v)}" for v in r)
    (gen / "sistema_natural.tex").write_text(
        r"\begin{equation}\setlength{\arraycolsep}{2.4pt}" "\n"
        r"\begin{bmatrix}" + cuerpo + r"\end{bmatrix}" "\n"
        r"\begin{bmatrix}c_1\\c_2\\c_3\\c_4\\c_5\\c_6\\c_7\\c_8\end{bmatrix}=" "\n"
        r"\begin{bmatrix}" + rr + r"\end{bmatrix}" "\n"
        r"\label{eq:sistema}\end{equation}" "\n", encoding="utf-8")


# =============================================================================================
# 11. Punto de entrada
# =============================================================================================
def main(argv=None):
    p = argparse.ArgumentParser(description="AirSpline · Modelado y Validación (Caso 3)")
    p.add_argument("--salida", default=str(Path(__file__).resolve().parent / "docs" / "informe"),
                   help="carpeta del informe (figuras en <salida>/figuras, tablas en <salida>/generado)")
    p.add_argument("--n-mc", type=int, default=N_MC_VALIDACION, help="realizaciones de la LOOCV bajo ruido")
    p.add_argument("--semilla", type=int, default=SEMILLA)
    p.add_argument("--sin-figuras", action="store_true")
    args = p.parse_args(argv)

    R = calcular_todo(n_mc=args.n_mc, semilla=args.semilla)
    imprimir_reporte(R)
    salida = Path(args.salida)
    exportar(R, salida)
    if not args.sin_figuras:
        for ruta in generar_figuras(R, salida / "figuras"):
            print("Figura:", ruta)
    print("Tablas y macros LaTeX en:", salida / "generado")
    return R


if __name__ == "__main__":
    main()
