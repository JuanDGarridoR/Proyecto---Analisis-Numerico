# -*- coding: utf-8 -*-
"""Pruebas de verificación y validación de AirSpline (rol de Modelado y Validación).

Ejecutar desde la raíz del repositorio:   python -m pytest -q tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modelado_validacion as mv  # noqa: E402

X, Y = mv.X_KM, mv.PM25


# ------------------------------------------------------------------ formulación / verificación
def test_thomas_igual_a_solucion_directa():
    rng = np.random.default_rng(0)
    n = 12
    sub, sup = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    diag = 3 + rng.uniform(0, 1, n)
    rhs = rng.normal(size=n)
    A = mv.matriz_densa(sub, diag, sup)
    assert np.allclose(mv.thomas(sub, diag, sup, rhs), np.linalg.solve(A, rhs), atol=1e-12)


def test_sistema_natural_del_caso():
    sub, diag, sup, rhs, idx = mv.sistema_tridiagonal(X, Y, "natural")
    assert list(idx) == list(range(1, 9))
    assert np.all(diag == 4) and np.all(sub[1:] == 1) and np.all(sup[:-1] == 1)
    assert rhs.tolist() == [27, 3, -30, -33, -15, -21, 3, 27]
    assert np.linalg.cond(mv.matriz_densa(sub, diag, sup)) < 3       # bien condicionado


def test_not_a_knot_primera_y_ultima_fila_malla_uniforme():
    sub, diag, sup, _, _ = mv.sistema_tridiagonal(X, Y, "not-a-knot")
    assert diag[0] == pytest.approx(6) and sup[0] == pytest.approx(0)
    assert diag[-1] == pytest.approx(6) and sub[-1] == pytest.approx(0)


@pytest.mark.parametrize("frontera", ["natural", "sujeta", "not-a-knot"])
def test_implementacion_propia_igual_a_scipy(frontera):
    tabla = mv.verificar_contra_scipy()
    fila = tabla.set_index("Frontera").loc[frontera]
    assert fila["máx |Δ coeficientes|"] < 1e-12
    assert fila["máx |Δ S(x)|"] < 1e-12
    assert fila["Diagonal estrictamente dominante"] == "sí"


def test_periodica_no_aplica_al_corredor():
    with pytest.raises(ValueError, match="periódica no aplica"):
        mv.sistema_tridiagonal(X, Y, "periodica")


@pytest.mark.parametrize("frontera", ["natural", "not-a-knot"])
def test_interpola_y_es_C2(frontera):
    s = mv.spline_propio(X, Y, frontera)
    assert np.allclose(s(X), Y, atol=1e-12)
    saltos = mv.saltos_en_nodos(s)
    for k in ("S", "S′", "S′′"):
        assert saltos[k].max() < 1e-10


def test_not_a_knot_tiene_tercera_derivada_continua_en_x1_y_x8():
    s3 = mv.saltos_en_nodos(mv.spline_propio(X, Y, "not-a-knot"))["S′′′"]
    assert s3[0] < 1e-10 and s3[-1] < 1e-10
    s3_nat = mv.saltos_en_nodos(mv.spline_propio(X, Y, "natural"))["S′′′"]
    assert s3_nat[0] > 1 and s3_nat[-1] > 1                         # la natural sí "rompe" ahí


def test_integral_propia_igual_a_scipy():
    s = mv.spline_propio(X, Y, "not-a-knot")
    assert s.integral() == pytest.approx(CubicSpline(X, Y).integrate(0, 9), abs=1e-10)


def test_orden_de_convergencia():
    conv = mv.estudio_convergencia()
    ultimo = conv[conv["n tramos"] == 144].set_index("Método")["Orden observado"]
    assert ultimo[mv.NAK] == pytest.approx(4, abs=0.3)
    assert ultimo["Spline sujeta (f′ exacta)"] == pytest.approx(4, abs=0.3)
    assert ultimo[mv.NAT] == pytest.approx(2, abs=0.2)               # borde con S'' ≠ 0
    assert ultimo[mv.LIN] == pytest.approx(2, abs=0.2)


# ------------------------------------------------------------------ fenómeno de Runge
def test_constantes_de_lebesgue():
    leb = mv.tabla_lebesgue().set_index("Método")["Constante de Lebesgue Λ"]
    assert leb[mv.LAG] > 15
    assert leb[mv.NAK] < 2.5 and leb[mv.NAT] < 2.0
    cr = mv.crecimiento_lebesgue((10, 20, 30)).set_index("Nodos")
    assert cr.loc[20, mv.LAG] > 100 * cr.loc[10, mv.LAG]             # crecimiento exponencial
    assert abs(cr.loc[30, mv.NAK] - cr.loc[10, mv.NAK]) < 0.05         # acotada


def test_runge_lagrange_diverge_spline_converge():
    rg = mv.estudio_runge()
    assert rg["Error máx. Lagrange"].iloc[-1] > 10 * rg["Error máx. Lagrange"].iloc[0]
    assert rg["Error máx. spline"].iloc[-1] < 0.1 * rg["Error máx. spline"].iloc[0]


# ------------------------------------------------------------------ validación LOOCV
def test_loocv_reproduce_el_cuaderno():
    _, err = mv.loocv()
    res = mv.metricas(err)
    assert mv.seleccionar_frontera(res) == mv.NAK
    assert res.loc[mv.NAK, "RMSE"] == pytest.approx(1.045, abs=5e-4)
    assert res.loc[mv.NAK, "Error máx."] == pytest.approx(1.528, abs=5e-4)
    assert res.loc[mv.NAK, "Estación peor"] == "E7"
    assert res.loc[mv.NAT, "RMSE"] == pytest.approx(1.529, abs=5e-4)
    assert res.loc[mv.LIN, "RMSE"] == pytest.approx(3.787, abs=5e-4)
    assert res.loc[mv.LAG, "RMSE"] == pytest.approx(8.483, abs=5e-4)
    assert 100 * (1 - res.loc[mv.NAK, "RMSE"] / res.loc[mv.LIN, "RMSE"]) == pytest.approx(72.4, abs=0.05)


def test_loocv_lineal_coincide_con_formula_cerrada():
    _, err = mv.loocv()
    esperado = (Y[:-2] + Y[2:]) / 2 - Y[1:-1]
    assert np.allclose(err[mv.LIN].values, esperado)


@pytest.mark.parametrize("nombre", mv.MODELOS_LINEALES)
def test_matriz_loocv_equivale_al_bucle(nombre):
    _, err = mv.loocv()
    assert np.allclose(mv.matriz_loocv(nombre) @ Y, err[nombre].values, atol=1e-9)


def test_factor_de_ruido_analitico_vs_montecarlo():
    fac = mv.factores_ruido()
    assert fac.loc["E5", mv.LIN] == pytest.approx(np.sqrt(1.5))
    rng = np.random.default_rng(1)
    eps = rng.normal(0, 1.0, size=(20000, len(X)))
    e = eps @ mv.matriz_loocv(mv.NAK).T                              # error inducido sólo por ruido
    assert e[:, 0].std() == pytest.approx(fac.loc["E2", mv.NAK], rel=0.03)


# ------------------------------------------------------------------ alertas
def test_tramos_con_raiz_exacta_en_la_malla():
    """Regresión: la raíz del lineal en km 4.5 (u = 90) cae justo en un punto de la malla."""
    lin = mv.construir_modelos(X, Y)[mv.LIN]
    roja = mv.tramos_sobre_umbral(lin, 90.0)
    assert len(roja) == 1
    assert roja[0][0] == pytest.approx(3 + 5 / 7, abs=1e-3) and roja[0][1] == pytest.approx(4.5, abs=1e-9)
    amarilla = mv.tramos_sobre_umbral(lin, 71.0)
    assert amarilla[0][1] == pytest.approx(6.5, abs=1e-9)


def test_tramos_de_alerta_del_modelo_principal():
    cs = mv.construir_modelos(X, Y)[mv.NAK]
    (a0, a1), = mv.tramos_sobre_umbral(cs, mv.UMBRAL_AMARILLA)
    (r0, r1), = mv.tramos_sobre_umbral(cs, mv.UMBRAL_ROJA)
    assert (a0, a1) == pytest.approx((2.109, 6.592), abs=1e-3)
    assert (r0, r1) == pytest.approx((3.488, 4.680), abs=1e-3)


def test_alertas_insensibles_a_la_frontera_del_spline():
    al = mv.sensibilidad_alertas().set_index("Método")
    for n in (mv.NAT, mv.SUJ):
        assert al.loc[n, "Roja: máx. desfase vs not-a-knot (m)"] < 10
        assert al.loc[n, "Amarilla: máx. desfase vs not-a-knot (m)"] < 10
