# -*- coding: utf-8 -*-
"""
AirSpline — aplicación de escritorio (ventana interactiva de matplotlib).

Ejecutar en VS Code:  python airspline_app.py
Controles: método/frontera, umbrales de alerta, σ del ruido, N de Monte Carlo,
kilómetro consultado (sensor virtual) y edición de las 10 lecturas.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, TextBox, Button, CheckButtons
from scipy.interpolate import CubicSpline, CubicHermiteSpline, PchipInterpolator, BarycentricInterpolator

# ----------------------------------------------------------------------------- datos
X = np.arange(0.0, 10.0, 1.0)
Y0 = np.array([45, 52, 68, 85, 92, 88, 79, 63, 48, 42], dtype=float)
XF = np.round(np.linspace(0, 9, 91), 1)
L = 9.0
COL = {"spline": "#2563eb", "amarilla": "#facc15", "roja": "#ef4444", "banda": "#93c5fd", "dato": "#1f2937"}


# ----------------------------------------------------------------------------- núcleo numérico
def ajustar(metodo, xs, ys):
    g = np.gradient(ys, xs)
    if metodo == "Spline not-a-knot":
        return CubicSpline(xs, ys, bc_type="not-a-knot")
    if metodo == "Spline natural":
        return CubicSpline(xs, ys, bc_type="natural")
    if metodo == "Spline sujeta":
        return CubicSpline(xs, ys, bc_type=((1, g[0]), (1, g[-1])))
    if metodo == "Hermite":
        return CubicHermiteSpline(xs, ys, g)
    if metodo == "PCHIP":
        return PchipInterpolator(xs, ys)
    if metodo == "Lineal":
        return lambda t: np.interp(t, xs, ys)
    return BarycentricInterpolator(xs, ys)          # Lagrange


def tramos(f, u, a=0.0, b=9.0):
    """Intervalos donde f(x) >= u (raíces por interpolación lineal en malla densa)."""
    xx = np.linspace(a, b, 9001)
    g = f(xx) - u
    idx = np.where(g[:-1] * g[1:] < 0)[0]                        # cambio de signo estricto
    r = np.concatenate([xx[g == 0],                               # raíces que caen justo en la malla
                        xx[idx] - g[idx] * (xx[idx + 1] - xx[idx]) / (g[idx + 1] - g[idx])])
    pts = np.unique(np.concatenate([[a], r, [b]]))
    return [(p, q) for p, q in zip(pts[:-1], pts[1:]) if f(0.5 * (p + q)) >= u]


def loocv_rmse(metodo, ys):
    e = []
    for i in range(1, len(X) - 1):
        m = np.ones(len(X), bool); m[i] = False
        e.append(float(ajustar(metodo, X[m], ys[m])(X[i])) - ys[i])
    e = np.abs(e)
    return np.sqrt(np.mean(e ** 2)), e.max()


def integral(f):
    return float(f.integrate(0, 9)) if hasattr(f, "integrate") else float(np.trapezoid(f(XF), XF))


# ----------------------------------------------------------------------------- interfaz
METODOS = ["Spline not-a-knot", "Spline natural", "Spline sujeta", "Hermite", "PCHIP", "Lineal", "Lagrange"]
st = {"metodo": METODOS[0], "y": Y0.copy(), "banda": True}

fig = plt.figure("AirSpline — Air Health Analytics", figsize=(15, 8.5))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.06, 0.40, 0.62, 0.50])
axv = fig.add_axes([0.06, 0.33, 0.62, 0.035])
axi = fig.add_axes([0.72, 0.40, 0.26, 0.50]); axi.axis("off")
fig.text(0.06, 0.95, "AirSpline", fontsize=22, weight="bold", color="#1e3a8a")
fig.text(0.17, 0.955, "sensores virtuales de PM₂.₅ con splines cúbicos C² · Monte Carlo σ-ruido",
         fontsize=11, color="#334155")

s_ua = Slider(fig.add_axes([0.10, 0.24, 0.22, 0.025]), "Umbral amarilla", 30, 100, valinit=70, valstep=1)
s_ur = Slider(fig.add_axes([0.10, 0.20, 0.22, 0.025]), "Umbral roja", 40, 120, valinit=90, valstep=1)
s_sig = Slider(fig.add_axes([0.10, 0.16, 0.22, 0.025]), "σ ruido", 0.0, 10.0, valinit=2.5, valstep=0.5)
s_n = Slider(fig.add_axes([0.10, 0.12, 0.22, 0.025]), "N Monte Carlo", 20, 500, valinit=100, valstep=10)
s_km = Slider(fig.add_axes([0.10, 0.08, 0.22, 0.025]), "Km consulta", 0, 9, valinit=4.0, valstep=0.1)
r_met = RadioButtons(fig.add_axes([0.40, 0.02, 0.13, 0.22]), METODOS, active=0)
c_band = CheckButtons(fig.add_axes([0.55, 0.22, 0.12, 0.05]), ["Banda 95 %"], [True])
b_reset = Button(fig.add_axes([0.55, 0.15, 0.12, 0.045]), "Restaurar datos")
fig.text(0.72, 0.30, "Lecturas de estaciones (µg/m³) — editar y Enter:", fontsize=9, weight="bold")
cajas = []
for i in range(10):
    fila, col = divmod(i, 5)
    tb = TextBox(fig.add_axes([0.745 + col * 0.05, 0.24 - fila * 0.05, 0.035, 0.035]), f"E{i+1}",
                 initial=f"{Y0[i]:.0f}", label_pad=0.05)
    cajas.append(tb)


def redibujar(_=None):
    ys, ua, ur = st["y"], s_ua.val, s_ur.val
    f = ajustar(st["metodo"], X, ys)
    c = f(XF)
    rng = np.random.default_rng(2024)
    curvas = np.array([ajustar(st["metodo"], X, ys + rng.normal(0, s_sig.val, 10))(XF) for _ in range(int(s_n.val))])
    lo, hi = np.percentile(curvas, [2.5, 97.5], axis=0)

    ax.clear()
    if st["banda"] and s_sig.val > 0:
        ax.fill_between(XF, lo, hi, color=COL["banda"], alpha=0.5, label="Banda 95 % MC")
    ax.plot(XF, c, color=COL["spline"], lw=2.5, label=st["metodo"])
    ax.fill_between(XF, ua, c, where=c >= ua, color=COL["amarilla"], alpha=0.45, interpolate=True, label="Alerta amarilla")
    ax.fill_between(XF, ur, c, where=c >= ur, color=COL["roja"], alpha=0.6, interpolate=True, label="Alerta roja")
    ax.axhline(ua, color="#ca8a04", ls="--", lw=1); ax.axhline(ur, color="#b91c1c", ls="--", lw=1)
    ax.plot(X, ys, "o", color=COL["dato"], ms=7, label="Estaciones")
    ax.axvline(s_km.val, color="k", ls=":", lw=1)
    ax.set_xlim(-0.1, 9.1); ax.set_ylabel("PM₂.₅ (µg/m³)"); ax.grid(alpha=0.3)
    ax.set_title("Mapa de alertas del corredor (0–9 km, malla 0.1 km)")
    ax.legend(fontsize=8, loc="upper right")

    axv.clear()
    colores = np.where(c >= ur, COL["roja"], np.where(c >= ua, COL["amarilla"], "#22c55e"))
    axv.bar(XF, 1, width=0.1, color=colores); axv.set_xlim(-0.1, 9.1)
    axv.set_yticks([]); axv.set_xlabel("Distancia en el corredor (km)")

    i = int(round(s_km.val / 0.1)); ck = float(f(s_km.val))
    nivel = "ALERTA ROJA" if ck >= ur else "ALERTA AMARILLA" if ck >= ua else "NORMAL"
    colk = COL["roja"] if ck >= ur else "#ca8a04" if ck >= ua else "#16a34a"
    rmse, emax = loocv_rmse(st["metodo"], ys)
    carga = integral(f)
    fmt = lambda t: "\n   ".join(f"km {p:.2f} – {q:.2f}" for p, q in t) or "ninguno"
    conf, prec = XF[lo >= ur], XF[hi >= ur]
    axi.clear(); axi.axis("off")
    axi.text(0, 1.0, f"Sensor {'físico' if abs(s_km.val - round(s_km.val)) < 1e-9 else 'virtual'} · km {s_km.val:.1f}",
             fontsize=11, color="#475569", va="top")
    axi.text(0, 0.93, f"{ck:.1f} µg/m³", fontsize=26, weight="bold", color=colk, va="top")
    axi.text(0, 0.80, nivel, fontsize=14, weight="bold", color=colk, va="top")
    axi.text(0, 0.73, f"IC 95 %: [{lo[i]:.1f}, {hi[i]:.1f}]   P(≥{ur:.0f}) = {100*(curvas[:, i] >= ur).mean():.0f} %",
             fontsize=10, va="top")
    axi.text(0, 0.64,
             f"Carga total ∫S dx = {carga:.1f} µg·km/m³\n"
             f"Media corredor = {carga/L:.2f} µg/m³\n"
             f"LOOCV: RMSE = {rmse:.2f} · e_max = {emax:.2f} µg/m³\n\n"
             f"Amarilla o superior:\n   {fmt(tramos(f, ua))}\n"
             f"Roja:\n   {fmt(tramos(f, ur))}\n"
             f"Roja confirmada (P2.5≥{ur:.0f}): {'km %.1f–%.1f' % (conf.min(), conf.max()) if len(conf) else '—'}\n"
             f"Roja precautoria (P97.5≥{ur:.0f}): {'km %.1f–%.1f' % (prec.min(), prec.max()) if len(prec) else '—'}",
             fontsize=9.5, va="top", family="monospace")
    fig.canvas.draw_idle()


def cambiar_metodo(m):
    st["metodo"] = m; redibujar()


def cambiar_banda(_):
    st["banda"] = not st["banda"]; redibujar()


def editar(i):
    def _f(texto):
        try:
            st["y"][i] = float(texto.replace(",", "."))
        except ValueError:
            cajas[i].set_val(f"{st['y'][i]:.0f}")
        redibujar()
    return _f


def restaurar(_):
    st["y"] = Y0.copy()
    for i, tb in enumerate(cajas):
        tb.eventson = False; tb.set_val(f"{Y0[i]:.0f}"); tb.eventson = True
    redibujar()


for s in (s_ua, s_ur, s_sig, s_n, s_km):
    s.on_changed(redibujar)
r_met.on_clicked(cambiar_metodo)
c_band.on_clicked(cambiar_banda)
b_reset.on_clicked(restaurar)
for i, tb in enumerate(cajas):
    tb.on_submit(editar(i))

redibujar()

if __name__ == "__main__":
    if matplotlib.get_backend().lower() == "agg":
        fig.savefig("airspline_app_captura.png", dpi=110)
        print("Backend sin ventana: captura guardada en airspline_app_captura.png")
    else:
        plt.show()
