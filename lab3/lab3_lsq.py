import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results")
DATA_FILE = os.path.join(BASE_DIR, "data.csv")
NODES_FILE = os.path.join(OUT_DIR, "nodes.txt")
DEGREES = range(1, 11)
FUTURE = np.array([25.0, 26.0, 27.0])
INK, INK2, MUTED, GRID_C, AXIS_C = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10.5,
    "axes.edgecolor": AXIS_C, "axes.labelcolor": INK2, "axes.titlesize": 11.5, "axes.titlecolor": INK,
    "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID_C, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "legend.frameon": False, "lines.linewidth": 1.6,
})

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()

def read_csv(filename):
    x, f = [], []
    with open(filename, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            x.append(float(row["Month"]))
            f.append(float(row["Temp"]))
    return np.array(x), np.array(f)

def write_nodes(x, f, path):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Вузлів: {len(x)}, крок h = {(x[-1] - x[0]) / (len(x) - 1):g}\n# i x_i f_i\n")
        for i, (xi, fi) in enumerate(zip(x, f)):
            fh.write(f"{i} {xi:g} {fi:g}\n")

def read_nodes(path):
    data = np.loadtxt(path, comments="#")
    return data[:, 1], data[:, 2]

def form_matrix(x, m, rho=None):
    rho = np.ones_like(x) if rho is None else rho
    return np.array([[np.sum(rho * x ** (k + l)) for l in range(m + 1)] for k in range(m + 1)])

def form_vector(x, f, m, rho=None):
    rho = np.ones_like(x) if rho is None else rho
    return np.array([np.sum(rho * f * x ** k) for k in range(m + 1)])

def gauss_solve(A, b):
    A, b = A.astype(float).copy(), b.astype(float).copy()
    n = len(b)
    for k in range(n):
        p = k + np.argmax(np.abs(A[k:, k]))
        if p != k:
            A[[k, p]], b[[k, p]] = A[[p, k]], b[[p, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x

def polynomial(x, coef):
    result = np.zeros_like(np.asarray(x, dtype=float))
    for a in coef[::-1]:
        result = result * x + a
    return result

def dispersion(f, phi):
    return np.sqrt(np.mean((phi - f) ** 2))

def error(f, phi):
    return np.abs(f - phi)

def lsq_basis(x, f, basis):
    G = np.array([g(x) for g in basis])
    return gauss_solve(G @ G.T, G @ f)
SEASONAL = (lambda t: np.ones_like(t), lambda t: np.cos(2 * np.pi * t / 12), lambda t: np.sin(2 * np.pi * t / 12))

def seasonal_model(x, coef):
    return sum(a * g(x) for a, g in zip(coef, SEASONAL))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    log = open(os.path.join(OUT_DIR, "console_output.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    xc, fc = read_csv(DATA_FILE)
    write_nodes(xc, fc, NODES_FILE)
    x, f = read_nodes(NODES_FILE)
    n = len(x) - 1
    print(f"Вузлів: {n + 1}, відрізок [{x[0]:g}; {x[-1]:g}], крок h = {(x[-1] - x[0]) / n:g}")
    xx = np.linspace(x[0], x[-1], 20 * n + 1)
    f_lin = np.interp(xx, x, f)
    fits, disp = {}, {}
    print(f"\n{'m':>2} | {'дисперсія delta':>15} | {'max |f - phi| у вузлах':>22} | {'cond(B)':>9} | "
          f"{'|phi - polyfit|':>15} | прогноз 25, 26, 27")
    for m in DEGREES:
        B = form_matrix(x, m)
        coef = gauss_solve(B, form_vector(x, f, m))
        phi = polynomial(x, coef)
        disp[m] = dispersion(f, phi)
        ref = np.polyfit(x, f, m)[::-1]
        dev = np.max(np.abs(polynomial(x, ref) - phi))
        fits[m] = coef
        fut = polynomial(FUTURE, coef)
        print(f"{m:2d} | {disp[m]:15.4f} | {np.max(error(f, phi)):22.4f} | {np.linalg.cond(B):9.1e} | "
              f"{dev:15.2e} | " + ", ".join(f"{v:.1f}" for v in fut))
    m_opt = min(disp, key=disp.get)
    coef = fits[m_opt]
    print(f"\nОптимальний степінь (мінімум дисперсії): m = {m_opt}, delta = {disp[m_opt]:.4f}")
    print("Коефіцієнти a_j:", ", ".join(f"a{j} = {a:.6e}" for j, a in enumerate(coef)))
    fut = polynomial(FUTURE, coef)
    print("Прогноз на місяці 25, 26, 27:", ", ".join(f"{v:.2f}" for v in fut), "°C")
    same_months = (f[0:3] + f[12:15]) / 2
    print("Середнє тих самих місяців двох попередніх років:", ", ".join(f"{v:.1f}" for v in same_months), "°C")
    s_coef = lsq_basis(x, f, SEASONAL)
    s_fut = seasonal_model(FUTURE, s_coef)
    print(f"\nСезонна модель МНК: phi(x) = {s_coef[0]:.3f} {s_coef[1]:+.3f} cos(2 pi x/12) "
          f"{s_coef[2]:+.3f} sin(2 pi x/12), дисперсія {dispersion(f, seasonal_model(x, s_coef)):.4f}")
    print("Прогноз сезонної моделі на місяці 25, 26, 27:", ", ".join(f"{v:.2f}" for v in s_fut), "°C")
    with open(os.path.join(OUT_DIR, "error_tabulation.txt"), "w", encoding="utf-8") as fh:
        fh.write("# x  " + "  ".join(f"eps_m{m}" for m in DEGREES) + "\n")
        errs = {m: error(f_lin, polynomial(xx, fits[m])) for m in DEGREES}
        for i in range(len(xx)):
            fh.write(f"{xx[i]:7.3f} " + " ".join(f"{errs[m][i]:9.4f}" for m in DEGREES) + "\n")
    xp = np.linspace(x[0], FUTURE[-1], 600)
    inside = xp <= x[-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.axvspan(x[-1], FUTURE[-1] + 0.5, color=GRID_C, alpha=0.5, lw=0)
    ax.scatter(x, f, s=30, color=INK, edgecolor="white", linewidth=1, zorder=4, label="фактичні дані")
    for m, col, ls in ((2, AQUA, ":"), (6, YELLOW, "-.")):
        ax.plot(xp[inside], polynomial(xp[inside], fits[m]), color=col, ls=ls, label=f"многочлен m = {m}")
    ax.plot(xp, polynomial(xp, fits[m_opt]), color=BLUE, label=f"многочлен m = {m_opt} (мінімум δ)")
    ax.plot(xp, seasonal_model(xp, s_coef), color=ORANGE, ls="--", label="сезонна модель МНК")
    ax.scatter(FUTURE, s_fut, s=40, marker="D", color=ORANGE, edgecolor="white", zorder=5)
    ax.annotate(f"прогноз m = {m_opt}: " + ", ".join(f"{v:.0f}" for v in fut) + " °C (за межами графіка)",
                (FUTURE[0], 33), ha="center", va="top", color=INK2, fontsize=9)
    ax.set_ylim(-15, 35)
    ax.set_xlabel("Місяць (затінено — прогноз на 25–27 місяці)")
    ax.set_ylabel("Температура, °C")
    ax.set_title("Апроксимація температури методом найменших квадратів")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9)
    fig.savefig(os.path.join(OUT_DIR, "fig1_approximation.png"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ms = list(DEGREES)
    ax.plot(ms, [disp[m] for m in ms], color=BLUE, marker="o", ms=6)
    ax.scatter([m_opt], [disp[m_opt]], s=90, facecolor="none", edgecolor=ORANGE, linewidth=2, zorder=5)
    ax.annotate(f"мінімум: m = {m_opt}", (m_opt, disp[m_opt]), textcoords="offset points", xytext=(0, 12),
                ha="center", color=INK2, fontsize=9.5)
    ax.set_xticks(ms)
    ax.set_xlabel("Степінь многочлена m")
    ax.set_ylabel("Дисперсія δ, °C")
    ax.set_title("Залежність дисперсії від степеня многочлена")
    fig.savefig(os.path.join(OUT_DIR, "fig2_dispersion.png"))
    plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, ms):
        ax.plot(xx, errs[m], color=ORANGE if m == m_opt else BLUE, lw=1.2)
        ax.set_title(f"m = {m}", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("Місяць")
    for ax in axes[:, 0]:
        ax.set_ylabel("ε(x), °C")
    fig.suptitle("Похибка апроксимації ε(x) = |f(x) − φ(x)|", color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_errors.png"))
    plt.close(fig)
    print(f"\nГрафіки та таблиці збережено в {OUT_DIR}")
    sys.stdout = sys.__stdout__
    log.close()

if __name__ == "__main__":
    main()
