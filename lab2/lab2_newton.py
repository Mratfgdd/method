import csv
import math
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results")
DATA_FILE = os.path.join(BASE_DIR, "data.csv")
X_TARGET = 6000.0
NODE_COUNTS = (5, 10, 20)
FIXED_STEP = 750.0
SUB = 20
N_SCAN = range(5, 41, 5)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_C, AXIS_C = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = {5: BLUE, 10: ORANGE, 20: AQUA}
STYLE = {5: "-", 10: "--", 20: "-."}
plt.rcParams.update({
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10.5,
    "axes.edgecolor": AXIS_C,
    "axes.labelcolor": INK2,
    "axes.titlesize": 11.5,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID_C,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
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

def header(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def read_data(filename):
    x = []
    y = []
    with open(filename, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row["n"]))
            y.append(float(row["t"]))
    return np.array(x), np.array(y)

def tabulate_function(func, a, b, count, path):
    x = np.linspace(a, b, count)
    y = func(x)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# Табуляція f(x) на [{a:g}; {b:g}]: вузлів {count}, крок h = {(b - a) / (count - 1):.6f}\n")
        fh.write("# i  x_i  f(x_i)\n")
        for i in range(count):
            fh.write(f"{i} {x[i]:.10f} {y[i]:.15e}\n")

def read_nodes(path):
    xs, fs = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            _, xi, fi = line.split()
            xs.append(float(xi))
            fs.append(float(fi))
    return np.array(xs), np.array(fs)

def omega(x_nodes, k, x):
    result = np.ones_like(np.asarray(x, dtype=float))
    for i in range(k + 1):
        result = result * (x - x_nodes[i])
    return result

def divided_difference(x, f, k):
    total = 0.0
    for i in range(k + 1):
        prod = 1.0
        for j in range(k + 1):
            if j != i:
                prod *= x[i] - x[j]
        total += f[i] / prod
    return total

def divided_difference_table(x, f):
    n = len(x)
    table = [[float(v) for v in f]]
    for k in range(1, n):
        prev = table[-1]
        table.append([(prev[i + 1] - prev[i]) / (x[i + k] - x[i]) for i in range(n - k)])
    return table

def newton_coefficients(x, f, method="explicit"):
    if method == "explicit":
        return np.array([divided_difference(x, f, k) for k in range(len(x))])
    return np.array([col[0] for col in divided_difference_table(x, f)])

def newton(x_nodes, coeffs, x):
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, coeffs[0])
    w = np.ones_like(x)
    for k in range(1, len(coeffs)):
        w = w * (x - x_nodes[k - 1])
        result = result + w * coeffs[k]
    return result

def lagrange(x_nodes, f, x):
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    for i in range(len(x_nodes)):
        li = np.ones_like(x)
        for j in range(len(x_nodes)):
            if j != i:
                li = li * (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result = result + f[i] * li
    return result

def finite_differences(f):
    table = [[float(v) for v in f]]
    for _ in range(1, len(f)):
        prev = table[-1]
        table.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
    return table

def factorial_power(t, k):
    result = np.ones_like(np.asarray(t, dtype=float))
    for j in range(k):
        result = result * (t - j)
    return result

def factorial_polynomial(x0, h, f, x):
    t = (np.asarray(x, dtype=float) - x0) / h
    d = finite_differences(f)
    return sum(d[k][0] / math.factorial(k) * factorial_power(t, k) for k in range(len(f)))

class EmpiricalModel:
    def __init__(self, n, t):
        self.n0 = n[0]
        u = np.log2(n / self.n0)
        self.c2, self.c1, self.c0 = np.polyfit(u, np.log(t), 2)
    def __call__(self, x):
        u = np.log2(np.asarray(x, dtype=float) / self.n0)
        return np.exp(self.c0 + self.c1 * u + self.c2 * u ** 2)

def runge(x):
    return 1.0 / (1.0 + 25.0 * np.asarray(x, dtype=float) ** 2)

def chebyshev_nodes(a, b, count):
    k = np.arange(count)
    return np.sort((a + b) / 2 + (b - a) / 2 * np.cos((2 * k + 1) * np.pi / (2 * count)))

def dd_table_text(x, table):
    n = len(x)
    head = f"{'i':>2} | {'x_i':>8} | " + " | ".join(f"{'порядок ' + str(k):>13}" for k in range(n))
    lines = [head, "-" * len(head)]
    for i in range(n):
        cells = [f"{table[k][i]:13.6e}" if i < n - k else " " * 13 for k in range(n)]
        lines.append(f"{i:2d} | {x[i]:8.0f} | " + " | ".join(cells))
    return "\n".join(lines)

def fd_table_text(u, table):
    n = len(u)
    head = f"{'i':>2} | {'u_i':>4} | " + " | ".join(f"{'D^' + str(k) + ' f_i':>8}" for k in range(n))
    lines = [head, "-" * len(head)]
    for i in range(n):
        cells = [f"{table[k][i]:8.2f}" if i < n - k else " " * 8 for k in range(n)]
        lines.append(f"{i:2d} | {u[i]:4.0f} | " + " | ".join(cells))
    return "\n".join(lines)

def tabulate_interpolation(x_nodes, coeffs, func, path):
    n = len(x_nodes) - 1
    xx = np.linspace(x_nodes[0], x_nodes[-1], SUB * n + 1)
    ff = func(xx)
    nn = newton(x_nodes, coeffs, xx)
    eps = np.abs(ff - nn)
    w = omega(x_nodes, n, xx)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# n = {n} ({n + 1} вузлів), крок табуляції {(xx[1] - xx[0]):.6f}\n")
        fh.write(f"# {'x':>14} {'f(x)':>22} {'N_n(x)':>22} {'eps(x)':>12} {'omega_n(x)':>14}\n")
        for i in range(len(xx)):
            fh.write(f"{xx[i]:16.6f} {ff[i]:22.15e} {nn[i]:22.15e} {eps[i]:12.4e} {w[i]:14.6e}\n")
    return xx, ff, nn, eps, w

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name))
    plt.close(fig)

def plot_data(xd, yd, coeffs, p_newton, p_fact):
    xx = np.linspace(xd[0], xd[-1], 600)
    uu = np.log2(xx / xd[0])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(xx, newton(xd, coeffs, xx), color=BLUE, label=r"многочлен Ньютона $N_4(n)$")
    ax.plot(xx, factorial_polynomial(0.0, 1.0, yd, uu), color=ORANGE, ls="--",
            label=r"факторіальний многочлен $P_4(u)$, $u = \log_2(n/1000)$")
    ax.scatter(xd, yd, s=36, color=INK, edgecolor="white", linewidth=1, zorder=4,
               label="експериментальні точки")
    ax.scatter([X_TARGET], [p_newton], s=44, marker="D", color=BLUE, edgecolor="white", zorder=5)
    ax.scatter([X_TARGET], [p_fact], s=44, marker="s", color=ORANGE, edgecolor="white", zorder=5)
    ax.annotate(f"$N_4$(6000) = {p_newton:.2f} мс\n$P_4$(6000) = {p_fact:.2f} мс",
                (X_TARGET, p_newton), textcoords="offset points", xytext=(-12, 14), ha="right",
                color=INK2, fontsize=9.5)
    ax.axvline(X_TARGET, color=AXIS_C, lw=1, ls=":")
    ax.set_xlabel("Розмір вхідних даних n")
    ax.set_ylabel("Час виконання t, мс")
    ax.set_title("Експериментальні дані та інтерполяційні криві")
    ax.legend(loc="upper left")
    save(fig, "fig1_data_interpolation.png")

def plot_model(xd, yd, model):
    xx = np.linspace(xd[0], xd[-1], 600)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(xx, model(xx), color=AQUA, label=r"модель $f(n)=\exp(c_0+c_1u+c_2u^2)$")
    ax.scatter(xd, yd, s=36, color=INK, edgecolor="white", linewidth=1, zorder=4,
               label="експериментальні точки")
    ax.set_xlabel("Розмір вхідних даних n")
    ax.set_ylabel("Час виконання t, мс")
    ax.set_title("Емпірична модель для дослідницької частини")
    ax.legend(loc="upper left")
    save(fig, "fig2_model.png")

def plot_case(N, xn, fn, xx, ff, nn, eps, w):
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.6, 1.7, 1.7]})
    ax1, ax2, ax3 = axes
    ax1.plot(xx, ff, color=INK, lw=1.2, label="f(x) — модель")
    ax1.plot(xx, nn, color=SERIES[N], ls="--", label=f"$N_{{{N - 1}}}(x)$ — многочлен Ньютона")
    ax1.scatter(xn, fn, s=22, color=SERIES[N], edgecolor="white", linewidth=0.8, zorder=3,
                label=f"вузли ({N})")
    ax1.set_ylabel("t, мс")
    ax1.set_title(f"{N} вузлів на [{xn[0]:.0f}; {xn[-1]:.0f}], крок h = {xn[1] - xn[0]:.1f}")
    ax1.legend(loc="upper left", fontsize=9.5)
    ax2.plot(xx, eps, color=SERIES[N])
    ax2.fill_between(xx, eps, color=SERIES[N], alpha=0.15, lw=0)
    ax2.set_ylabel("ε(x), мс")
    ax2.set_ylim(bottom=0)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
    ax3.plot(xx, w, color=INK2, lw=1.2)
    ax3.axhline(0, color=AXIS_C, lw=1)
    ax3.set_ylabel(f"$\\omega_{{{N - 1}}}(x)$")
    ax3.set_xlabel("n")
    fig.tight_layout()
    save(fig, f"fig3_case_{N}.png")

def plot_errors(curves, name, title, mark_target=True):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for N, (xx, eps) in curves.items():
        ax.semilogy(xx, np.maximum(eps, 1e-16), color=SERIES[N], ls=STYLE[N],
                    label=f"{N} вузлів (max {eps.max():.2e} мс)")
    if mark_target:
        ax.axvline(X_TARGET, color=MUTED, lw=1, ls=":")
        ax.annotate("n = 6000", (X_TARGET, 1), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(4, -14), color=INK2, fontsize=9)
    ax.set_ylim(bottom=1e-12)
    ax.set_xlabel("n")
    ax.set_ylabel("ε(x), мс (логарифмічна шкала)")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9.5)
    save(fig, name)

def plot_runge(xx, curves_eq, curves_ch):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, curves, title in ((axes[0], curves_eq, "Рівновіддалені вузли"),
                              (axes[1], curves_ch, "Вузли Чебишова")):
        ax.plot(xx, runge(xx), color=INK, lw=1.3, label="f(x) = 1/(1 + 25x²)")
        for N, (_, yy) in curves.items():
            ax.plot(xx, yy, color=SERIES[N], ls=STYLE[N], label=f"{N} вузлів")
        ax.set_ylim(-1.2, 2.2)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.legend(loc="upper center", fontsize=9)
    axes[0].set_ylabel("y")
    fig.tight_layout()
    save(fig, "fig6_runge.png")

def plot_scan(ns, err_table, err_explicit, err_runge_eq, err_runge_ch):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax = axes[0]
    ax.semilogy(ns, err_table, color=BLUE, marker="o", ms=5, label="рекурентна таблиця")
    ax.semilogy(ns, err_explicit, color=ORANGE, marker="s", ms=5, ls="--", label="явна формула")
    ax.set_title("Модель t(n) на [1000; 16000]")
    ax.set_xlabel("кількість вузлів")
    ax.set_ylabel("max ε, мс")
    ax.legend(loc="upper left", fontsize=9.5)
    ax = axes[1]
    ax.semilogy(ns, err_runge_eq, color=BLUE, marker="o", ms=5, label="рівновіддалені вузли")
    ax.semilogy(ns, err_runge_ch, color=ORANGE, marker="s", ms=5, ls="--", label="вузли Чебишова")
    ax.set_title("Функція Рунге на [−1; 1]")
    ax.set_xlabel("кількість вузлів")
    ax.set_ylabel("max ε")
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig7_error_vs_nodes.png")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    log = open(os.path.join(OUT_DIR, "console_output.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    header("1. Експериментальні дані (data.csv)")
    xd, yd = read_data(DATA_FILE)
    print("x:", xd.tolist())
    print("y:", yd.tolist())
    print(f"{'i':>2} | {'n':>7} | {'t, мс':>6}")
    for i in range(len(xd)):
        print(f"{i:2d} | {xd[i]:7.0f} | {yd[i]:6.1f}")
    header("2. Таблиця розділених різниць (експериментальні дані)")
    table = divided_difference_table(xd, yd)
    text = dd_table_text(xd, table)
    print(text)
    with open(os.path.join(OUT_DIR, "divided_differences.txt"), "w", encoding="utf-8") as fh:
        fh.write("Таблиця розділених різниць, варіант 1 (n -> t, мс)\n\n" + text + "\n")
    coeffs = newton_coefficients(xd, yd, "explicit")
    coeffs_tab = newton_coefficients(xd, yd, "table")
    print("\nКоефіцієнти многочлена Ньютона f(x_0, ..., x_k), явна формула:")
    for k, c in enumerate(coeffs):
        print(f"  k = {k}: {c: .10e}")
    print(f"Розбіжність з рекурентною таблицею: {np.max(np.abs(coeffs - coeffs_tab) / np.abs(coeffs_tab)):.2e} (відносна)")
    header(f"3. Прогноз часу виконання при n = {X_TARGET:.0f}")
    p_newton = float(newton(xd, coeffs, X_TARGET))
    p_lagr = float(lagrange(xd, yd, X_TARGET))
    print(f"Многочлен Ньютона N_4({X_TARGET:.0f})  = {p_newton:.6f} мс")
    print(f"Многочлен Лагранжа L_4({X_TARGET:.0f}) = {p_lagr:.6f} мс (різниця {abs(p_newton - p_lagr):.1e})")
    print("\nДодавання вузлів у порядку близькості до n = 6000 (властивість форми Ньютона):")
    order = sorted(range(len(xd)), key=lambda i: (abs(xd[i] - X_TARGET), xd[i]))
    xs, fs = xd[order], yd[order]
    value, w = 0.0, 1.0
    print(f"{'k':>2} | {'новий вузол':>11} | {'доданок':>12} | {'P_k(6000), мс':>13}")
    for k in range(len(xs)):
        term = divided_difference(xs, fs, k) * w
        value += term
        w *= X_TARGET - xs[k]
        print(f"{k + 1:2d} | {xs[k]:11.0f} | {term:12.6f} | {value:13.6f}")
    print("\nФакторіальний многочлен: вузли n_i = 1000 * 2^i рівновіддалені за u = log2(n / 1000)")
    u = np.log2(xd / xd[0])
    assert np.allclose(np.diff(u), 1.0)
    fd = finite_differences(yd)
    print(fd_table_text(u, fd))
    u_t = math.log2(X_TARGET / xd[0])
    p_fact = float(factorial_polynomial(u[0], 1.0, yd, u_t))
    p_newton_u = float(newton(u, newton_coefficients(u, yd), u_t))
    print(f"u* = log2({X_TARGET:.0f} / 1000) = {u_t:.6f}")
    print(f"Факторіальний многочлен P_4(u*) = {p_fact:.6f} мс")
    print(f"Контроль: многочлен Ньютона в змінній u дає {p_newton_u:.6f} мс (різниця {abs(p_fact - p_newton_u):.1e})")
    plot_data(xd, yd, coeffs, p_newton, p_fact)
    header("4. Емпірична модель f(n) для дослідницької частини")
    model = EmpiricalModel(xd, yd)
    print(f"ln t = c0 + c1 u + c2 u^2, u = log2(n/1000):  c0 = {model.c0:.6f}, c1 = {model.c1:.6f}, c2 = {model.c2:.6f}")
    print(f"{'n':>7} | {'t, мс':>6} | {'f(n), мс':>9} | {'відхилення, %':>13}")
    for xi, yi in zip(xd, yd):
        print(f"{xi:7.0f} | {yi:6.1f} | {float(model(xi)):9.4f} | {(float(model(xi)) / yi - 1) * 100:13.2f}")
    f_target = float(model(X_TARGET))
    print(f"f({X_TARGET:.0f}) = {f_target:.6f} мс")
    plot_model(xd, yd, model)
    a, b = xd[0], xd[-1]
    header(f"5. Інтерполяція моделі на [{a:.0f}; {b:.0f}]: 5, 10, 20 вузлів (вплив кроку h)")
    curves = {}
    for N in NODE_COUNTS:
        path = os.path.join(OUT_DIR, f"nodes_N{N}.txt")
        tabulate_function(model, a, b, N, path)
        xn, fn = read_nodes(path)
        h = xn[1] - xn[0]
        c_exp = newton_coefficients(xn, fn, "explicit")
        c_tab = newton_coefficients(xn, fn, "table")
        fd_n = finite_differences(fn)
        rel = max(abs(c_tab[k] - fd_n[k][0] / (math.factorial(k) * h ** k)) / abs(c_tab[k])
                  for k in range(N) if c_tab[k] != 0)
        xx, ff, nn, eps, w = tabulate_interpolation(xn, c_exp, model, os.path.join(OUT_DIR, f"tab_N{N}.txt"))
        curves[N] = (xx, eps)
        vn = float(newton(xn, c_exp, X_TARGET))
        vf = float(factorial_polynomial(xn[0], h, fn, X_TARGET))
        vl = float(lagrange(xn, fn, X_TARGET))
        print(f"\n--- {N} вузлів, h = {h:.4f} (файл nodes_N{N}.txt, табуляція tab_N{N}.txt)")
        print("    коефіцієнти Ньютона f(x_0..x_k):", " ".join(f"{c:.4e}" for c in c_exp))
        print(f"    f(x_0..x_k) = Delta^k f_0 / (k! h^k): max відносна розбіжність {rel:.1e}")
        print(f"    явна формула vs таблиця: max |різниця| коефіцієнтів {np.max(np.abs(c_exp - c_tab)):.1e}")
        print(f"    N({X_TARGET:.0f}) = {vn:.9f}   P({X_TARGET:.0f}) факторіальний = {vf:.9f}   "
              f"L({X_TARGET:.0f}) = {vl:.9f}")
        print(f"    f({X_TARGET:.0f}) = {f_target:.9f}   |ε({X_TARGET:.0f})| = {abs(f_target - vn):.3e}"
              f"{'  (6000 - вузол сітки)' if np.any(np.isclose(xn, X_TARGET)) else ''}")
        print(f"    max ε = {eps.max():.3e} (x = {xx[np.argmax(eps)]:.1f}),  середня ε = {eps.mean():.3e},  "
              f"max |omega_n| = {np.abs(w).max():.3e}")
        print(f"    Лагранж vs Ньютон на сітці: max |L - N| = {np.max(np.abs(lagrange(xn, fn, xx) - nn)):.1e}")
        plot_case(N, xn, fn, xx, ff, nn, eps, w)
    plot_errors(curves, "fig4_fixed_interval.png",
                f"Похибка на сталому інтервалі [{a:.0f}; {b:.0f}] (різний крок)")
    header(f"6. Сталий крок h = {FIXED_STEP:.0f}, змінний інтервал [a; a + h(N - 1)]")
    curves = {}
    for N in NODE_COUNTS:
        bN = a + FIXED_STEP * (N - 1)
        path = os.path.join(OUT_DIR, f"nodes_h{FIXED_STEP:.0f}_N{N}.txt")
        tabulate_function(model, a, bN, N, path)
        xn, fn = read_nodes(path)
        c_exp = newton_coefficients(xn, fn, "explicit")
        xx = np.linspace(xn[0], xn[-1], SUB * (N - 1) + 1)
        eps = np.abs(model(xx) - newton(xn, c_exp, xx))
        curves[N] = (xx, eps)
        e_t = abs(f_target - float(newton(xn, c_exp, X_TARGET)))
        where = "всередині" if xn[0] <= X_TARGET <= xn[-1] else "поза інтервалом (екстраполяція)"
        print(f"{N:2d} вузлів: [{xn[0]:.0f}; {xn[-1]:.0f}]  max ε = {eps.max():.3e},  середня ε = {eps.mean():.3e},  "
              f"|ε(6000)| = {e_t:.3e} - {where}")
    plot_errors(curves, "fig5_fixed_step.png", f"Похибка при сталому кроці h = {FIXED_STEP:.0f} (різний інтервал)")
    header("7. Ефект Рунге: залежність max ε від кількості вузлів")
    ns = list(N_SCAN)
    err_tab, err_exp, err_r_eq, err_r_ch = [], [], [], []
    xr = np.linspace(-1, 1, 2001)
    runge_eq, runge_ch = {}, {}
    print(f"{'N':>3} | {'модель: таблиця':>15} | {'модель: явна ф-ла':>17} | {'Рунге: рівновідд.':>17} | {'Рунге: Чебишов':>14}")
    for N in ns:
        xn = np.linspace(a, b, N)
        fn = model(xn)
        xx = np.linspace(a, b, SUB * (N - 1) + 1)
        err_tab.append(np.max(np.abs(model(xx) - newton(xn, newton_coefficients(xn, fn, "table"), xx))))
        err_exp.append(np.max(np.abs(model(xx) - newton(xn, newton_coefficients(xn, fn, "explicit"), xx))))
        xe = np.linspace(-1, 1, N)
        xc = chebyshev_nodes(-1, 1, N)
        ye = newton(xe, newton_coefficients(xe, runge(xe), "table"), xr)
        yc = newton(xc, newton_coefficients(xc, runge(xc), "table"), xr)
        err_r_eq.append(np.max(np.abs(runge(xr) - ye)))
        err_r_ch.append(np.max(np.abs(runge(xr) - yc)))
        if N in NODE_COUNTS:
            runge_eq[N] = (xe, ye)
            runge_ch[N] = (xc, yc)
        print(f"{N:3d} | {err_tab[-1]:15.3e} | {err_exp[-1]:17.3e} | {err_r_eq[-1]:17.3e} | {err_r_ch[-1]:14.3e}")
    plot_runge(xr, runge_eq, runge_ch)
    plot_scan(ns, err_tab, err_exp, err_r_eq, err_r_ch)
    print(f"\nПідсумок: t({X_TARGET:.0f}) ≈ {p_newton:.2f} мс (Ньютон), {p_fact:.2f} мс (факторіальний, log2), "
          f"{f_target:.2f} мс (модель)")
    print(f"\nГрафіки та таблиці збережено в {OUT_DIR}")
    sys.stdout = sys.__stdout__
    log.close()

if __name__ == "__main__":
    main()
