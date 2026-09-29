import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "results")
CACHE_FILE = os.path.join(BASE_DIR, "elevation_cache.json")
URL = ("https://api.open-elevation.com/api/v1/lookup?locations="
       "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
       "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
       "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
       "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
       "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
       "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
       "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106")
NODE_COUNTS = (10, 15, 20)
N_GRID = 3001
STEEP = 15.0
MASS, G = 80.0, 9.81
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID_C, AXIS_C = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SERIES = {10: "#2a78d6", 15: "#eb6834", 20: "#1baf7a"}
STYLE = {10: "-", 15: "--", 20: "-."}
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

def fetch_elevations():
    try:
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        data = response.json()
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("Дані отримано з Open-Elevation API")
    except (requests.RequestException, ValueError) as e:
        print(f"API недоступний ({e}); використовую збережені дані {CACHE_FILE}")
        with open(CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
    return data["results"]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def tabulate(results):
    coords = [(p["latitude"], p["longitude"]) for p in results]
    elevations = np.array([p["elevation"] for p in results], dtype=float)
    distances = [0.0]
    for i in range(1, len(results)):
        d = haversine(*coords[i - 1], *coords[i])
        distances.append(distances[-1] + d)
    return coords, np.array(distances), elevations

def write_tabulation(coords, distances, elevations, path):
    n = len(coords)
    lines = [
        "Табуляція вузлів маршруту Заросляк -> Говерла (Open-Elevation API)",
        f"Кількість вузлів: {n}",
        "",
        " №  |  Latitude  |  Longitude | Elevation (m) | Distance (m)",
        "-" * 62,
    ]
    for i in range(n):
        lines.append(f"{i:3d} | {coords[i][0]:.6f} | {coords[i][1]:.6f} | "
                     f"{elevations[i]:13.2f} | {distances[i]:12.2f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def build_tridiagonal_system(x, y):
    n = len(x) - 1
    h = np.diff(x)
    alpha, beta = np.zeros(n), np.zeros(n)
    gamma, delta = np.zeros(n), np.zeros(n)
    beta[0] = 1.0
    for i in range(2, n + 1):
        k = i - 1
        h_prev, h_cur = h[i - 2], h[i - 1]
        alpha[k] = h_prev
        beta[k] = 2 * (h_prev + h_cur)
        gamma[k] = h_cur if i < n else 0.0
        delta[k] = 3 * ((y[i] - y[i - 1]) / h_cur - (y[i - 1] - y[i - 2]) / h_prev)
    return alpha, beta, gamma, delta

def tridiagonal_solve(alpha, beta, gamma, delta):
    m = len(beta)
    A, B = np.zeros(m - 1), np.zeros(m - 1)
    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]
    for i in range(1, m - 1):
        denom = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denom
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denom
    x = np.zeros(m)
    x[m - 1] = (delta[m - 1] - alpha[m - 1] * B[m - 2]) / (alpha[m - 1] * A[m - 2] + beta[m - 1])
    for i in range(m - 2, -1, -1):
        x[i] = A[i] * x[i + 1] + B[i]
    return x, A, B

def diagonal_dominance(alpha, beta, gamma):
    lhs, rhs = np.abs(beta), np.abs(alpha) + np.abs(gamma)
    return bool(np.all(lhs >= rhs) and np.any(lhs > rhs))

def system_residual(alpha, beta, gamma, delta, c):
    lhs = beta * c
    lhs[1:] += alpha[1:] * c[:-1]
    lhs[:-1] += gamma[:-1] * c[1:]
    return lhs - delta

def spline_coefficients(x, y, c):
    n = len(x) - 1
    h = np.diff(x)
    a = y[:-1].copy()
    b, d = np.zeros(n), np.zeros(n)
    for k in range(n):
        if k < n - 1:
            b[k] = (y[k + 1] - y[k]) / h[k] - h[k] / 3 * (c[k + 1] + 2 * c[k])
            d[k] = (c[k + 1] - c[k]) / (3 * h[k])
        else:
            b[k] = (y[k + 1] - y[k]) / h[k] - 2 / 3 * h[k] * c[k]
            d[k] = -c[k] / (3 * h[k])
    return a, b, d

class NaturalCubicSpline:
    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.h = np.diff(self.x)
        self.alpha, self.beta, self.gamma, self.delta = build_tridiagonal_system(self.x, self.y)
        self.c, self.A, self.B = tridiagonal_solve(self.alpha, self.beta, self.gamma, self.delta)
        self.a, self.b, self.d = spline_coefficients(self.x, self.y, self.c)
    def _locate(self, xx):
        xx = np.asarray(xx, dtype=float)
        k = np.clip(np.searchsorted(self.x, xx, side="right") - 1, 0, len(self.a) - 1)
        return k, xx - self.x[k]
    def __call__(self, xx):
        k, t = self._locate(xx)
        return self.a[k] + self.b[k] * t + self.c[k] * t ** 2 + self.d[k] * t ** 3
    def derivative(self, xx, order=1):
        k, t = self._locate(xx)
        if order == 1:
            return self.b[k] + 2 * self.c[k] * t + 3 * self.d[k] * t ** 2
        return 2 * self.c[k] + 6 * self.d[k] * t

def print_system(sp):
    print(f"{'i':>3} | {'h_i':>10} | {'alpha_i':>10} | {'beta_i':>10} | {'gamma_i':>10} | {'delta_i':>12}")
    print("-" * 70)
    for k in range(len(sp.beta)):
        print(f"{k + 1:3d} | {sp.h[k]:10.4f} | {sp.alpha[k]:10.4f} | {sp.beta[k]:10.4f} | "
              f"{sp.gamma[k]:10.4f} | {sp.delta[k]:12.8f}")

def print_sweep(sp):
    print(f"{'i':>3} | {'A_i':>14} | {'B_i':>14} | {'c_i':>14}")
    print("-" * 54)
    for k in range(len(sp.c)):
        A = f"{sp.A[k]:14.8f}" if k < len(sp.A) else f"{'-':>14}"
        B = f"{sp.B[k]:14.6e}" if k < len(sp.B) else f"{'-':>14}"
        print(f"{k + 1:3d} | {A} | {B} | {sp.c[k]:14.6e}")

def print_coefficients(sp):
    print(f"{'i':>3} | {'[x_(i-1), x_i], м':>21} | {'a_i':>10} | {'b_i':>11} | {'c_i':>13} | {'d_i':>13}")
    print("-" * 86)
    for k in range(len(sp.a)):
        seg = f"[{sp.x[k]:.1f}; {sp.x[k + 1]:.1f}]"
        print(f"{k + 1:3d} | {seg:>21} | {sp.a[k]:10.4f} | {sp.b[k]:11.6f} | "
              f"{sp.c[k]:13.6e} | {sp.d[k]:13.6e}")

def check_spline(sp):
    k = np.arange(len(sp.a))
    t = sp.h
    s_right = sp.a + sp.b * t + sp.c * t ** 2 + sp.d * t ** 3
    d1_right = sp.b + 2 * sp.c * t + 3 * sp.d * t ** 2
    d2_right = 2 * sp.c + 6 * sp.d * t
    print(f"max |S_i(x_(i-1)) - y_(i-1)|      = {np.max(np.abs(sp.a - sp.y[:-1])):.3e} м")
    print(f"max |S_i(x_i) - y_i|              = {np.max(np.abs(s_right - sp.y[1:])):.3e} м")
    print(f"max |S'_i(x_i) - S'_(i+1)(x_i)|   = {np.max(np.abs(d1_right[:-1] - sp.b[1:])):.3e}")
    print(f"max |S''_i(x_i) - S''_(i+1)(x_i)| = {np.max(np.abs(d2_right[:-1] - 2 * sp.c[1:])):.3e}")
    print(f"S''(x_0) = {2 * sp.c[0]:.3e},  S''(x_n) = {d2_right[k[-1]]:.3e}")

def check_full_system(sp):
    n, h = len(sp.a), sp.h
    M, r = np.zeros((4 * n, 4 * n)), np.zeros(4 * n)
    row = 0
    for i in range(n):
        M[row, 4 * i] = 1
        r[row] = sp.y[i]
        M[row + 1, 4 * i:4 * i + 4] = [1, h[i], h[i] ** 2, h[i] ** 3]
        r[row + 1] = sp.y[i + 1]
        row += 2
    for i in range(n - 1):
        M[row, 4 * i + 1:4 * i + 4] = [1, 2 * h[i], 3 * h[i] ** 2]
        M[row, 4 * (i + 1) + 1] = -1
        M[row + 1, 4 * i + 2:4 * i + 4] = [2, 6 * h[i]]
        M[row + 1, 4 * (i + 1) + 2] = -2
        row += 2
    M[row, 2] = 2
    M[row + 1, 4 * (n - 1) + 2:4 * n] = [2, 6 * h[-1]]
    s = np.linalg.solve(M, r).reshape(n, 4)
    diff = max(np.max(np.abs(s[:, j] - v)) for j, v in enumerate((sp.a, sp.b, sp.c, sp.d)))
    print(f"Повна система {4 * n} рівнянь (numpy.linalg.solve): "
          f"max різниця коефіцієнтів з прогонкою = {diff:.3e}")

def select_nodes(n_points, k):
    return np.unique(np.round(np.linspace(0, n_points - 1, k)).astype(int))

def runs(mask):
    segs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if start is not None and (not m or i == len(mask) - 1):
            segs.append((start, i if m else i - 1))
            start = None
    return segs

def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name))
    plt.close(fig)

def plot_nodes(x, y):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(x, y, color=AXIS_C, lw=1, zorder=1)
    ax.scatter(x, y, s=34, color=BLUE, edgecolor="white", linewidth=1, zorder=3,
               label=r"вузли $(x_i,\ y_i)$")
    ax.annotate("Заросляк", (x[0], y[0]), textcoords="offset points", xytext=(8, -6),
                ha="left", va="top", color=INK2, fontsize=9.5)
    ax.annotate("Говерла", (x[-1], y[-1]), textcoords="offset points", xytext=(-8, 6),
                ha="right", va="bottom", color=INK2, fontsize=9.5)
    ax.margins(y=0.1)
    ax.set_xlabel("Кумулятивна відстань, м")
    ax.set_ylabel("Висота, м")
    ax.set_title(f"Дискретний профіль висоти маршруту ({len(x)} вузол)")
    ax.legend(loc="upper left")
    save(fig, "fig1_nodes.png")

def plot_full_spline(sp, xx, yy):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(xx, yy, color=BLUE, label="кубічний сплайн S(x)")
    ax.scatter(sp.x, sp.y, s=30, color=INK, edgecolor="white", linewidth=1, zorder=3,
               label="вузли інтерполяції")
    ax.set_xlabel("Кумулятивна відстань, м")
    ax.set_ylabel("Висота, м")
    ax.set_title(f"Гладкий профіль висоти: вільний кубічний сплайн ({len(sp.x)} вузол)")
    ax.legend(loc="upper left")
    save(fig, "fig2_spline_full.png")

def plot_subsets(x, y, xx, yy_ref, subsets):
    fig, axes = plt.subplots(len(subsets), 1, figsize=(8, 9), sharex=True)
    for ax, (k, (idx, _, yy_k)) in zip(axes, subsets.items()):
        rest = np.setdiff1d(np.arange(len(x)), idx)
        ax.plot(xx, yy_ref, color=MUTED, lw=1.2, ls=":", label=f"сплайн за {len(x)} вузлами")
        ax.plot(xx, yy_k, color=SERIES[k], ls=STYLE[k], label=f"сплайн за {k} вузлами")
        ax.scatter(x[idx], y[idx], s=28, color=SERIES[k], edgecolor="white", linewidth=1,
                   zorder=3, label="використані вузли")
        if len(rest):
            ax.scatter(x[rest], y[rest], s=30, facecolor="white", edgecolor=INK2, linewidth=1.1,
                       zorder=3, label="відкинуті вузли")
        ax.set_title(f"{k} вузлів")
        ax.set_ylabel("Висота, м")
        ax.legend(loc="upper left", fontsize=9)
    axes[-1].set_xlabel("Кумулятивна відстань, м")
    fig.tight_layout()
    save(fig, "fig3_splines_10_15_20.png")

def plot_error_k(k, x, y, idx, xx, yy_ref, yy_k):
    err = np.abs(yy_ref - yy_k)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(xx, yy_ref, color=INK, lw=1.2, label=f"y = f(x) (сплайн за {len(x)} вузлами)")
    ax1.plot(xx, yy_k, color=SERIES[k], ls="--", label=f"y_набл (сплайн за {k} вузлами)")
    ax1.scatter(x[idx], y[idx], s=26, color=SERIES[k], edgecolor="white", linewidth=1, zorder=3)
    ax1.set_ylabel("Висота, м")
    ax1.set_title(f"Функція, наближення та похибка: {k} вузлів")
    ax1.legend(loc="upper left")
    ax2.plot(xx, err, color=SERIES[k])
    ax2.fill_between(xx, err, color=SERIES[k], alpha=0.15, lw=0)
    i_max = np.argmax(err)
    right = xx[i_max] > xx[0] + 0.7 * (xx[-1] - xx[0])
    ax2.annotate(f"max ε = {err[i_max]:.2f} м", (xx[i_max], err[i_max]),
                 textcoords="offset points", xytext=(-10, -4) if right else (10, -4),
                 ha="right" if right else "left", color=INK2, fontsize=9.5)
    ax2.set_ylim(0, err.max() * 1.25 if err.max() > 0 else 1)
    ax2.set_ylabel("ε = |y − y_набл|, м")
    ax2.set_xlabel("Кумулятивна відстань, м")
    fig.tight_layout()
    save(fig, f"fig4_error_{k}.png")

def plot_error_compare(xx, errors):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k, err in errors.items():
        ax.plot(xx, err, color=SERIES[k], ls=STYLE[k], label=f"{k} вузлів (max {err.max():.1f} м)")
    ax.set_xlabel("Кумулятивна відстань, м")
    ax.set_ylabel("ε = |y − y_набл|, м")
    ax.set_title("Похибка наближення залежно від кількості вузлів")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    save(fig, "fig5_error_compare.png")

def plot_gradient(xx, grad, segments):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for s, e in segments:
        ax.axvspan(xx[s], xx[e], color=ORANGE, alpha=0.12, lw=0)
    ax.plot(xx, grad, color=BLUE, label="нахил S'(x)·100 %")
    ax.axhline(STEEP, color=ORANGE, ls="--", lw=1.2, label=f"поріг {STEEP:.0f} %")
    ax.axhline(0, color=AXIS_C, lw=1)
    ax.set_xlabel("Кумулятивна відстань, м")
    ax.set_ylabel("Градієнт, %")
    ax.set_title(f"Крутизна маршруту (затінено ділянки > {STEEP:.0f} %)")
    ax.legend(loc="upper left")
    save(fig, "fig6_gradient.png")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    log = open(os.path.join(OUT_DIR, "console_output.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    header("1-2. Запит до Open-Elevation API")
    results = fetch_elevations()
    n_pts = len(results)
    print("Кількість вузлів:", n_pts)
    header("3. Табуляція вузлів (записано у results/tabulation.txt)")
    coords, x, y = tabulate(results)
    print(f"{'№':>3} | {'Latitude':>10} | {'Longitude':>10} | {'Elevation (m)':>13}")
    for i, p in enumerate(results):
        print(f"{i:3d} | {p['latitude']:10.6f} | {p['longitude']:10.6f} | {p['elevation']:13.2f}")
    write_tabulation(coords, x, y, os.path.join(OUT_DIR, "tabulation.txt"))
    header("4. Табуляція (відстань, висота) - вузли {x_i, y_i}")
    print(f"{'№':>3} | {'Distance (m)':>12} | {'Elevation (m)':>13}")
    for i in range(n_pts):
        print(f"{i:3d} | {x[i]:12.2f} | {y[i]:13.2f}")
    plot_nodes(x, y)
    sp = NaturalCubicSpline(x, y)
    n = len(sp.a)
    header(f"6. Коефіцієнти трьохдіагональної системи для c_i (n = {n} інтервалів)")
    print_system(sp)
    print(f"\nДіагональна перевага |beta_i| >= |alpha_i| + |gamma_i|: "
          f"{'виконується' if diagonal_dominance(sp.alpha, sp.beta, sp.gamma) else 'НЕ виконується'}")
    header("7. Розв'язок системи методом прогонки")
    print_sweep(sp)
    res = system_residual(sp.alpha, sp.beta, sp.gamma, sp.delta, sp.c)
    print(f"\nПеревірка: max |L(c) - delta| = {np.max(np.abs(res)):.3e}")
    header("8. Коефіцієнти c_i")
    for k in range(n):
        print(f"c_{k + 1:<2d} = {sp.c[k]: .8e}")
    header("9. Коефіцієнти кубічних сплайнів a_i, b_i, c_i, d_i")
    print_coefficients(sp)
    print("\nПеревірка умов інтерполяції та гладкості:")
    check_spline(sp)
    check_full_system(sp)
    xx = np.linspace(x[0], x[-1], N_GRID)
    yy_full = sp(xx)
    plot_full_spline(sp, xx, yy_full)
    header("10-12. Сплайни за 10, 15, 20 вузлами; f(x) - сплайн за всіма вузлами")
    subsets, errors, stats = {}, {}, {}
    for k in NODE_COUNTS:
        idx = select_nodes(n_pts, k)
        sp_k = NaturalCubicSpline(x[idx], y[idx])
        yy_k = sp_k(xx)
        subsets[k] = (idx, sp_k, yy_k)
        errors[k] = np.abs(yy_full - yy_k)
        rest = np.setdiff1d(np.arange(n_pts), idx)
        hold = np.abs(y[rest] - sp_k(x[rest]))
        stats[k] = (errors[k].max(), errors[k].mean(), np.sqrt(np.mean(errors[k] ** 2)),
                    hold.max() if len(hold) else 0.0, hold.mean() if len(hold) else 0.0)
        print(f"\n--- {k} вузлів: номери вузлів {idx.tolist()}")
        print(f"    відкинуті вузли: {rest.tolist()}")
        print(f"    прогонка: max |L(c) - delta| = "
              f"{np.max(np.abs(system_residual(sp_k.alpha, sp_k.beta, sp_k.gamma, sp_k.delta, sp_k.c))):.3e}")
        print_coefficients(sp_k)
        plot_error_k(k, x, y, idx, xx, yy_full, yy_k)
    plot_subsets(x, y, xx, yy_full, subsets)
    plot_error_compare(xx, errors)
    header("11. Вплив кількості вузлів на точність")
    print("ε(x) = |f(x) - y_набл(x)| на сітці з", N_GRID, "точок; "
          "похибка у відкинутих вузлах - порівняно з виміряною висотою")
    print(f"{'k':>3} | {'max ε, м':>9} | {'сер. ε, м':>9} | {'СКВ ε, м':>9} | "
          f"{'max у відк. вузлах':>18} | {'сер. у відк. вузлах':>19}")
    print("-" * 82)
    for k, (emax, emean, erms, hmax, hmean) in stats.items():
        print(f"{k:3d} | {emax:9.3f} | {emean:9.3f} | {erms:9.3f} | {hmax:18.3f} | {hmean:19.3f}")
    header("Додатково 1. Характеристики маршруту")
    total_ascent = sum(max(y[i] - y[i - 1], 0) for i in range(1, n_pts))
    total_descent = sum(max(y[i - 1] - y[i], 0) for i in range(1, n_pts))
    dyy = np.diff(yy_full)
    print(f"Загальна довжина маршруту (м): {x[-1]:.2f}")
    print(f"Сумарний набір висоти (м):     {total_ascent:.2f}")
    print(f"Сумарний спуск (м):            {total_descent:.2f}")
    print(f"Набір / спуск за сплайном (м): {dyy[dyy > 0].sum():.2f} / {-dyy[dyy < 0].sum():.2f}")
    print(f"Висота старту / фінішу (м):    {y[0]:.2f} / {y[-1]:.2f}")
    print(f"Мін. / макс. висота сплайна (м): {yy_full.min():.2f} / {yy_full.max():.2f}")
    print(f"Середній нахил (y_n - y_0)/L:  {(y[-1] - y[0]) / x[-1] * 100:.2f} %")
    t_naismith = x[-1] / 1000 / 5 + total_ascent / 600
    print(f"Час за правилом Нейсміта (5 км/год + 1 год на 600 м набору): "
          f"{t_naismith:.2f} год ≈ {int(t_naismith)} год {round(t_naismith % 1 * 60)} хв")
    header("Додатково 2. Аналіз градієнта (через похідну сплайна)")
    grad_full = sp.derivative(xx) * 100
    grad_np = np.gradient(yy_full, xx) * 100
    print(f"Контроль: max |S'(x)·100 - np.gradient| = {np.max(np.abs(grad_full - grad_np)):.4f} %")
    print(f"Максимальний підйом (%): {grad_full.max():.2f}  (x = {xx[np.argmax(grad_full)]:.1f} м)")
    print(f"Максимальний спуск (%):  {grad_full.min():.2f}  (x = {xx[np.argmin(grad_full)]:.1f} м)")
    print(f"Середній градієнт (%):   {np.mean(np.abs(grad_full)):.2f}")
    steep = runs(grad_full > STEEP)
    steep_len = sum(xx[e] - xx[s] for s, e in steep)
    print(f"\nДілянки з крутизною > {STEEP:.0f} %:")
    for s, e in steep:
        seg = grad_full[s:e + 1]
        print(f"  {xx[s]:7.1f} - {xx[e]:7.1f} м  (довжина {xx[e] - xx[s]:6.1f} м, "
              f"висота {yy_full[s]:.0f} -> {yy_full[e]:.0f} м, макс. {seg.max():.1f} %)")
    print(f"Сумарно: {steep_len:.1f} м ({steep_len / x[-1] * 100:.1f} % довжини маршруту)")
    print("\nРозподіл довжини маршруту за крутизною:")
    dx = xx[1] - xx[0]
    for lo, hi in ((-np.inf, 0), (0, 5), (5, 10), (10, 15), (15, 25), (25, 35), (35, np.inf)):
        length = np.count_nonzero((grad_full >= lo) & (grad_full < hi)) * dx
        name = "спуск (< 0 %)" if lo == -np.inf else (f"> {lo:.0f} %" if hi == np.inf else f"{lo:.0f}-{hi:.0f} %")
        print(f"  {name:>14}: {length:7.1f} м ({length / x[-1] * 100:5.1f} %)")
    plot_gradient(xx, grad_full, steep)
    header("Додатково 3. Механічна енергія підйому")
    energy = MASS * G * total_ascent
    print(f"Маса: {MASS:.0f} кг, g = {G} м/с²")
    print(f"Механічна робота (Дж):  {energy:.1f}")
    print(f"Механічна робота (кДж): {energy / 1000:.2f}")
    print(f"Енергія (ккал):         {energy / 4184:.2f}")
    print(f"\nГрафіки та таблиці збережено в {OUT_DIR}")
    sys.stdout = sys.__stdout__
    log.close()

if __name__ == "__main__":
    main()
