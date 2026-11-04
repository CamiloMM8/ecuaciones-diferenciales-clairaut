"""Gráfica de la familia y = Cx + f(C) y de la solución singular."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.core.expr import Expr

from .clairaut import ClairautResult, SingularBranch
from .parser import P, X

_LINE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
]
_SINGULAR_COLORS = ["#9b1b30", "#111111", "#7a4e00"]


def _slug(text: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in text.strip())
    cleaned = cleaned.strip("_") or "f"
    return cleaned[:40]


def _eval_array(expr: Expr, symbol: sp.Symbol, values: np.ndarray) -> np.ndarray:
    """Evalúa una expresión sobre un arreglo y devuelve complejos (nan si falla)."""
    try:
        func = sp.lambdify(symbol, expr, modules=["numpy"])
        with np.errstate(all="ignore"):
            raw = func(np.asarray(values, dtype=float))
        out = np.asarray(raw, dtype=complex)
        if out.shape != values.shape:
            raise ValueError("forma inesperada")
        return out
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        out = np.empty(values.shape, dtype=complex)
        for index, value in enumerate(values):
            try:
                out[index] = complex(expr.subs(symbol, float(value)).evalf())
            except (TypeError, ValueError):
                out[index] = complex(np.nan, np.nan)
        return out


def _real_mask(values: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    return np.isfinite(values.real) & np.isfinite(values.imag) & (np.abs(values.imag) < tol)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Tramos contiguos de una máscara, para no unir puntos separados por un hueco."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _plot_masked(ax, x_vals: np.ndarray, y_vals: np.ndarray, mask: np.ndarray, **kwargs) -> bool:
    label = kwargs.pop("label", None)
    drawn = False
    for start, end in _segments(mask):
        if end - start < 2:
            continue
        ax.plot(
            x_vals[start:end],
            y_vals[start:end],
            label=label if not drawn else None,
            **kwargs,
        )
        drawn = True
    return drawn


def _singular_mask(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    p_vals: np.ndarray,
    fp: Expr,
    f: Expr,
) -> np.ndarray:
    """
    Conserva solo los x donde p y y son reales y se cumple x + f'(p) = 0.

    La segunda condición evita dibujar la prolongación algebraica de una raíz
    extraña, como la rama x > 0 de √p.
    """
    mask = _real_mask(y_vals) & _real_mask(p_vals)
    if not np.any(mask):
        return mask

    p_real = np.where(mask, p_vals.real, np.nan)
    fp_vals = _eval_array(fp, P, p_real)
    f_vals = _eval_array(f, P, p_real)
    fp_ok = _real_mask(fp_vals)
    f_ok = _real_mask(f_vals)
    cond = np.abs(x_vals + fp_vals.real)
    recon = x_vals * p_vals.real + f_vals.real
    scale = 1.0 + np.abs(x_vals) + np.abs(recon)
    consistent = np.abs(y_vals.real - recon) <= 1e-3 * scale
    mask = mask & fp_ok & f_ok & (cond <= 1e-3 * (1.0 + np.abs(x_vals))) & consistent
    return mask


def _sample_parametric(
    x_of_p: Expr,
    y_of_p: Expr,
    a: float,
    b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Muestrea p y se queda con los puntos cuya x cae en [a, b]."""
    grids = (
        np.linspace(-12.0, 12.0, 1400),
        np.linspace(1e-3, 12.0, 400),
        np.linspace(-12.0, -1e-3, 400),
    )
    xs: list[float] = []
    ys: list[float] = []
    for grid in grids:
        x_vals = _eval_array(x_of_p, P, grid)
        y_vals = _eval_array(y_of_p, P, grid)
        ok = _real_mask(x_vals) & _real_mask(y_vals)
        for x_val, y_val, keep in zip(x_vals.real, y_vals.real, ok):
            if keep and a <= float(x_val) <= b and np.isfinite(y_val):
                xs.append(float(x_val))
                ys.append(float(y_val))
    if not xs:
        return np.array([]), np.array([])
    order = np.argsort(xs)
    return np.asarray(xs)[order], np.asarray(ys)[order]


def _ylim(singular_vals: list[float], line_vals: list[float]) -> tuple[float, float]:
    """
    Encuadra la envolvente y deja un margen para ver los tramos de recta
    tangentes. Las rectas que se salen de esa ventana se recortan.
    """
    singular = np.asarray([value for value in singular_vals if np.isfinite(value)], dtype=float)
    lines = np.asarray([value for value in line_vals if np.isfinite(value)], dtype=float)

    if singular.size >= 8:
        low, high = np.percentile(singular, [5, 95])
        median = float(np.median(singular))
        mad = float(np.median(np.abs(singular - median)))
        cap = max(2.0, 8.0 * max(mad, 0.25))
        low = max(float(low), median - cap)
        high = min(float(high), median + cap)
        span = max(high - low, 1.0)
        low -= 0.35 * span
        high += 0.55 * span
    elif lines.size:
        low, high = (float(value) for value in np.percentile(lines, [8, 92]))
        span = max(high - low, 1.0)
        low -= 0.15 * span
        high += 0.15 * span
    else:
        low, high = -1.0, 1.0

    span = max(high - low, 1.0)
    if 0 < low < span:
        low = -0.08 * span
    if -span < high < 0:
        high = 0.08 * span
    if high - low < 1e-6:
        low, high = low - 1.0, high + 1.0
    return float(low), float(high)


def _format_c(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def plot_clairaut(
    result: ClairautResult,
    f_text: str,
    a: float,
    b: float,
    output_dir: Path,
    case_label: str,
) -> tuple[Path | None, str | None, str | None]:
    """
    Dibuja la familia y la solución singular en una sola figura.

    Retorna (ruta_png, error, nota). La nota avisa si alguna rama no corta
    el intervalo, sin invalidar el resto de la gráfica.
    """
    if a >= b:
        return None, "Intervalo de gráfica inválido.", None

    x_vals = np.linspace(a, b, 700)
    fig, ax = plt.subplots(figsize=(10, 6))
    singular_y: list[float] = []
    line_y: list[float] = []
    notes: list[str] = []
    drawn = 0

    try:
        for index, status in enumerate(result.constants):
            if not status.valid or status.f_numeric is None:
                continue
            y_line = status.value * x_vals + status.f_numeric
            finite = np.isfinite(y_line)
            if np.count_nonzero(finite) < 2:
                continue
            color = _LINE_COLORS[index % len(_LINE_COLORS)]
            ax.plot(
                x_vals[finite],
                y_line[finite],
                color=color,
                linewidth=1.6,
                label=f"C = {_format_c(status.value)}",
                zorder=2,
            )
            line_y.extend(y_line[finite].tolist())
            drawn += 1

        for index, branch in enumerate(result.branches, start=1):
            used = _draw_branch(ax, branch, result, x_vals, index, singular_y)
            if used:
                drawn += 1
            else:
                notes.append(
                    f"La rama singular {index} no tiene puntos reales dentro de [{a:g}, {b:g}]."
                )

        if result.parametric is not None and not result.branches:
            x_param, y_param = result.parametric
            px, py = _sample_parametric(x_param, y_param, a, b)
            if px.size >= 2:
                ax.plot(
                    px,
                    py,
                    color=_SINGULAR_COLORS[0],
                    linewidth=2.8,
                    label="Solución singular",
                    zorder=4,
                )
                singular_y.extend(py.tolist())
                drawn += 1
            else:
                notes.append(
                    "La forma paramétrica de la envolvente no corta el intervalo elegido."
                )

        if drawn == 0:
            return None, "No hay curvas reales para graficar en el intervalo indicado.", None

        low, high = _ylim(singular_y, line_y)
        ax.set_xlim(a, b)
        ax.set_ylim(low, high)
        ax.axhline(0, color="black", linewidth=0.9, zorder=1)
        ax.axvline(0, color="black", linewidth=0.9, zorder=1)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, framealpha=0.95)
        try:
            f_latex = sp.latex(result.f_simplified)
            title = f"Ecuación de Clairaut\n$y = xp + {f_latex}$"
        except (TypeError, ValueError):
            title = f"Ecuación de Clairaut\ny = xp + ({f_text})"
        ax.set_title(title, fontsize=13, pad=10)

        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"grafica_{_slug(case_label)}.png"
        fig.tight_layout()
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)

    note = " ".join(notes) if notes else None
    return filepath, None, note


def _draw_branch(
    ax,
    branch: SingularBranch,
    result: ClairautResult,
    x_vals: np.ndarray,
    index: int,
    singular_y: list[float],
) -> bool:
    y_vals = _eval_array(branch.y_of_x, X, x_vals)
    p_vals = _eval_array(branch.p_of_x, X, x_vals)
    mask = _singular_mask(x_vals, y_vals, p_vals, result.fp, result.f_simplified)
    label = "Solución singular" if len(result.branches) == 1 else f"Solución singular {index}"
    drawn = _plot_masked(
        ax,
        x_vals,
        y_vals.real,
        mask,
        color=_SINGULAR_COLORS[(index - 1) % len(_SINGULAR_COLORS)],
        linewidth=2.8,
        linestyle="-",
        label=label,
        zorder=4,
    )
    if drawn:
        singular_y.extend(y_vals.real[mask].tolist())
    return drawn
