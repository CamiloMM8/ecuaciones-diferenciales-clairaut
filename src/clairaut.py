"""Método clásico para la ecuación de Clairaut y = x p + f(p).

No se usa dsolve. La solución general sale de p' = 0, es decir p = C.
La solución singular sale del otro factor, x + f'(p) = 0, y de sustituir
ese p en y = x p + f(p).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from sympy.core.expr import Expr

from .parser import C, P, X

# Ventana amplia para decidir en qué parte del eje real vive cada rama,
# independiente del intervalo que el usuario elija para la gráfica.
_PROBE_LEFT = -8.0
_PROBE_RIGHT = 8.0
_PROBE_SAMPLES = 81


@dataclass
class ConstantStatus:
    """Un valor de C y si f(C) es un número real."""

    value: float
    valid: bool
    f_value: Expr | None = None
    f_numeric: float | None = None
    reason: str = ""


@dataclass
class SingularBranch:
    """Una rama explícita de la solución singular."""

    p_of_x: Expr
    y_substituted: Expr
    y_of_x: Expr
    domain_note: str
    verified_symbolic: bool
    verified_numeric: bool
    symbolic_residual: Expr
    sample_xs: list[float] = field(default_factory=list)


@dataclass
class ClairautResult:
    """Resultado simbólico del método, listo para mostrarse paso a paso."""

    f: Expr
    f_simplified: Expr
    fp: Expr
    general: Expr
    condition: Expr
    constants: list[ConstantStatus]
    branches: list[SingularBranch]
    # Forma paramétrica de la envolvente: x(p), y(p). Solo se usa si no hubo
    # ramas explícitas y f' sí depende de p.
    parametric: tuple[Expr, Expr] | None
    no_singular_reason: str | None
    discarded_note: str | None = None


def _as_expr_list(solutions) -> list[Expr]:
    """Normaliza lo que devuelve solve a una lista de expresiones en p."""
    if solutions is None:
        return []
    items = solutions if isinstance(solutions, (list, tuple, set)) else [solutions]
    found: list[Expr] = []
    for item in items:
        if isinstance(item, dict):
            if P in item and isinstance(item[P], Expr):
                found.append(item[P])
            continue
        if isinstance(item, Expr) and not isinstance(item, sp.Rel):
            found.append(item)
    return found


def _depends_on_p(expr: Expr) -> bool:
    return P in expr.free_symbols


def _is_undefined(value: Expr) -> bool:
    if getattr(value, "is_infinite", False) or getattr(value, "is_nan", False):
        return True
    try:
        if value in (sp.zoo, sp.oo, -sp.oo, sp.nan):
            return True
    except TypeError:
        return True
    return bool(value.has(sp.zoo, sp.oo, -sp.oo, sp.nan))


def classify_constants(f: Expr, c_values: list[float]) -> list[ConstantStatus]:
    """
    La recta y = C x + f(C) solo se grafica si f(C) es real y finita.

    Así se excluye C = 0 en 1/p y los C negativos en √p, sin fijar el programa
    a esos ejemplos.
    """
    statuses: list[ConstantStatus] = []
    for raw in c_values:
        c_sym = sp.nsimplify(raw, rational=True)
        try:
            value = sp.simplify(f.subs(P, c_sym))
        except (TypeError, ValueError):
            statuses.append(
                ConstantStatus(raw, False, reason="f(C) no se pudo evaluar.")
            )
            continue

        if _is_undefined(value):
            statuses.append(
                ConstantStatus(
                    raw,
                    False,
                    reason="f no está definida en este valor de C.",
                )
            )
            continue

        try:
            numeric = complex(value.evalf())
        except (TypeError, ValueError):
            statuses.append(
                ConstantStatus(raw, False, value, reason="f(C) no es numérico.")
            )
            continue

        if abs(numeric.imag) > 1e-8 or not np.isfinite(numeric.real):
            statuses.append(
                ConstantStatus(
                    raw,
                    False,
                    value,
                    reason="f(C) no es un número real.",
                )
            )
            continue

        statuses.append(
            ConstantStatus(raw, True, value, float(numeric.real))
        )
    return statuses


def _solve_unrestricted(fp: Expr) -> list[Expr]:
    try:
        return _as_expr_list(sp.solve(sp.Eq(X + fp, 0), P))
    except (NotImplementedError, TypeError, ValueError):
        return []


def _solve_with_assumptions(fp: Expr) -> list[Expr]:
    """
    Segundo intento cuando solve no despeja p.

    Hace falta, por ejemplo, en f(p) = √p: f'(p) es positiva, así que
    x = -f'(p) obliga a x < 0 y p > 0. Sin esas hipótesis SymPy devuelve
    el conjunto vacío.
    """
    trials = (
        (True, False, False, True),
        (True, False, True, False),
        (False, True, False, True),
        (False, True, True, False),
    )
    found: list[Expr] = []
    for p_pos, p_neg, x_pos, x_neg in trials:
        p_s = sp.Symbol("p", real=True, positive=p_pos, negative=p_neg)
        x_s = sp.Symbol("x", real=True, positive=x_pos, negative=x_neg)
        try:
            solutions = sp.solve(sp.Eq(x_s + fp.subs(P, p_s), 0), p_s)
        except (NotImplementedError, TypeError, ValueError):
            continue
        for solution in _as_expr_list(solutions):
            rewritten = sp.simplify(solution.xreplace({x_s: X, p_s: P}))
            if rewritten.free_symbols - {X}:
                continue
            found.append(rewritten)
    return found


def _unique_exprs(expressions: list[Expr]) -> list[Expr]:
    unique: list[Expr] = []
    for expr in expressions:
        if any(_same_expr(expr, previous) for previous in unique):
            continue
        unique.append(expr)
    return unique


def _same_expr(left: Expr, right: Expr) -> bool:
    try:
        return sp.simplify(left - right) == 0
    except (TypeError, ValueError):
        return False


def _probe_points() -> np.ndarray:
    grid = np.linspace(_PROBE_LEFT, _PROBE_RIGHT, _PROBE_SAMPLES)
    # Se conserva el cero para detectar ramas que llegan al origen (como p³)
    # y se descarta solo si la evaluación no es real.
    return np.unique(np.append(grid, 0.0))


def _eval_complex(expr: Expr, symbol: sp.Symbol, value: float) -> complex | None:
    try:
        numeric = complex(expr.subs(symbol, value).evalf())
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric.real) or not np.isfinite(numeric.imag):
        return None
    return numeric


def _real_value(expr: Expr, symbol: sp.Symbol, value: float, tol: float = 1e-7) -> float | None:
    numeric = _eval_complex(expr, symbol, value)
    if numeric is None or abs(numeric.imag) > tol:
        return None
    return float(numeric.real)


def _branch_samples(
    p_of_x: Expr, fp: Expr, f: Expr
) -> list[tuple[float, float, float]]:
    """
    Puntos (x, p, y) en los que p(x) es real y cumple x + f'(p) = 0.

    Esa segunda comprobación descarta raíces extrañas. En √p, p = 1/(4x²)
    también es positiva para x > 0, pero allí no cumple la condición porque
    f'(p) no puede ser negativa.
    """
    samples: list[tuple[float, float, float]] = []
    for raw_x in _probe_points():
        p_val = _real_value(p_of_x, X, float(raw_x))
        if p_val is None:
            continue
        fp_val = _real_value(fp, P, p_val)
        f_val = _real_value(f, P, p_val)
        if fp_val is None or f_val is None:
            continue
        if abs(float(raw_x) + fp_val) > 1e-5 * (1.0 + abs(float(raw_x))):
            continue
        y_val = float(raw_x) * p_val + f_val
        if not np.isfinite(y_val):
            continue
        samples.append((float(raw_x), p_val, y_val))
    return samples


def _domain_note(samples: list[tuple[float, float, float]]) -> str:
    xs = [item[0] for item in samples]
    if not xs:
        return "No se encontraron puntos reales que cumplan la condición."
    hits = len(xs)
    total = len(_probe_points())
    if hits == total:
        return "Cumple x + f'(p) = 0 en todo el intervalo real analizado."
    if hits / total > 0.9:
        return (
            "Cumple x + f'(p) = 0 en el intervalo real analizado, "
            "salvo puntos aislados donde la expresión no está definida."
        )
    if all(value < 0 for value in xs):
        return "Cumple x + f'(p) = 0 solo cuando x es negativo."
    if all(value > 0 for value in xs):
        return "Cumple x + f'(p) = 0 solo cuando x es positivo."
    if all(value <= 0 for value in xs):
        return "Cumple x + f'(p) = 0 solo cuando x es menor o igual que cero."
    if all(value >= 0 for value in xs):
        return "Cumple x + f'(p) = 0 solo cuando x es mayor o igual que cero."
    return "Cumple x + f'(p) = 0 solo en una parte del eje real."


def _matches_on(left: Expr, right: Expr, xs: list[float]) -> bool:
    if not xs:
        return False
    for raw_x in xs:
        a_val = _real_value(left, X, raw_x, tol=1e-6)
        b_val = _real_value(right, X, raw_x, tol=1e-6)
        if a_val is None or b_val is None:
            return False
        scale = max(1.0, abs(a_val), abs(b_val))
        if abs(a_val - b_val) > 1e-5 * scale:
            return False
    return True


def _beautify(expr: Expr, xs: list[float]) -> Expr:
    """
    Simplifica y, si toda la rama cae en x < 0 o en x > 0, reescribe la
    fórmula con esa hipótesis. Así 1/p queda como ±2√x y √p como −1/(4x).
    """
    simplified = sp.simplify(sp.powsimp(expr, force=True))
    simplified = sp.simplify(simplified)
    if not xs:
        return simplified

    replacement: Expr | None = None
    if all(value < 0 for value in xs):
        x_neg = sp.Symbol("x", negative=True)
        replacement = sp.simplify(simplified.xreplace({X: x_neg})).xreplace({x_neg: X})
    elif all(value > 0 for value in xs):
        x_pos = sp.Symbol("x", positive=True)
        replacement = sp.simplify(simplified.xreplace({X: x_pos})).xreplace({x_pos: X})

    if replacement is None or replacement == simplified:
        return simplified
    if sp.count_ops(replacement) < sp.count_ops(simplified) and _matches_on(
        simplified, replacement, xs[:6]
    ):
        return replacement
    return simplified


def _verify_branch(
    y_of_x: Expr, f: Expr, samples: list[tuple[float, float, float]]
) -> tuple[bool, bool, Expr]:
    """
    Comprueba y − x y' − f(y') = 0.

    La comprobación simbólica puede quedar con un valor absoluto (ramas de
    una raíz). En ese caso se acepta la comprobación numérica sobre la muestra.
    """
    try:
        derivative = sp.diff(y_of_x, X)
        residual = sp.simplify(y_of_x - X * derivative - f.subs(P, derivative))
        symbolic_ok = residual == 0 or bool(residual.equals(0))
    except (TypeError, ValueError):
        residual = sp.Integer(1)
        symbolic_ok = False

    numeric_ok = False
    if samples:
        checked = 0
        passed = 0
        for raw_x, _p_val, y_val in samples[:8]:
            y_here = _real_value(y_of_x, X, raw_x, tol=1e-5)
            if y_here is None:
                continue
            checked += 1
            scale = max(1.0, abs(y_val), abs(y_here))
            if abs(y_here - y_val) <= 1e-4 * scale:
                passed += 1
        numeric_ok = checked > 0 and passed == checked
    return symbolic_ok, numeric_ok, residual


def _build_branch(p_of_x: Expr, f: Expr, fp: Expr) -> SingularBranch | None:
    samples = _branch_samples(p_of_x, fp, f)
    # Si el residual simbólico es idénticamente cero, la rama es válida aunque
    # la muestra numérica haya caído en singularidades.
    try:
        condition_residual = sp.simplify((X + fp).subs(P, p_of_x))
        condition_identity = condition_residual == 0 or bool(condition_residual.equals(0))
    except (TypeError, ValueError):
        condition_identity = False

    if not samples and not condition_identity:
        return None

    # Suma sin combinar términos: primero xp y después f(p), ya con p sustituido.
    # Así el paso intermedio de p² se ve como −x²/2 + x²/4 antes de llegar a −x²/4.
    linear = sp.expand(X * p_of_x)
    nonlinear = sp.expand(f.subs(P, p_of_x))
    y_substituted = sp.Add(linear, nonlinear, evaluate=False)
    xs = [item[0] for item in samples]
    y_of_x = _beautify(y_substituted, xs)
    symbolic_ok, numeric_ok, residual = _verify_branch(y_of_x, f, samples)
    if not symbolic_ok and not numeric_ok and not condition_identity:
        return None

    return SingularBranch(
        p_of_x=sp.simplify(p_of_x),
        y_substituted=y_substituted,
        y_of_x=y_of_x,
        domain_note=_domain_note(samples),
        verified_symbolic=symbolic_ok,
        verified_numeric=numeric_ok or condition_identity,
        symbolic_residual=residual,
        sample_xs=xs,
    )


def _parametric_curve(f: Expr, fp: Expr) -> tuple[Expr, Expr] | None:
    """Envolvente paramétrica x = −f'(p), y = f(p) − p f'(p)."""
    if not _depends_on_p(fp):
        return None
    x_of_p = sp.simplify(-fp)
    y_of_p = sp.simplify(f - P * fp)
    if not _depends_on_p(x_of_p) and not _depends_on_p(y_of_p):
        return None
    return x_of_p, y_of_p


def analyze_clairaut(f: Expr, c_values: list[float]) -> ClairautResult:
    """Ejecuta el método de Clairaut sobre una f(p) ya validada."""
    f_simplified = sp.simplify(f)
    # f'(p) es el dato que convierte la derivación de y = xp + f(p)
    # en el factor (x + f'(p)) p' = 0.
    fp = sp.simplify(sp.diff(f_simplified, P))
    # expand conserva la forma Cx + f(C). simplify la factoriza y deja de parecerse
    # a la fórmula del enunciado (por ejemplo C(C + x) en lugar de Cx + C²).
    general = sp.expand(C * X + f_simplified.subs(P, C))
    condition = sp.simplify(X + fp)
    constants = classify_constants(f_simplified, c_values)

    no_singular: str | None = None
    branches: list[SingularBranch] = []
    discarded = 0

    if not _depends_on_p(fp):
        # f' constante: x + f'(p) = 0 no despeja p. La familia es un haz de
        # rectas (o rectas paralelas) y no tiene curva envolvente.
        no_singular = (
            "f'(p) es constante, así que x + f'(p) = 0 no determina p. "
            "No hay solución singular."
        )
    else:
        candidates = _unique_exprs(_solve_unrestricted(fp))
        built = [_build_branch(candidate, f_simplified, fp) for candidate in candidates]
        discarded += sum(branch is None for branch in built)
        branches = [branch for branch in built if branch is not None]

        if not branches:
            assumed = _unique_exprs(_solve_with_assumptions(fp))
            built = [_build_branch(candidate, f_simplified, fp) for candidate in assumed]
            discarded += sum(branch is None for branch in built)
            branches = [branch for branch in built if branch is not None]

        if not branches:
            no_singular = (
                "No fue posible despejar una rama real p(x) a partir de "
                "x + f'(p) = 0. Se usa la forma paramétrica de la envolvente "
                "cuando f'(p) no es constante."
            )

    parametric = None if branches else _parametric_curve(f_simplified, fp)
    if branches:
        parametric = None
    elif parametric is None and no_singular is None:
        no_singular = "No hay solución singular."

    discarded_note = None
    if discarded:
        discarded_note = (
            "Se descartaron expresiones de p que no cumplen "
            "x + f'(p) = 0 sobre los reales."
        )

    return ClairautResult(
        f=f,
        f_simplified=f_simplified,
        fp=fp,
        general=general,
        condition=condition,
        constants=constants,
        branches=branches,
        parametric=parametric,
        no_singular_reason=None if branches or parametric else no_singular,
        discarded_note=discarded_note,
    )


