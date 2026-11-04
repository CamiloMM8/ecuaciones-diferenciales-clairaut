"""Orquesta el análisis y arma el texto paso a paso de la interfaz."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import sympy as sp

from .clairaut import ClairautResult, SingularBranch, analyze_clairaut
from .latex_fmt import block, bold_labeled, labeled
from .parser import C, P, X, Y, ValidationResult, validate_input
from .plotting import plot_clairaut


@dataclass
class AnalysisResult:
    """Análisis completo, con las secciones que ve el usuario."""

    validation: ValidationResult
    clairaut: ClairautResult | None = None
    plot_path: Path | None = None
    plot_error: str | None = None
    plot_note: str | None = None
    stopped_reason: str | None = None
    sections: dict[str, list[str]] = field(default_factory=dict)


def _latex(expr) -> str:
    return sp.latex(expr)


def _format_c(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _prime_symbol():
    """Símbolo y' para escribir la ecuación como en el enunciado."""
    return sp.Symbol(r"y'")


def run_analysis(
    f_text: str,
    c_text: str,
    a: float,
    b: float,
    output_dir: Path,
    case_label: str = "custom",
) -> AnalysisResult:
    """Valida la entrada, aplica el método de Clairaut y genera la gráfica."""
    validation = validate_input(f_text, c_text, a, b)
    result = AnalysisResult(validation=validation)
    sections: dict[str, list[str]] = {"Validación": [validation.message]}

    if not validation.valid or validation.parsed is None:
        result.sections = sections
        result.stopped_reason = validation.message
        return result

    parsed = validation.parsed
    clairaut = analyze_clairaut(parsed.f, parsed.c_values)
    result.clairaut = clairaut

    sections["Ecuación diferencial"] = _section_equation(clairaut)
    sections["Reescritura con p"] = _section_rewrite(clairaut)
    sections["Solución general"] = _section_general(clairaut)
    sections["Derivada f'(p)"] = [
        "Se deriva f respecto de p. Esta derivada aparece al derivar la ecuación respecto de x.",
        bold_labeled("f'(p) =", _latex(clairaut.fp)),
    ]
    sections["Condición para la solución singular"] = _section_condition(clairaut)
    sections["Despeje de p"] = _section_solve(clairaut)
    sections["Sustitución"] = _section_substitution(clairaut)
    sections["Solución singular"] = _section_singular(clairaut)
    sections["Restricciones"] = _section_restrictions(clairaut)
    sections["Gráfica"] = _section_plot_intro(clairaut, a, b)

    plot_path, plot_error, plot_note = plot_clairaut(
        clairaut, parsed.f_text, a, b, output_dir, case_label
    )
    result.plot_path = plot_path
    result.plot_error = plot_error
    result.plot_note = plot_note
    if plot_error:
        sections["Gráfica"].append(f"**No se generó la gráfica:** {plot_error}")
    elif plot_path is not None:
        sections["Gráfica"].append(f"Gráfica guardada en `{plot_path.name}`.")
    if plot_note:
        sections["Gráfica"].append(plot_note)

    result.sections = sections
    return result


def _section_equation(clairaut: ClairautResult) -> list[str]:
    yp = _prime_symbol()
    f_of_yp = clairaut.f_simplified.subs(P, yp)
    equation = sp.Eq(Y, X * yp + f_of_yp)
    return [
        "A partir de la función ingresada se construye la ecuación de Clairaut.",
        block(_latex(equation)),
        labeled("f(p) =", _latex(clairaut.f_simplified)),
    ]


def _section_rewrite(clairaut: ClairautResult) -> list[str]:
    lines = [
        "Se escribe la derivada como p = y'. La ecuación queda:",
        block(rf"y = xp + {_latex(clairaut.f_simplified)}"),
    ]
    if not _same(clairaut.f, clairaut.f_simplified):
        lines.append(labeled("f simplificada:", _latex(clairaut.f_simplified)))
    return lines


def _section_general(clairaut: ClairautResult) -> list[str]:
    lines = [
        "Se deriva y = x p + f(p) respecto de x:",
        block(r"p = p + \left(x + f'(p)\right) p'"),
        "Al restar p en ambos lados queda el producto",
        block(r"\left(x + f'(p)\right) p' = 0"),
        "Una posibilidad es p' = 0. Entonces p es una constante C. "
        "Al sustituirla en la ecuación se obtiene la solución general, una familia de rectas:",
        block(rf"y = Cx + {_latex(clairaut.f_simplified.subs(P, C))}"),
    ]
    valid = [status for status in clairaut.constants if status.valid and status.f_value is not None]
    if valid:
        lines.append("Para los valores de C pedidos, las rectas son:")
        for status in valid:
            assert status.f_value is not None
            c_sym = sp.nsimplify(status.value, rational=True)
            particular = sp.Eq(Y, sp.expand(c_sym * X + status.f_value))
            lines.append(labeled(f"C = {_format_c(status.value)}:", _latex(particular)))
    else:
        lines.append("Ningún valor de C de la lista deja f(C) definida en los reales.")
    return lines


def _section_condition(clairaut: ClairautResult) -> list[str]:
    lines = [
        "La otra posibilidad del producto es el factor que no corresponde a una constante:",
        block(r"x + f'(p) = 0"),
        labeled("Con la derivada calculada:", rf"x + {_latex(clairaut.fp)} = 0"),
    ]
    if clairaut.fp == 0 or sp.simplify(clairaut.fp) == 0:
        lines.append("Como f'(p) = 0, la condición se reduce a x = 0 y no define una curva envolvente.")
    return lines


def _section_solve(clairaut: ClairautResult) -> list[str]:
    if not clairaut.branches:
        lines = [
            "No se obtuvo una expresión explícita p(x) que cumpla la condición sobre los reales."
        ]
        if clairaut.parametric is not None:
            lines.append(
                "La envolvente se puede describir sin despejar p, con el parámetro p:"
            )
            x_of_p, y_of_p = clairaut.parametric
            lines.append(labeled("x(p) =", _latex(x_of_p)))
            lines.append(labeled("y(p) =", _latex(y_of_p)))
        elif clairaut.no_singular_reason:
            lines.append(clairaut.no_singular_reason)
        return lines

    lines = ["Se despeja p en función de x."]
    for index, branch in enumerate(clairaut.branches, start=1):
        prefix = "p =" if len(clairaut.branches) == 1 else f"Rama {index}: p ="
        lines.append(bold_labeled(prefix, _latex(branch.p_of_x)))
        lines.append(branch.domain_note)
    if clairaut.discarded_note:
        lines.append(clairaut.discarded_note)
    return lines


def _section_substitution(clairaut: ClairautResult) -> list[str]:
    if not clairaut.branches:
        if clairaut.parametric is not None:
            return [
                "Como p no quedó despejado, la sustitución ya está hecha en la forma paramétrica: "
                "x = −f'(p) se reemplaza en y = x p + f(p) y resulta y = f(p) − p f'(p)."
            ]
        return ["No hay una expresión de p para sustituir en y = x p + f(p)."]

    lines = ["Cada valor de p se reemplaza en y = x p + f(p)."]
    for index, branch in enumerate(clairaut.branches, start=1):
        title = "Sustitución" if len(clairaut.branches) == 1 else f"Rama {index}"
        lines.append(f"**{title}**")
        lines.append(labeled("Al sustituir:", _latex(branch.y_substituted)))
        if _latex(branch.y_substituted) != _latex(branch.y_of_x):
            lines.append(labeled("Al simplificar:", _latex(branch.y_of_x)))
    return lines


def _section_singular(clairaut: ClairautResult) -> list[str]:
    if not clairaut.branches and clairaut.parametric is None:
        return [clairaut.no_singular_reason or "No hay solución singular."]

    lines: list[str] = []
    if clairaut.branches:
        lines.append("La solución singular es la envolvente de la familia de rectas.")
        for index, branch in enumerate(clairaut.branches, start=1):
            if len(clairaut.branches) > 1:
                lines.append(f"**Rama {index}**")
            lines.append(block(_latex(sp.Eq(Y, branch.y_of_x))))
            lines.append(branch.domain_note)
            lines.extend(_verification_lines(branch))
    else:
        assert clairaut.parametric is not None
        x_of_p, y_of_p = clairaut.parametric
        lines.extend(
            [
                "No quedó una fórmula explícita y(x). La solución singular se presenta en forma paramétrica:",
                labeled("x(p) =", _latex(x_of_p)),
                labeled("y(p) =", _latex(y_of_p)),
                "Al variar p se recorre la envolvente de la familia.",
            ]
        )
    return lines


def _verification_lines(branch: SingularBranch) -> list[str]:
    lines = [
        "Comprobación: si y es la solución singular, debe cumplirse "
        "y − x y' − f(y') = 0 en su dominio."
    ]
    if branch.verified_symbolic:
        lines.append(
            "**Verificación simbólica correcta:** el residual se simplifica a 0."
        )
        return lines

    lines.append(labeled("Residual simbólico:", _latex(branch.symbolic_residual)))
    if branch.verified_numeric:
        lines.append(
            "El residual no se redujo a 0 por las ramas de las raíces o los valores absolutos. "
            "**La comprobación numérica sí se cumple** en puntos interiores del dominio."
        )
    else:
        lines.append(
            "No se pudo confirmar numéricamente esta rama. Revise el dominio antes de usarla."
        )
    return lines


def _section_restrictions(clairaut: ClairautResult) -> list[str]:
    lines: list[str] = []
    excluded = [status for status in clairaut.constants if not status.valid]
    if not excluded:
        lines.append(
            "Todos los valores de C indicados producen un f(C) real. "
            "Esas rectas entran en la gráfica."
        )
    else:
        lines.append(
            "Algunos valores de C no generan una recta real, porque f(C) no está definida "
            "o no es real. Esas constantes se omiten en la familia:"
        )
        for status in excluded:
            lines.append(f"- C = {_format_c(status.value)}: {status.reason}")

    for index, branch in enumerate(clairaut.branches, start=1):
        name = "La solución singular" if len(clairaut.branches) == 1 else f"La rama {index}"
        lines.append(f"{name}: {branch.domain_note}")

    if clairaut.discarded_note:
        lines.append(clairaut.discarded_note)
    if not lines:
        lines.append("No hay restricciones adicionales para los datos ingresados.")
    return lines


def _section_plot_intro(clairaut: ClairautResult, a: float, b: float) -> list[str]:
    valid = [_format_c(status.value) for status in clairaut.constants if status.valid]
    if valid:
        family = "Se grafican las rectas de C = " + ", ".join(valid) + "."
    else:
        family = "No hay rectas reales de la solución general para graficar."

    if clairaut.branches:
        singular = (
            "En la misma figura se traza la solución singular."
            if len(clairaut.branches) == 1
            else f"En la misma figura se trazan las {len(clairaut.branches)} ramas de la solución singular."
        )
    elif clairaut.parametric is not None:
        singular = "En la misma figura se traza la envolvente paramétrica."
    else:
        singular = "No hay solución singular para superponer."

    return [
        family,
        singular,
        f"Intervalo horizontal: [{a:g}, {b:g}]. "
        "La escala vertical se ajusta para dejar visible la envolvente; "
        "los tramos de recta que se salen de esa ventana se recortan.",
        "La figura incluye título, ejes coordenados, cuadrícula, escala y leyenda.",
    ]


def _same(left, right) -> bool:
    try:
        return sp.simplify(left - right) == 0
    except (TypeError, ValueError):
        return False
