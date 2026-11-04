"""Parseo y validación de f(p), de los valores de C y del intervalo gráfico."""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp
from sympy.core.expr import Expr

# p es la única variable libre que el usuario puede escribir.
P = sp.symbols("p")
X = sp.symbols("x")
Y = sp.symbols("y")
C = sp.symbols("C")

_LOCALS = {
    "p": P,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "pi": sp.pi,
    "E": sp.E,
    "abs": sp.Abs,
}


@dataclass
class ParsedInput:
    """Entrada ya validada."""

    f: Expr
    f_text: str
    c_values: list[float]
    a: float
    b: float


@dataclass
class ValidationResult:
    """Resultado de la validación de la entrada."""

    valid: bool
    message: str
    parsed: ParsedInput | None = None


def parse_c_values(c_text: str) -> tuple[list[float] | None, str | None]:
    """Convierte '−3, −2, ..., 3' en una lista de constantes reales."""
    raw = c_text.strip()
    if not raw:
        return None, "Indique al menos un valor de C."

    values: list[float] = []
    for part in raw.replace(";", ",").split(","):
        piece = part.strip()
        if not piece:
            continue
        try:
            value = float(piece)
        except ValueError:
            return None, f"El valor de C «{piece}» no es un número."
        if not math.isfinite(value):
            return None, f"El valor de C «{piece}» no es finito."
        if value not in values:
            values.append(value)

    if not values:
        return None, "Indique al menos un valor de C."
    return values, None


def validate_input(f_text: str, c_text: str, a: float, b: float) -> ValidationResult:
    """
    Valida f(p), la lista de C y el intervalo [a, b].

    Una entrada inválida se reporta como error de validación: no se intenta
    construir una solución singular a partir de texto mal formado.
    """
    text = f_text.strip()
    if not text:
        return ValidationResult(False, "La función f(p) no puede estar vacía.")

    try:
        expr = sp.parse_expr(text, local_dict=_LOCALS)
    except (sp.SympifyError, SyntaxError, TypeError, ValueError) as exc:
        return ValidationResult(False, f"No se pudo interpretar f(p): {exc}")

    if not isinstance(expr, Expr) or isinstance(expr, sp.Rel):
        return ValidationResult(False, "f(p) debe ser una expresión, no una ecuación.")

    extra = expr.free_symbols - {P}
    if extra:
        names = ", ".join(sorted(str(symbol) for symbol in extra))
        return ValidationResult(
            False,
            f"f(p) solo puede depender de p. Aparecen otras variables: {names}.",
        )

    c_values, c_error = parse_c_values(c_text)
    if c_error or c_values is None:
        return ValidationResult(False, c_error or "La lista de C no es válida.")

    if not math.isfinite(a) or not math.isfinite(b):
        return ValidationResult(False, "Los extremos del intervalo deben ser números finitos.")
    if a >= b:
        return ValidationResult(False, "El intervalo debe cumplir a < b.")

    return ValidationResult(
        True,
        "Entrada válida: f depende solo de p, hay al menos un valor de C y a < b.",
        ParsedInput(f=expr, f_text=text, c_values=c_values, a=a, b=b),
    )
