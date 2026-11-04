"""Prueba los ejemplos del enunciado de ecuaciones de Clairaut."""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clairaut import analyze_clairaut
from src.parser import C, X, validate_input
from src.steps import run_analysis

OUTPUT = ROOT / "output"
C_TEXT = "-3, -2, -1, 0, 1, 2, 3"
A, B = -6.0, 6.0


def _equal(left, right) -> bool:
    try:
        return sp.simplify(left - right) == 0
    except (TypeError, ValueError):
        return False


def _on_domain(expr, target, symbol_assumption: str) -> bool:
    if symbol_assumption == "positive":
        x_dom = sp.Symbol("x", positive=True)
    else:
        x_dom = sp.Symbol("x", negative=True)
    try:
        return sp.simplify(expr.xreplace({X: x_dom}) - target.xreplace({X: x_dom})) == 0
    except (TypeError, ValueError):
        return False


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f"  FALLO: {message}")


def _check_case(name: str, f_text: str, errors: list[str]) -> None:
    print(f"\n{name}")
    print("-" * 40)
    result = run_analysis(f_text, C_TEXT, A, B, OUTPUT, name)
    if not result.validation.valid or result.clairaut is None:
        _fail(errors, f"{name}: validación {result.validation.message}")
        return

    clairaut = result.clairaut
    print(f"  f' = {clairaut.fp}")
    print(f"  general = {clairaut.general}")
    for branch in clairaut.branches:
        flag = "simbólica" if branch.verified_symbolic else "numérica" if branch.verified_numeric else "NO"
        print(f"  p = {branch.p_of_x}")
        print(f"  y = {branch.y_of_x}  [{flag}]  {branch.domain_note}")
    if clairaut.parametric is not None:
        print(f"  paramétrica x = {clairaut.parametric[0]}, y = {clairaut.parametric[1]}")
    excluded = [status.value for status in clairaut.constants if not status.valid]
    print(f"  C excluidos: {excluded}")
    print(f"  gráfica: {result.plot_path}")

    if result.plot_error or result.plot_path is None or not result.plot_path.exists():
        _fail(errors, f"{name}: no se generó la gráfica ({result.plot_error})")
    elif result.plot_path.stat().st_size < 1000:
        _fail(errors, f"{name}: la gráfica está vacía")

    for branch in clairaut.branches:
        if not branch.verified_symbolic and not branch.verified_numeric:
            _fail(errors, f"{name}: rama sin verificación y = {branch.y_of_x}")


def check_obligatorio(errors: list[str]) -> None:
    _check_case("obligatorio p**2", "p**2", errors)
    parsed = validate_input("p**2", C_TEXT, A, B)
    assert parsed.parsed is not None
    result = analyze_clairaut(parsed.parsed.f, parsed.parsed.c_values)
    expected_general = C * X + C**2
    if not _equal(result.general, expected_general):
        _fail(errors, f"general de p**2 inesperada: {result.general}")
    if len(result.branches) != 1 or not _equal(result.branches[0].y_of_x, -(X**2) / 4):
        _fail(errors, f"singular de p**2 inesperada: {[b.y_of_x for b in result.branches]}")
    if any(not status.valid for status in result.constants):
        _fail(errors, "p**2 no debería excluir ningún C")
    if not result.branches[0].verified_symbolic:
        _fail(errors, "p**2 debería verificarse de forma simbólica")


def check_p2_plus_1(errors: list[str]) -> None:
    _check_case("p**2 + 1", "p**2 + 1", errors)
    parsed = validate_input("p**2 + 1", C_TEXT, A, B)
    assert parsed.parsed is not None
    result = analyze_clairaut(parsed.parsed.f, parsed.parsed.c_values)
    if len(result.branches) != 1 or not _equal(result.branches[0].y_of_x, 1 - (X**2) / 4):
        _fail(errors, f"singular de p**2+1 inesperada: {[b.y_of_x for b in result.branches]}")


def check_inverse(errors: list[str]) -> None:
    _check_case("1/p", "1/p", errors)
    parsed = validate_input("1/p", C_TEXT, A, B)
    assert parsed.parsed is not None
    result = analyze_clairaut(parsed.parsed.f, parsed.parsed.c_values)
    excluded = {status.value for status in result.constants if not status.valid}
    if excluded != {0.0}:
        _fail(errors, f"1/p debería excluir solo C = 0, excluyó {excluded}")
    targets = {2 * sp.sqrt(X), -2 * sp.sqrt(X)}
    matched = set()
    for branch in result.branches:
        for target in targets:
            if _on_domain(branch.y_of_x, target, "positive"):
                matched.add(target)
    if len(result.branches) != 2 or len(matched) != 2:
        _fail(errors, f"ramas de 1/p inesperadas: {[b.y_of_x for b in result.branches]}")


def check_cube(errors: list[str]) -> None:
    _check_case("p**3", "p**3", errors)
    parsed = validate_input("p**3", C_TEXT, A, B)
    assert parsed.parsed is not None
    result = analyze_clairaut(parsed.parsed.f, parsed.parsed.c_values)
    positive = sp.sqrt(3) * (-X) ** sp.Rational(3, 2) / 9
    expected = {2 * positive, -2 * positive}
    if len(result.branches) != 2:
        _fail(errors, f"p**3 debería tener 2 ramas, tiene {len(result.branches)}")
        return
    matched = 0
    for branch in result.branches:
        if any(_equal(branch.y_of_x, target) for target in expected):
            matched += 1
        if not any(sample <= 0 for sample in branch.sample_xs):
            _fail(errors, "p**3 debería ser real para x ≤ 0")
        if any(sample > 1e-8 for sample in branch.sample_xs):
            _fail(errors, f"p**3 no debería ser real para x > 0: {branch.sample_xs}")
    if matched != 2:
        _fail(errors, f"fórmulas de p**3 inesperadas: {[b.y_of_x for b in result.branches]}")


def check_sqrt(errors: list[str]) -> None:
    _check_case("sqrt(p)", "sqrt(p)", errors)
    parsed = validate_input("sqrt(p)", C_TEXT, A, B)
    assert parsed.parsed is not None
    result = analyze_clairaut(parsed.parsed.f, parsed.parsed.c_values)
    excluded = {status.value for status in result.constants if not status.valid}
    if excluded != {-3.0, -2.0, -1.0}:
        _fail(errors, f"sqrt(p) debería excluir C negativos, excluyó {excluded}")
    if not any(status.valid and status.value == 0.0 for status in result.constants):
        _fail(errors, "sqrt(p) debería aceptar C = 0")
    if len(result.branches) != 1:
        _fail(errors, f"sqrt(p) debería tener 1 rama, tiene {len(result.branches)}")
        return
    branch = result.branches[0]
    if not _on_domain(branch.y_of_x, -1 / (4 * X), "negative"):
        _fail(errors, f"envolvente de sqrt(p) inesperada: {branch.y_of_x}")
    if any(sample >= 0 for sample in branch.sample_xs):
        _fail(errors, "la envolvente de sqrt(p) no debe incluir x ≥ 0")


def check_validation(errors: list[str]) -> None:
    print("\nValidación")
    print("-" * 40)
    invalid_interval = validate_input("p**2", C_TEXT, 2.0, 1.0)
    invalid_symbol = validate_input("x + p", C_TEXT, A, B)
    invalid_syntax = validate_input("2 +", C_TEXT, A, B)
    print(f"  símbolo: {invalid_symbol.message}")
    print(f"  sintaxis: {invalid_syntax.message}")
    print(f"  intervalo: {invalid_interval.message}")
    if invalid_symbol.valid or invalid_syntax.valid or invalid_interval.valid:
        _fail(errors, "alguna entrada inválida fue aceptada")


def main() -> None:
    errors: list[str] = []
    check_obligatorio(errors)
    check_p2_plus_1(errors)
    check_inverse(errors)
    check_cube(errors)
    check_sqrt(errors)
    check_validation(errors)
    print("\n" + "=" * 60)
    if errors:
        print(f"{len(errors)} fallo(s):")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print("Todos los ejemplos del enunciado pasaron.")


if __name__ == "__main__":
    main()
