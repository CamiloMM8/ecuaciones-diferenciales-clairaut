"""Ejemplos del enunciado: el obligatorio y las funciones adicionales."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Ejemplo:
    """Función f(p) precargada."""

    id: str
    name: str
    f_text: str
    description: str


EJEMPLOS: list[Ejemplo] = [
    Ejemplo(
        id="obligatorio",
        name="Ejemplo obligatorio — f(p) = p²",
        f_text="p**2",
        description="y = x y' + (y')². Solución general y = Cx + C². Singular y = −x²/4.",
    ),
    Ejemplo(
        id="p3",
        name="f(p) = p³",
        f_text="p**3",
        description="y = x y' + (y')³. La solución singular tiene dos ramas reales para x ≤ 0.",
    ),
    Ejemplo(
        id="inv",
        name="f(p) = 1/p",
        f_text="1/p",
        description="y = x y' + 1/y'. C = 0 no está definido. La envolvente es y = ±2√x, x > 0.",
    ),
    Ejemplo(
        id="p2_1",
        name="f(p) = p² + 1",
        f_text="p**2 + 1",
        description="y = x y' + (y')² + 1. Solución singular y = 1 − x²/4.",
    ),
    Ejemplo(
        id="sqrt",
        name="f(p) = √p",
        f_text="sqrt(p)",
        description="y = x y' + √(y'). Solo C ≥ 0 es real. La envolvente vive en x < 0.",
    ),
]


def get_ejemplo_by_name(name: str) -> Ejemplo | None:
    for ejemplo in EJEMPLOS:
        if ejemplo.name == name:
            return ejemplo
    return None
