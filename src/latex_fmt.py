"""Formateo de LaTeX compatible con Streamlit (delimitadores $...$)."""


def inline(latex: str) -> str:
    """Envuelve una expresión LaTeX para renderizado inline en Markdown."""
    return f"${latex}$"


def labeled(label: str, latex: str) -> str:
    """Línea con etiqueta de texto y fórmula renderizable."""
    return f"{label} {inline(latex)}"


def bold_labeled(label: str, latex: str) -> str:
    """Línea con etiqueta en negrita y fórmula renderizable."""
    return f"**{label}** {inline(latex)}"


def block(latex: str) -> str:
    """Fórmula en renglón propio."""
    return f"$${latex}$$"
