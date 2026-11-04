"""Interfaz Streamlit para ecuaciones diferenciales de Clairaut."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ejemplos import EJEMPLOS
from src.steps import run_analysis

OUTPUT_DIR = Path(__file__).parent / "output"
DEFAULT_C = "-3, -2, -1, 0, 1, 2, 3"

st.set_page_config(
    page_title="Ecuaciones de Clairaut",
    page_icon="📐",
    layout="wide",
)

st.title("Visualizador de ecuaciones diferenciales de Clairaut")
st.markdown(
    "Ingrese una función $f(p)$ para construir la ecuación "
    r"$y = x y' + f(y')$, obtener la familia de rectas "
    r"$y = Cx + f(C)$ y, cuando exista, la solución singular."
)

with st.sidebar:
    st.header("Entrada de datos")

    example_options = {ejemplo.name: ejemplo for ejemplo in EJEMPLOS}
    selected_name = st.selectbox(
        "Ejemplos",
        options=["Personalizado"] + list(example_options.keys()),
    )

    if selected_name != "Personalizado":
        example = example_options[selected_name]
        default_f = example.f_text
        st.info(example.description)
    else:
        default_f = "p**2"

    f_text = st.text_input(
        "f(p)",
        value=default_f,
        key=f"f_{selected_name}",
        help="Ejemplos: p**2, p**3, 1/p, p**2 + 1, sqrt(p)",
    )
    c_text = st.text_input(
        "Valores de C",
        value=DEFAULT_C,
        help="Separe las constantes con comas. El enunciado pide de −3 a 3.",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        a = st.number_input("a (inicio de x)", value=-6.0, format="%.4f")
    with col_b:
        b = st.number_input("b (fin de x)", value=6.0, format="%.4f")

    analyze = st.button("Analizar ecuación", type="primary", use_container_width=True)

st.markdown("---")

_EXPANDED = {
    "Ecuación diferencial",
    "Solución general",
    "Solución singular",
    "Gráfica",
}

if analyze:
    case_label = (
        example_options[selected_name].id
        if selected_name != "Personalizado"
        else "custom"
    )
    with st.spinner("Analizando ecuación..."):
        result = run_analysis(f_text, c_text, float(a), float(b), OUTPUT_DIR, case_label)

    if not result.validation.valid:
        st.error(result.validation.message)
    else:
        st.success("Análisis completado.")
        for title, lines in result.sections.items():
            with st.expander(title, expanded=title in _EXPANDED):
                for line in lines:
                    st.markdown(line)

        if result.plot_path and result.plot_path.exists():
            st.subheader("Familia de soluciones y envolvente")
            st.image(str(result.plot_path), use_container_width=True)
            if result.plot_note:
                st.warning(result.plot_note)
            with open(result.plot_path, "rb") as image_file:
                st.download_button(
                    "Descargar gráfica (PNG)",
                    data=image_file.read(),
                    file_name=result.plot_path.name,
                    mime="image/png",
                )
        elif result.plot_error:
            st.warning(result.plot_error)
else:
    st.info(
        "Seleccione un ejemplo o escriba f(p) en el panel lateral "
        "y pulse **Analizar ecuación**."
    )

st.caption(
    "Aplicación académica desarrollada para el curso de Ecuaciones diferenciales "
    "de la Universidad del Tolima"
)
