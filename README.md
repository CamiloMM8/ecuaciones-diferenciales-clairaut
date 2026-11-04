# Ecuaciones diferenciales de Clairaut

Visualizador para ecuaciones de Clairaut de la forma

\[
y = x y' + f(y').
\]

El programa construye la ecuación a partir de \(f(p)\), muestra el método paso a paso, obtiene la solución general \(y = Cx + f(C)\), busca la solución singular y grafica la familia junto con su envolvente.

Desarrollado para la asignatura de Ecuaciones Diferenciales — Ingeniería de Sistemas.

## Requisitos

- Python 3.10 o superior
- Dependencias listadas en `requirements.txt`

## Instalación

```powershell
python -m pip install -r requirements.txt
```

## Ejecución

```powershell
streamlit run app.py
```

Se abrirá una interfaz web en el navegador (por defecto `http://localhost:8501`).

## Uso

1. Escriba la función `f(p)`, o elija uno de los ejemplos del enunciado.
2. Indique los valores de `C` (por defecto `-3, -2, -1, 0, 1, 2, 3`) y el intervalo horizontal `[a, b]`.
3. Pulse **Analizar ecuación**.

Sintaxis de la función: `p**2`, `p**3`, `1/p`, `p**2 + 1`, `sqrt(p)`.

El programa mostrará paso a paso:

- Validación de la entrada
- Ecuación \(y = x y' + f(y')\)
- Reescritura \(y = xp + f(p)\)
- Solución general \(y = Cx + f(C)\)
- Derivada \(f'(p)\)
- Condición \(x + f'(p) = 0\)
- Despeje de \(p\), cuando sea posible
- Sustitución en \(y = xp + f(p)\)
- Solución singular, o el motivo por el que no existe
- Restricciones de dominio y valores de \(C\) que no son reales
- Gráfica de la familia y de la envolvente

Las gráficas se guardan en la carpeta `output/`.

## Ejemplos del enunciado

| Ejemplo | f(p) | Qué conviene observar |
|---------|------|------------------------|
| Obligatorio | `p**2` | General \(y = Cx + C^2\). Singular \(y = -x^2/4\). |
| Adicional | `p**3` | Dos ramas reales solo para \(x \le 0\). |
| Adicional | `1/p` | Se omite \(C = 0\). Envolvente \(y = \pm 2\sqrt{x}\), \(x > 0\). |
| Adicional | `p**2 + 1` | Singular \(y = 1 - x^2/4\). |
| Adicional | `sqrt(p)` | Solo \(C \ge 0\). Envolvente real para \(x < 0\). |

## Pruebas

Desde la carpeta del proyecto:

```powershell
python test_ejemplos.py
```

## Estructura del proyecto

```
ecuaciones-diferenciales-clairaut/
├── app.py              # Interfaz Streamlit
├── requirements.txt
├── README.md
├── test_ejemplos.py    # Ejemplos del enunciado
├── src/
│   ├── parser.py       # Entrada y validación
│   ├── clairaut.py     # Método de Clairaut
│   ├── plotting.py     # Familia y envolvente
│   ├── steps.py        # Pasos que ve el usuario
│   ├── latex_fmt.py    # Fórmulas en la interfaz
│   └── ejemplos.py     # Ejemplos precargados
└── output/             # Gráficas exportadas
```
