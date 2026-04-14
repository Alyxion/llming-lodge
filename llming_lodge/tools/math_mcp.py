"""MathMCP -- in-process MCP server for mathematical computation.

Provides Wolfram Alpha-like capabilities:
- Expression evaluation (basic + advanced)
- LaTeX input support
- Symbolic math (derivatives, integrals, equation solving, simplification)
- Step-by-step solution display
- Variable storage across conversation
- 2D plotting (functions, parametric, polar)
- 3D plotting (surfaces, parametric 3D)
- Matrix operations, statistics
- Rich Plotly visualizations via document plugin (or __rich_mcp__ fallback)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from llming_models.tools.mcp import InProcessMCPServer

logger = logging.getLogger(__name__)

# Decimal places for plot data — 4 gives sub-pixel accuracy while reducing
# JSON payload by ~60 % compared to full float64 precision.
_PLOT_DECIMALS = 4


def _rnd(arr: Any) -> Any:
    """Round a numpy array for compact JSON serialization.

    Handles arrays containing None/NaN by preserving None in output.
    """
    if not isinstance(arr, np.ndarray):
        return arr
    # For object arrays (contain None) or float arrays with NaN
    if arr.dtype == object:
        # Nested list (e.g. 2D z-matrix with None) — round element-wise
        def _r(v: Any) -> Any:
            if v is None:
                return None
            if isinstance(v, (list, np.ndarray)):
                return [_r(x) for x in v]
            try:
                return round(float(v), _PLOT_DECIMALS)
            except (TypeError, ValueError):
                return None
        return [_r(x) for x in arr]
    # Pure numeric array — fast path
    return np.round(arr, _PLOT_DECIMALS).tolist()

# Thread pool for CPU-bound sympy computations
_math_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="math-mcp")

# Default timeout for computations (seconds)
DEFAULT_TIMEOUT = 15.0
MAX_TIMEOUT = 60.0


def _parse_latex_to_sympy(expr_str: str) -> str:
    """Convert LaTeX notation to sympy-parseable string where possible."""
    import re
    s = expr_str.strip()
    # Remove surrounding $..$ or $$..$$
    s = re.sub(r'^\$\$?(.*?)\$?\$$', r'\1', s, flags=re.DOTALL)
    # \frac{a}{b} -> (a)/(b)
    while r'\frac' in s:
        s = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', s)
    # \sqrt{x} -> sqrt(x)
    s = re.sub(r'\\sqrt\{([^{}]*)\}', r'sqrt(\1)', s)
    # \sin, \cos, etc.
    for fn in ('sin', 'cos', 'tan', 'log', 'ln', 'exp', 'asin', 'acos', 'atan',
               'sinh', 'cosh', 'tanh', 'sec', 'csc', 'cot'):
        s = s.replace(f'\\{fn}', fn)
    # \pi -> pi, \infty -> oo
    s = s.replace(r'\pi', 'pi')
    s = s.replace(r'\infty', 'oo')
    s = s.replace(r'\inf', 'oo')
    # \cdot -> *
    s = s.replace(r'\cdot', '*')
    # \times -> *
    s = s.replace(r'\times', '*')
    # \left, \right -> nothing
    s = s.replace(r'\left', '')
    s = s.replace(r'\right', '')
    # ^ for power (already python-compatible)
    s = s.replace(r'\^', '**')
    # x^{n} -> x**(n)
    s = re.sub(r'\^{([^{}]*)}', r'**(\1)', s)
    # Only transform recognized LaTeX commands (whitelist)
    _LATEX_COMMANDS = {'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'frac', 'pi', 'infty',
                       'alpha', 'beta', 'gamma', 'delta', 'theta', 'lambda', 'sigma', 'omega',
                       'int', 'sum', 'prod', 'lim', 'partial', 'nabla', 'cdot', 'times', 'div',
                       'left', 'right', 'begin', 'end', 'text', 'mathrm',
                       'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'sec', 'csc', 'cot',
                       'inf', 'mu', 'phi', 'rho', 'epsilon', 'zeta', 'eta', 'iota', 'kappa',
                       'nu', 'xi', 'omicron', 'tau', 'upsilon', 'chi', 'psi'}
    s = re.sub(r'\\([a-zA-Z]+)', lambda m: m.group(1) if m.group(1) in _LATEX_COMMANDS else m.group(0), s)
    # Handle implicit multiplication: 2x -> 2*x, x(... -> x*(
    # This is tricky — only do basic cases
    s = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', s)
    return s


def _safe_sympy_env(variables: dict[str, Any]) -> dict[str, Any]:
    """Build a safe evaluation namespace with sympy functions and user variables."""
    import sympy as sp
    env = {
        # Core
        'pi': sp.pi, 'e': sp.E, 'E': sp.E, 'I': sp.I, 'i': sp.I,
        'oo': sp.oo, 'inf': sp.oo, 'nan': sp.nan,
        'true': True, 'false': False,
        # Functions
        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
        'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan, 'atan2': sp.atan2,
        'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
        'sec': sp.sec, 'csc': sp.csc, 'cot': sp.cot,
        'sqrt': sp.sqrt, 'cbrt': sp.cbrt, 'root': sp.root,
        'exp': sp.exp, 'log': sp.log, 'ln': sp.log, 'log10': lambda x: sp.log(x, 10),
        'log2': lambda x: sp.log(x, 2),
        'abs': sp.Abs, 'Abs': sp.Abs, 'sign': sp.sign,
        'floor': sp.floor, 'ceiling': sp.ceiling, 'ceil': sp.ceiling,
        'factorial': sp.factorial, 'binomial': sp.binomial,
        'gcd': sp.gcd, 'lcm': sp.lcm,
        'Min': sp.Min, 'Max': sp.Max, 'min': sp.Min, 'max': sp.Max,
        # Symbolic constructors
        'Symbol': sp.Symbol, 'symbols': sp.symbols,
        'Rational': sp.Rational, 'Integer': sp.Integer, 'Float': sp.Float,
        'Matrix': sp.Matrix, 'eye': sp.eye, 'zeros': sp.zeros, 'ones': sp.ones,
        'Eq': sp.Eq, 'Ne': sp.Ne, 'Lt': sp.Lt, 'Le': sp.Le, 'Gt': sp.Gt, 'Ge': sp.Ge,
        'Sum': sp.Sum, 'Product': sp.Product, 'Integral': sp.Integral,
        'Derivative': sp.Derivative, 'Limit': sp.Limit,
        'Function': sp.Function, 'Piecewise': sp.Piecewise,
        'Poly': sp.Poly, 'series': sp.series,
        # Common symbols (auto-created)
        'x': sp.Symbol('x'), 'y': sp.Symbol('y'), 'z': sp.Symbol('z'),
        't': sp.Symbol('t'), 'n': sp.Symbol('n', integer=True),
        'k': sp.Symbol('k', integer=True),
        'a': sp.Symbol('a'), 'b': sp.Symbol('b'), 'c': sp.Symbol('c'),
        'r': sp.Symbol('r'), 'theta': sp.Symbol('theta'),
        'phi': sp.Symbol('phi'), 'alpha': sp.Symbol('alpha'),
        'beta': sp.Symbol('beta'), 'gamma': sp.Symbol('gamma'),
        'omega': sp.Symbol('omega'), 'sigma': sp.Symbol('sigma'),
        'mu': sp.Symbol('mu'), 'lambda_': sp.Symbol('lambda'),
    }
    # Overlay user variables
    for k, v in variables.items():
        env[k] = v
    return env


_DANGEROUS_PATTERNS = ['__', 'import', 'exec(', 'eval(', 'compile(', 'getattr(', 'setattr(',
                       'globals(', 'locals(', 'vars(', 'dir(', 'open(', 'file(', 'os.', 'sys.',
                       'subprocess', 'builtins']


def _sympy_eval(expr_str: str, variables: dict[str, Any]) -> Any:
    """Parse and evaluate a sympy expression string in a safe namespace."""
    cleaned = _parse_latex_to_sympy(expr_str)

    # Reject expressions containing dangerous patterns
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in cleaned.lower():
            raise ValueError(f"Expression contains forbidden pattern: {pattern}")

    env = _safe_sympy_env(variables)
    # Use sympy's parser for better implicit multiplication handling
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    result = parse_expr(cleaned, local_dict=env, transformations=transformations)
    return result


def _expr_to_latex(expr: Any) -> str:
    """Convert a sympy expression to LaTeX string."""
    import sympy as sp
    try:
        return sp.latex(expr)
    except Exception:
        return str(expr)


def _expr_to_str(expr: Any) -> str:
    """Convert a sympy expression to pretty string."""
    import sympy as sp
    try:
        return sp.pretty(expr, use_unicode=True)
    except Exception:
        return str(expr)


def _build_text_result(title: str, latex: str, result_text: str,
                       steps: list[Any] | None = None, extra_info: dict[Any, Any] | None = None) -> str:
    """Build a plain-text result with LaTeX in $$...$$ delimiters.

    Designed for chat clients that render markdown + KaTeX natively.
    The LLM receives this text and can present it directly.
    """
    parts = [f"**{title}**"]
    if latex:
        parts.append(f"\n$$\n{latex}\n$$")
    if result_text:
        parts.append(f"\n`{result_text}`")
    if steps:
        parts.append("\n**Steps:**")
        for i, step in enumerate(steps):
            step_title = step.get("title", f"Step {i+1}")
            step_expr = step.get("latex", "")
            step_note = step.get("note", "")
            line = f"{i+1}. **{step_title}**"
            if step_expr:
                line += f"  $${step_expr}$$"
            if step_note:
                line += f"  *{step_note}*"
            parts.append(line)
    if extra_info:
        info_str = "  ".join(f"**{k}:** {v}" for k, v in extra_info.items())
        parts.append(f"\n{info_str}")
    return "\n".join(parts)


def _build_rich_result(title: str, latex: str, result_text: str,
                       steps: list[Any] | None = None,
                       extra_info: dict[Any, Any] | None = None) -> str:
    """Build a data-only __rich_mcp__ envelope for math results.

    The envelope contains ONLY data — no HTML, CSS, or JS.
    The client-side ``math_result`` renderer in builtin-plugins.js
    builds the presentation from this data at render time.

    Plots are NOT handled here — they go through the Plotly document
    plugin via ``DocumentSessionStore.create()``.
    """
    # Build LLM summary
    summary_parts = [title]
    if result_text:
        summary_parts.append(result_text)
    if steps:
        for s in steps:
            summary_parts.append(f"  {s.get('title', '')}: {s.get('note', '')}")

    render_data: dict[str, Any] = {
        'type': 'math_result',
        'title': title,
    }
    if latex:
        render_data['latex'] = latex
    if result_text:
        render_data['result_text'] = result_text
    if steps:
        render_data['steps'] = steps
    if extra_info:
        render_data['extra_info'] = extra_info

    envelope = {
        '__rich_mcp__': {
            'version': '1.0',
            'min_viewer_version': '1.0',
            'render': render_data,
            'llm_summary': '\n'.join(summary_parts),
        },
    }
    return json.dumps(envelope)


def _js_escape(s: str) -> str:
    """Escape string for safe insertion into JS string literals."""
    return (s.replace('\\', '\\\\').replace("'", "\\'")
             .replace('"', '\\"').replace('\n', '\\n')
             .replace('<', '&lt;').replace('>', '&gt;'))


def _html_escape(s: str) -> str:
    """Basic HTML escaping."""
    return (s.replace('&', '&amp;').replace('<', '&lt;')
             .replace('>', '&gt;').replace('"', '&quot;'))




class MathMCP(InProcessMCPServer):
    """In-process MCP server providing mathematical computation tools.

    Each chat session gets its own instance with independent variable storage.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT, rich_output: bool = False,
                 document_store: Any = None) -> None:
        """
        Args:
            timeout: Max seconds for each computation.
            rich_output: If True, return ``__rich_mcp__`` HTML envelopes for
                all results. If False (default), return plain text with
                ``$$LaTeX$$`` for non-plot tools — suitable for any chat
                client with markdown + KaTeX support.
            document_store: If provided, plots are created as Plotly documents
                via the document plugin system instead of ``__rich_mcp__``
                envelopes.
        """
        self._variables: dict[str, Any] = {}
        self._timeout = min(timeout, MAX_TIMEOUT)
        self._rich = rich_output
        self._doc_store = document_store

    def _result(self, title: str, latex: str, result_text: str,
                steps: list[Any] | None = None, extra_info: dict[Any, Any] | None = None) -> str:
        """Build a math result using the configured output mode."""
        if self._rich:
            return _build_rich_result(title, latex, result_text,
                                      steps=steps, extra_info=extra_info)
        return _build_text_result(title, latex, result_text,
                                  steps=steps, extra_info=extra_info)

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "math_evaluate",
                "displayName": "Evaluate",
                "displayDescription": "Evaluate a mathematical expression",
                "icon": "calculate",
                "description": (
                    "Evaluate a mathematical expression. Supports arithmetic, algebra, "
                    "trigonometry, logarithms, and more. Input can be standard math notation "
                    "or LaTeX. Returns exact symbolic result when possible, with optional "
                    "numerical approximation. Examples: '2^10 + sqrt(144)', "
                    "'sin(pi/4)', '\\\\frac{3}{7} + \\\\frac{2}{5}', 'sum(1/k**2, (k, 1, 100))'"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate (standard or LaTeX notation)",
                        },
                        "numeric": {
                            "type": "boolean",
                            "description": "If true, also return a floating-point approximation",
                            "default": False,
                        },
                    },
                    "required": ["expression"],
                },
            },
            {
                "name": "math_solve",
                "displayName": "Solve Equation",
                "displayDescription": "Solve equations or systems of equations",
                "icon": "functions",
                "description": (
                    "Solve equations symbolically. Can solve single equations, systems of "
                    "equations, and inequalities. Returns step-by-step solution when possible. "
                    "Examples: 'x**2 - 5*x + 6 = 0', 'x**2 + y**2 = 1, x + y = 1', "
                    "'2*x + 3 > 7'. Use = for equations; expressions without = are assumed = 0."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "equations": {
                            "type": "string",
                            "description": "Equation(s) to solve. Separate multiple equations with commas or semicolons.",
                        },
                        "variables": {
                            "type": "string",
                            "description": "Variable(s) to solve for (comma-separated). Auto-detected if not specified.",
                        },
                    },
                    "required": ["equations"],
                },
            },
            {
                "name": "math_differentiate",
                "displayName": "Differentiate",
                "displayDescription": "Compute symbolic derivatives",
                "icon": "trending_up",
                "description": (
                    "Compute the derivative of an expression with respect to a variable. "
                    "Supports higher-order derivatives. Returns step-by-step differentiation. "
                    "Examples: 'x**3 + 2*x', 'sin(x)*cos(x)', 'exp(-x**2)'"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Expression to differentiate",
                        },
                        "variable": {
                            "type": "string",
                            "description": "Variable to differentiate with respect to (default: x)",
                            "default": "x",
                        },
                        "order": {
                            "type": "integer",
                            "description": "Order of derivative (default: 1)",
                            "default": 1,
                        },
                    },
                    "required": ["expression"],
                },
            },
            {
                "name": "math_integrate",
                "displayName": "Integrate",
                "displayDescription": "Compute symbolic integrals",
                "icon": "area_chart",
                "description": (
                    "Compute the integral of an expression. Supports indefinite and definite "
                    "integrals. Returns step-by-step integration when possible. "
                    "Examples: indefinite 'x**2 + 1', definite with limits 'sin(x)' from 0 to pi."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Expression to integrate",
                        },
                        "variable": {
                            "type": "string",
                            "description": "Integration variable (default: x)",
                            "default": "x",
                        },
                        "lower": {
                            "type": "string",
                            "description": "Lower bound for definite integral (omit for indefinite)",
                        },
                        "upper": {
                            "type": "string",
                            "description": "Upper bound for definite integral (omit for indefinite)",
                        },
                    },
                    "required": ["expression"],
                },
            },
            {
                "name": "math_simplify",
                "displayName": "Simplify",
                "displayDescription": "Simplify or expand mathematical expressions",
                "icon": "compress",
                "description": (
                    "Simplify, expand, or factor a mathematical expression. "
                    "Modes: simplify (default), expand, factor, cancel, trigsimp, "
                    "apart (partial fractions), collect."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Expression to simplify",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["simplify", "expand", "factor", "cancel", "trigsimp", "apart", "collect"],
                            "description": "Simplification mode",
                            "default": "simplify",
                        },
                    },
                    "required": ["expression"],
                },
            },
            {
                "name": "math_limit",
                "displayName": "Limit",
                "displayDescription": "Compute limits of expressions",
                "icon": "swap_horiz",
                "description": (
                    "Compute the limit of an expression as a variable approaches a value. "
                    "Supports limits at infinity. Examples: 'sin(x)/x as x->0', "
                    "'(1+1/n)**n as n->oo'"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Expression to take the limit of",
                        },
                        "variable": {
                            "type": "string",
                            "description": "Variable approaching the point (default: x)",
                            "default": "x",
                        },
                        "point": {
                            "type": "string",
                            "description": "Value the variable approaches (use 'oo' for infinity)",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["+", "-", "+-"],
                            "description": "Direction of approach (default: +-)",
                            "default": "+-",
                        },
                    },
                    "required": ["expression", "point"],
                },
            },
            {
                "name": "math_series",
                "displayName": "Taylor Series",
                "displayDescription": "Compute Taylor/Maclaurin series expansion",
                "icon": "stacked_line_chart",
                "description": (
                    "Compute the Taylor series expansion of an expression around a point. "
                    "Examples: 'exp(x) around 0', 'sin(x) around 0 to order 10'"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Expression to expand",
                        },
                        "variable": {
                            "type": "string",
                            "description": "Expansion variable (default: x)",
                            "default": "x",
                        },
                        "point": {
                            "type": "string",
                            "description": "Expansion point (default: 0 for Maclaurin)",
                            "default": "0",
                        },
                        "order": {
                            "type": "integer",
                            "description": "Number of terms (default: 6)",
                            "default": 6,
                        },
                    },
                    "required": ["expression"],
                },
            },
            {
                "name": "math_matrix",
                "displayName": "Matrix Operations",
                "displayDescription": "Matrix computations (determinant, inverse, eigenvalues, etc.)",
                "icon": "grid_on",
                "description": (
                    "Perform matrix operations. Input matrix as nested list. "
                    "Operations: determinant, inverse, eigenvalues, eigenvectors, rank, "
                    "rref (row echelon), transpose, trace, nullspace, multiply. "
                    "For multiply, provide a second matrix."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "matrix": {
                            "type": "array",
                            "description": "Matrix as nested array, e.g. [[1,2],[3,4]]",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["determinant", "inverse", "eigenvalues", "eigenvectors",
                                     "rank", "rref", "transpose", "trace", "nullspace", "multiply"],
                            "description": "Operation to perform",
                        },
                        "matrix_b": {
                            "type": "array",
                            "description": "Second matrix for multiply operation",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                    },
                    "required": ["matrix", "operation"],
                },
            },
            {
                "name": "math_statistics",
                "displayName": "Statistics",
                "displayDescription": "Compute statistical measures on a dataset",
                "icon": "query_stats",
                "description": (
                    "Compute statistical measures on a list of numbers: mean, median, mode, "
                    "standard deviation, variance, min, max, quartiles, skewness, kurtosis, "
                    "correlation, regression. Can also generate a histogram or box plot."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "description": "List of numbers",
                            "items": {"type": "number"},
                        },
                        "data_y": {
                            "type": "array",
                            "description": "Optional second dataset for correlation/regression",
                            "items": {"type": "number"},
                        },
                        "visualize": {
                            "type": "string",
                            "enum": ["histogram", "boxplot", "scatter", "none"],
                            "description": "Type of visualization to include",
                            "default": "none",
                        },
                    },
                    "required": ["data"],
                },
            },
            {
                "name": "math_plot_2d",
                "displayName": "Plot 2D",
                "displayDescription": "Create 2D function plots",
                "icon": "show_chart",
                "description": (
                    "Plot one or more mathematical functions in 2D. Supports: "
                    "- Cartesian: y = f(x), e.g. 'sin(x)', 'x**2 - 1' "
                    "- Parametric: x(t), y(t) "
                    "- Polar: r(theta) "
                    "- Implicit: f(x,y) = 0 "
                    "Multiple functions can be plotted together."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "functions": {
                            "type": "array",
                            "description": "List of function expressions to plot",
                            "items": {"type": "string"},
                        },
                        "x_range": {
                            "type": "array",
                            "description": "X-axis range [min, max] (default: [-10, 10])",
                            "items": {"type": "number"},
                        },
                        "y_range": {
                            "type": "array",
                            "description": "Y-axis range [min, max] (auto if not specified)",
                            "items": {"type": "number"},
                        },
                        "title": {
                            "type": "string",
                            "description": "Plot title",
                        },
                        "plot_type": {
                            "type": "string",
                            "enum": ["cartesian", "parametric", "polar", "implicit"],
                            "description": "Type of plot (default: cartesian)",
                            "default": "cartesian",
                        },
                        "parametric_range": {
                            "type": "array",
                            "description": "Parameter range [min, max] for parametric/polar plots (default: [0, 2*pi])",
                            "items": {"type": "number"},
                        },
                        "points": {
                            "type": "integer",
                            "description": "Number of sample points (default: 500, max: 1000)",
                            "default": 500,
                        },
                    },
                    "required": ["functions"],
                },
            },
            {
                "name": "math_plot_3d",
                "displayName": "Plot 3D",
                "displayDescription": "Create 3D surface and curve plots",
                "icon": "view_in_ar",
                "description": (
                    "Plot 3D surfaces or curves. Supports: "
                    "- Surface: z = f(x, y), e.g. 'sin(sqrt(x**2 + y**2))' "
                    "- Parametric surface: x(u,v), y(u,v), z(u,v) "
                    "- Parametric curve: x(t), y(t), z(t) "
                    "Multiple surfaces/curves can be plotted together."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "functions": {
                            "type": "array",
                            "description": "Function expression(s). For surface: ['z_expr']. For parametric surface: ['x_expr', 'y_expr', 'z_expr']. For parametric curve: ['x(t)', 'y(t)', 'z(t)']",
                            "items": {"type": "string"},
                        },
                        "x_range": {
                            "type": "array",
                            "description": "X range [min, max] (default: [-5, 5])",
                            "items": {"type": "number"},
                        },
                        "y_range": {
                            "type": "array",
                            "description": "Y range [min, max] (default: [-5, 5])",
                            "items": {"type": "number"},
                        },
                        "title": {
                            "type": "string",
                            "description": "Plot title",
                        },
                        "plot_type": {
                            "type": "string",
                            "enum": ["surface", "parametric_surface", "parametric_curve"],
                            "description": "Type of 3D plot (default: surface)",
                            "default": "surface",
                        },
                        "u_range": {
                            "type": "array",
                            "description": "Parameter u range for parametric plots",
                            "items": {"type": "number"},
                        },
                        "v_range": {
                            "type": "array",
                            "description": "Parameter v range for parametric surface",
                            "items": {"type": "number"},
                        },
                        "points": {
                            "type": "integer",
                            "description": "Grid resolution per axis (default: 40, max: 80)",
                            "default": 40,
                        },
                        "colorscale": {
                            "type": "string",
                            "description": "Plotly colorscale name (default: Viridis)",
                            "default": "Viridis",
                        },
                    },
                    "required": ["functions"],
                },
            },
            {
                "name": "math_set_variable",
                "displayName": "Set Variable",
                "displayDescription": "Store a variable for use in later computations",
                "icon": "bookmark",
                "description": (
                    "Store a named variable with a value for use in subsequent computations. "
                    "The value can be a number, expression, or matrix. Variables persist "
                    "across tool calls within this chat session."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Variable name (e.g. 'f', 'result', 'A')",
                        },
                        "value": {
                            "type": "string",
                            "description": "Value expression (e.g. '3*x + 1', '42', '[[1,2],[3,4]]')",
                        },
                    },
                    "required": ["name", "value"],
                },
            },
            {
                "name": "math_get_variables",
                "displayName": "Get Variables",
                "displayDescription": "List all stored variables",
                "icon": "list",
                "description": "List all variables stored in this session with their current values.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "math_evaluate":
                return await self._evaluate(arguments)
            elif name == "math_solve":
                return await self._solve(arguments)
            elif name == "math_differentiate":
                return await self._differentiate(arguments)
            elif name == "math_integrate":
                return await self._integrate(arguments)
            elif name == "math_simplify":
                return await self._simplify(arguments)
            elif name == "math_limit":
                return await self._limit(arguments)
            elif name == "math_series":
                return await self._series(arguments)
            elif name == "math_matrix":
                return await self._matrix(arguments)
            elif name == "math_statistics":
                return await self._statistics(arguments)
            elif name == "math_plot_2d":
                return await self._plot_2d(arguments)
            elif name == "math_plot_3d":
                return await self._plot_3d(arguments)
            elif name == "math_set_variable":
                return await self._set_variable(arguments)
            elif name == "math_get_variables":
                return await self._get_variables()
            else:
                return json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as e:
            logger.error("[MATH_MCP] Error in %s: %s", name, traceback.format_exc())
            return json.dumps({"error": str(e)})

    async def get_prompt_hints(self) -> list[str]:
        return [
            "You have powerful math tools available. For ANY mathematical computation, "
            "equation solving, plotting, or analysis — always use the math_* tools. "
            "Do NOT compute results manually. The tools handle symbolic math, LaTeX input, "
            "2D/3D plotting, statistics, and matrix operations. Variables set with "
            "math_set_variable persist across the conversation."
        ]

    async def _run_in_thread(self, fn, *args):
        """Run a CPU-bound function in the thread pool with timeout."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(_math_executor, fn, *args),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Computation timed out after {self._timeout}s. "
                "Try simplifying the expression or increasing the timeout."
            )

    # ── Evaluate ──────────────────────────────────────────────

    async def _evaluate(self, args: dict) -> str:
        expr_str = args["expression"]
        want_numeric = args.get("numeric", False)

        def _compute():
            import sympy as sp
            result = _sympy_eval(expr_str, self._variables)
            # Evaluate unevaluated expressions (Sum, Product, Integral, etc.)
            if hasattr(result, 'doit'):
                try:
                    evaluated = result.doit()
                    if evaluated != result:
                        result = evaluated
                except Exception:
                    pass
            # Try to simplify
            try:
                result = sp.simplify(result)
            except Exception:
                pass

            latex = _expr_to_latex(result)
            pretty = _expr_to_str(result)

            info = {"Type": type(result).__name__}
            numeric_val = None
            if want_numeric:
                try:
                    numeric_val = float(result.evalf())
                    info["Numeric"] = f"{numeric_val:.10g}"
                except Exception:
                    try:
                        numeric_val = complex(result.evalf())
                        info["Numeric"] = f"{numeric_val}"
                    except Exception:
                        pass

            return latex, pretty, info

        latex, pretty, info = await self._run_in_thread(_compute)
        return self._result(
            title="Evaluation Result",
            latex=latex,
            result_text=pretty,
            extra_info=info,
        )

    # ── Solve ─────────────────────────────────────────────────

    async def _solve(self, args: dict) -> str:
        eqs_str = args["equations"]
        vars_str = args.get("variables", "")

        def _compute():
            import sympy as sp
            import re

            env = _safe_sympy_env(self._variables)

            # Parse equations
            raw_eqs = re.split(r'[;]|\s*,\s*(?![^(]*\))', eqs_str)
            equations = []
            for eq_str in raw_eqs:
                eq_str = eq_str.strip()
                if not eq_str:
                    continue
                if '=' in eq_str and '==' not in eq_str and '<=' not in eq_str and '>=' not in eq_str:
                    parts = eq_str.split('=', 1)
                    lhs = _sympy_eval(parts[0].strip(), self._variables)
                    rhs = _sympy_eval(parts[1].strip(), self._variables)
                    equations.append(sp.Eq(lhs, rhs))
                else:
                    equations.append(_sympy_eval(eq_str, self._variables))

            # Determine variables to solve for
            if vars_str:
                solve_vars = [env.get(v.strip(), sp.Symbol(v.strip())) for v in vars_str.split(',')]
            else:
                all_syms = set()
                for eq in equations:
                    if hasattr(eq, 'free_symbols'):
                        all_syms |= eq.free_symbols
                solve_vars = sorted(all_syms, key=str)

            # Solve
            if len(equations) == 1:
                solutions = sp.solve(equations[0], solve_vars, dict=True)
            else:
                solutions = sp.solve(equations, solve_vars, dict=True)

            # Build steps
            steps = []
            steps.append({
                "title": "Given",
                "latex": ", \\quad ".join(_expr_to_latex(eq) for eq in equations),
                "note": f"Solving for {', '.join(str(v) for v in solve_vars)}",
            })

            # Format solutions
            if isinstance(solutions, list) and solutions:
                for i, sol in enumerate(solutions):
                    if isinstance(sol, dict):
                        sol_latex = ", \\quad ".join(
                            f"{_expr_to_latex(k)} = {_expr_to_latex(v)}"
                            for k, v in sol.items()
                        )
                        steps.append({
                            "title": f"Solution {i+1}" if len(solutions) > 1 else "Solution",
                            "latex": sol_latex,
                        })
                    else:
                        steps.append({
                            "title": f"Solution {i+1}" if len(solutions) > 1 else "Solution",
                            "latex": _expr_to_latex(sol),
                        })
            elif isinstance(solutions, dict):
                sol_latex = ", \\quad ".join(
                    f"{_expr_to_latex(k)} = {_expr_to_latex(v)}"
                    for k, v in solutions.items()
                )
                steps.append({"title": "Solution", "latex": sol_latex})
            else:
                steps.append({"title": "Result", "latex": _expr_to_latex(solutions)})

            result_str = _expr_to_str(solutions)
            return steps, result_str, {"Solutions": len(solutions) if isinstance(solutions, list) else 1}

        steps, result_str, info = await self._run_in_thread(_compute)
        return self._result(
            title="Equation Solver",
            latex="",
            result_text=result_str,
            steps=steps,
            extra_info=info,
        )

    # ── Differentiate ─────────────────────────────────────────

    async def _differentiate(self, args: dict) -> str:
        expr_str = args["expression"]
        var_str = args.get("variable", "x")
        order = args.get("order", 1)

        def _compute():
            import sympy as sp
            expr = _sympy_eval(expr_str, self._variables)
            var = _safe_sympy_env(self._variables).get(var_str, sp.Symbol(var_str))

            steps = [{
                "title": "Original expression",
                "latex": _expr_to_latex(expr),
            }]

            current = expr
            for i in range(order):
                derivative = sp.diff(current, var)
                derivative = sp.simplify(derivative)
                ord_label = {1: "1st", 2: "2nd", 3: "3rd"}.get(i + 1, f"{i+1}th")
                steps.append({
                    "title": f"{ord_label} derivative" if order > 1 else "Derivative",
                    "latex": f"\\frac{{d{''.join(['^{'+str(i+1)+'}'] if i > 0 else [])}}}{{d{_expr_to_latex(var)}{''.join(['^{'+str(i+1)+'}'] if i > 0 else [])}}} = {_expr_to_latex(derivative)}",
                })
                current = derivative

            result_latex = _expr_to_latex(current)
            result_str = _expr_to_str(current)
            return result_latex, result_str, steps

        result_latex, result_str, steps = await self._run_in_thread(_compute)
        return self._result(
            title="Differentiation",
            latex=result_latex,
            result_text=result_str,
            steps=steps,
        )

    # ── Integrate ─────────────────────────────────────────────

    async def _integrate(self, args: dict) -> str:
        expr_str = args["expression"]
        var_str = args.get("variable", "x")
        lower = args.get("lower")
        upper = args.get("upper")

        def _compute():
            import sympy as sp
            expr = _sympy_eval(expr_str, self._variables)
            env = _safe_sympy_env(self._variables)
            var = env.get(var_str, sp.Symbol(var_str))

            steps = [{
                "title": "Integrand",
                "latex": _expr_to_latex(expr),
            }]

            is_definite = lower is not None and upper is not None
            if is_definite:
                lo = _sympy_eval(lower, self._variables)
                hi = _sympy_eval(upper, self._variables)
                result = sp.integrate(expr, (var, lo, hi))
                steps.append({
                    "title": "Definite integral",
                    "latex": f"\\int_{{{_expr_to_latex(lo)}}}^{{{_expr_to_latex(hi)}}} {_expr_to_latex(expr)} \\, d{_expr_to_latex(var)} = {_expr_to_latex(result)}",
                })
                info = {"Type": "Definite"}
                try:
                    info["Numeric"] = f"{float(result.evalf()):.10g}"
                except Exception:
                    pass
            else:
                result = sp.integrate(expr, var)
                steps.append({
                    "title": "Antiderivative",
                    "latex": f"\\int {_expr_to_latex(expr)} \\, d{_expr_to_latex(var)} = {_expr_to_latex(result)} + C",
                })
                info = {"Type": "Indefinite"}

            result_latex = _expr_to_latex(result)
            result_str = _expr_to_str(result)
            return result_latex, result_str, steps, info

        result_latex, result_str, steps, info = await self._run_in_thread(_compute)
        return self._result(
            title="Integration",
            latex=result_latex,
            result_text=result_str,
            steps=steps,
            extra_info=info,
        )

    # ── Simplify ──────────────────────────────────────────────

    async def _simplify(self, args: dict) -> str:
        expr_str = args["expression"]
        mode = args.get("mode", "simplify")

        def _compute():
            import sympy as sp
            expr = _sympy_eval(expr_str, self._variables)

            steps = [{
                "title": "Original",
                "latex": _expr_to_latex(expr),
            }]

            ops = {
                "simplify": sp.simplify,
                "expand": sp.expand,
                "factor": sp.factor,
                "cancel": sp.cancel,
                "trigsimp": sp.trigsimp,
                "apart": sp.apart,
                "collect": lambda e: sp.collect(e, list(e.free_symbols)[:1]) if e.free_symbols else e,
            }
            fn = ops.get(mode, sp.simplify)
            result = fn(expr)

            steps.append({
                "title": mode.capitalize(),
                "latex": _expr_to_latex(result),
            })

            return _expr_to_latex(result), _expr_to_str(result), steps

        result_latex, result_str, steps = await self._run_in_thread(_compute)
        return self._result(
            title=f"Simplification ({mode})",
            latex=result_latex,
            result_text=result_str,
            steps=steps,
        )

    # ── Limit ─────────────────────────────────────────────────

    async def _limit(self, args: dict) -> str:
        expr_str = args["expression"]
        var_str = args.get("variable", "x")
        point_str = args["point"]
        direction = args.get("direction", "+-")

        def _compute():
            import sympy as sp
            expr = _sympy_eval(expr_str, self._variables)
            env = _safe_sympy_env(self._variables)
            var = env.get(var_str, sp.Symbol(var_str))
            point = _sympy_eval(point_str, self._variables)

            result = sp.limit(expr, var, point, direction)

            steps = [
                {
                    "title": "Expression",
                    "latex": _expr_to_latex(expr),
                },
                {
                    "title": "Limit",
                    "latex": f"\\lim_{{{_expr_to_latex(var)} \\to {_expr_to_latex(point)}}} {_expr_to_latex(expr)} = {_expr_to_latex(result)}",
                },
            ]

            return _expr_to_latex(result), _expr_to_str(result), steps

        result_latex, result_str, steps = await self._run_in_thread(_compute)
        return self._result(
            title="Limit",
            latex=result_latex,
            result_text=result_str,
            steps=steps,
        )

    # ── Series ────────────────────────────────────────────────

    async def _series(self, args: dict) -> str:
        expr_str = args["expression"]
        var_str = args.get("variable", "x")
        point_str = args.get("point", "0")
        order = args.get("order", 6)

        def _compute():
            import sympy as sp
            expr = _sympy_eval(expr_str, self._variables)
            env = _safe_sympy_env(self._variables)
            var = env.get(var_str, sp.Symbol(var_str))
            point = _sympy_eval(point_str, self._variables)

            expansion = sp.series(expr, var, point, n=order)
            # Remove O() term for clean display
            expansion_no_o = expansion.removeO()

            steps = [
                {
                    "title": "Function",
                    "latex": _expr_to_latex(expr),
                },
                {
                    "title": f"Taylor expansion around {_expr_to_latex(point)} (order {order})",
                    "latex": _expr_to_latex(expansion),
                },
                {
                    "title": "Polynomial approximation",
                    "latex": _expr_to_latex(expansion_no_o),
                },
            ]

            return _expr_to_latex(expansion), _expr_to_str(expansion), steps

        result_latex, result_str, steps = await self._run_in_thread(_compute)
        return self._result(
            title="Taylor Series",
            latex=result_latex,
            result_text=result_str,
            steps=steps,
        )

    # ── Matrix ────────────────────────────────────────────────

    async def _matrix(self, args: dict) -> str:
        matrix_data = args["matrix"]
        operation = args["operation"]
        matrix_b = args.get("matrix_b")

        def _compute():
            import sympy as sp
            M = sp.Matrix(matrix_data)

            steps = [{
                "title": "Input matrix",
                "latex": _expr_to_latex(M),
            }]

            if operation == "determinant":
                result = M.det()
                steps.append({"title": "Determinant", "latex": f"\\det(A) = {_expr_to_latex(result)}"})
            elif operation == "inverse":
                result = M.inv()
                steps.append({"title": "Inverse", "latex": f"A^{{-1}} = {_expr_to_latex(result)}"})
            elif operation == "eigenvalues":
                result = M.eigenvals()
                ev_parts = [f"\\lambda = {_expr_to_latex(k)} \\text{{ (mult. {v})}}" for k, v in result.items()]
                steps.append({"title": "Eigenvalues", "latex": ", \\quad ".join(ev_parts)})
            elif operation == "eigenvectors":
                result = M.eigenvects()
                for eigenval, mult, vecs in result:
                    vec_latex = ", ".join(_expr_to_latex(v) for v in vecs)
                    steps.append({
                        "title": f"Eigenvalue {_expr_to_latex(eigenval)} (mult. {mult})",
                        "latex": vec_latex,
                    })
            elif operation == "rank":
                result = M.rank()
                steps.append({"title": "Rank", "latex": str(result)})
            elif operation == "rref":
                result, pivots = M.rref()
                steps.append({"title": "Row echelon form", "latex": _expr_to_latex(result)})
                steps.append({"title": "Pivot columns", "note": str(list(pivots))})
            elif operation == "transpose":
                result = M.T
                steps.append({"title": "Transpose", "latex": _expr_to_latex(result)})
            elif operation == "trace":
                result = M.trace()
                steps.append({"title": "Trace", "latex": f"\\text{{tr}}(A) = {_expr_to_latex(result)}"})
            elif operation == "nullspace":
                result = M.nullspace()
                for i, v in enumerate(result):
                    steps.append({"title": f"Null vector {i+1}", "latex": _expr_to_latex(v)})
            elif operation == "multiply" and matrix_b:
                B = sp.Matrix(matrix_b)
                result = M * B
                steps.append({"title": "Matrix B", "latex": _expr_to_latex(B)})
                steps.append({"title": "Product A*B", "latex": _expr_to_latex(result)})
            else:
                result = str(M)

            return _expr_to_str(result), steps, {"Dimensions": f"{M.rows}x{M.cols}"}

        result_str, steps, info = await self._run_in_thread(_compute)
        return self._result(
            title=f"Matrix: {operation.replace('_', ' ').title()}",
            latex="",
            result_text=result_str,
            steps=steps,
            extra_info=info,
        )

    # ── Statistics ────────────────────────────────────────────

    async def _statistics(self, args: dict) -> str:
        data = args["data"]
        data_y = args.get("data_y")
        visualize = args.get("visualize", "none")

        def _compute():
            from scipy import stats as scipy_stats
            arr = np.array(data, dtype=float)

            result = {
                "Count": len(arr),
                "Mean": float(np.mean(arr)),
                "Median": float(np.median(arr)),
                "Std Dev": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0,
                "Variance": float(np.var(arr, ddof=1)) if len(arr) > 1 else 0,
                "Min": float(np.min(arr)),
                "Max": float(np.max(arr)),
                "Range": float(np.ptp(arr)),
                "Q1": float(np.percentile(arr, 25)),
                "Q3": float(np.percentile(arr, 75)),
                "IQR": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            }
            try:
                mode_result = scipy_stats.mode(arr, keepdims=True)
                result["Mode"] = float(mode_result.mode[0])
            except Exception:
                pass
            if len(arr) > 2:
                result["Skewness"] = float(scipy_stats.skew(arr))
                result["Kurtosis"] = float(scipy_stats.kurtosis(arr))

            if data_y and len(data_y) == len(data):
                arr_y = np.array(data_y, dtype=float)
                corr, p_val = scipy_stats.pearsonr(arr, arr_y)
                result["Correlation (r)"] = float(corr)
                result["p-value"] = float(p_val)
                slope, intercept, r_val, _, std_err = scipy_stats.linregress(arr, arr_y)
                result["Regression slope"] = float(slope)
                result["Regression intercept"] = float(intercept)
                result["R-squared"] = float(r_val**2)

            # Build Plotly visualization
            plotly_json = None
            if visualize == "histogram":
                plotly_json = {
                    "data": [{"x": data, "type": "histogram", "marker": {"color": "rgba(99,102,241,0.7)"}}],
                    "layout": {
                        "title": {"text": "Histogram"},
                        "xaxis": {"title": {"text": "Value"}},
                        "yaxis": {"title": {"text": "Frequency"}},
                    },
                }
            elif visualize == "boxplot":
                plotly_json = {
                    "data": [{"y": data, "type": "box", "marker": {"color": "#6366f1"}, "boxpoints": "outliers"}],
                    "layout": {"title": {"text": "Box Plot"}},
                }
            elif visualize == "scatter" and data_y:
                traces = [{"x": data, "y": data_y, "mode": "markers", "type": "scatter",
                           "marker": {"color": "#6366f1", "size": 6}}]
                if "Regression slope" in result:
                    x_line = [min(data), max(data)]
                    y_line = [result["Regression slope"]*x + result["Regression intercept"] for x in x_line]
                    traces.append({"x": x_line, "y": y_line, "mode": "lines", "type": "scatter",
                                   "line": {"color": "#ec4899", "width": 2}, "name": "Regression"})
                plotly_json = {
                    "data": traces,
                    "layout": {
                        "title": {"text": "Scatter Plot"},
                        "xaxis": {"title": {"text": "X"}},
                        "yaxis": {"title": {"text": "Y"}},
                    },
                }

            return result, plotly_json

        result, plotly_json = await self._run_in_thread(_compute)

        info = {k: f"{v:.6g}" if isinstance(v, float) else str(v) for k, v in result.items()}
        result_text = "\n".join(f"{k}: {v}" for k, v in info.items())

        if plotly_json:
            result = {"status": "chart_created", "name": "Statistical Analysis", "stats": result_text}
            if self._doc_store:
                doc = self._doc_store.create(type="plotly", name="Statistical Analysis", data=plotly_json)
                result["document_id"] = doc.id
                result["__inline_doc__"] = {"type": "plotly", "name": "Statistical Analysis", "data": plotly_json, "id": doc.id}
            return json.dumps(result)
        return self._result(
            title="Statistical Analysis",
            latex="",
            result_text=result_text,
            extra_info={"N": len(data)},
        )

    # ── Plot 2D ───────────────────────────────────────────────

    async def _plot_2d(self, args: dict) -> str:
        functions = args["functions"]
        x_range = args.get("x_range", [-10, 10])
        y_range = args.get("y_range")
        title = args.get("title", "")
        plot_type = args.get("plot_type", "cartesian")
        param_range = args.get("parametric_range", [0, 2 * math.pi])
        n_points = min(args.get("points", 500), 1000)

        def _compute():
            import sympy as sp
            env = _safe_sympy_env(self._variables)
            traces = []
            colors = ['#6366f1', '#ec4899', '#22c55e', '#f59e0b', '#8b5cf6', '#06b6d4', '#f43f5e', '#84cc16']

            if plot_type == "cartesian":
                x_vals = np.linspace(x_range[0], x_range[1], n_points)
                for i, fn_str in enumerate(functions):
                    expr = _sympy_eval(fn_str, self._variables)
                    x_sym = env['x']
                    f_np = sp.lambdify(x_sym, expr, modules=['numpy'])
                    try:
                        y_vals = f_np(x_vals)
                        y_vals = np.where(np.isfinite(y_vals), y_vals, None)
                    except Exception:
                        y_vals = [None] * len(x_vals)
                    traces.append({
                        "x": _rnd(x_vals),
                        "y": _rnd(y_vals),
                        "type": "scatter",
                        "mode": "lines",
                        "name": fn_str,
                        "line": {"color": colors[i % len(colors)], "width": 2},
                    })

            elif plot_type == "parametric":
                t_vals = np.linspace(param_range[0], param_range[1], n_points)
                t_sym = env['t']
                for i in range(0, len(functions), 2):
                    x_expr = _sympy_eval(functions[i], self._variables)
                    y_expr = _sympy_eval(functions[i+1] if i+1 < len(functions) else '0', self._variables)
                    x_np = sp.lambdify(t_sym, x_expr, modules=['numpy'])
                    y_np = sp.lambdify(t_sym, y_expr, modules=['numpy'])
                    x_vals = x_np(t_vals)
                    y_vals = y_np(t_vals)
                    traces.append({
                        "x": _rnd(np.where(np.isfinite(x_vals), x_vals, None)),
                        "y": _rnd(np.where(np.isfinite(y_vals), y_vals, None)),
                        "type": "scatter",
                        "mode": "lines",
                        "name": f"({functions[i]}, {functions[i+1] if i+1 < len(functions) else '0'})",
                        "line": {"color": colors[(i//2) % len(colors)], "width": 2},
                    })

            elif plot_type == "polar":
                theta_vals = np.linspace(param_range[0], param_range[1], n_points)
                for i, fn_str in enumerate(functions):
                    expr = _sympy_eval(fn_str, self._variables)
                    theta_sym = env['theta']
                    r_np = sp.lambdify(theta_sym, expr, modules=['numpy'])
                    r_vals = r_np(theta_vals)
                    traces.append({
                        "r": _rnd(np.where(np.isfinite(r_vals), np.abs(r_vals), None)),
                        "theta": _rnd(np.degrees(theta_vals)),
                        "type": "scatterpolar",
                        "mode": "lines",
                        "name": fn_str,
                        "line": {"color": colors[i % len(colors)], "width": 2},
                    })

            elif plot_type == "implicit":
                x_vals = np.linspace(x_range[0], x_range[1], n_points)
                y_lo = y_range[0] if y_range else x_range[0]
                y_hi = y_range[1] if y_range else x_range[1]
                y_vals = np.linspace(y_lo, y_hi, n_points)
                X, Y = np.meshgrid(x_vals, y_vals)
                for i, fn_str in enumerate(functions):
                    expr = _sympy_eval(fn_str, self._variables)
                    x_sym, y_sym = env['x'], env['y']
                    f_np = sp.lambdify((x_sym, y_sym), expr, modules=['numpy'])
                    Z = f_np(X, Y)
                    traces.append({
                        "x": _rnd(x_vals),
                        "y": _rnd(y_vals),
                        "z": _rnd(Z),
                        "type": "contour",
                        "contours": {"start": 0, "end": 0, "size": 0.01},
                        "showscale": False,
                        "line": {"color": colors[i % len(colors)], "width": 2},
                        "name": fn_str,
                    })

            layout = {
                "title": {"text": title or ", ".join(functions)},
                "showlegend": len(functions) > 1,
            }
            if plot_type != "polar":
                layout["xaxis"] = {"title": {"text": "x"}}
                layout["yaxis"] = {"title": {"text": "y"}, "scaleanchor": None}
                if y_range:
                    layout["yaxis"]["range"] = y_range

            return {"data": traces, "layout": layout}

        plotly_json = await self._run_in_thread(_compute)
        _title = title or "2D Plot"
        result = {"status": "chart_created", "name": _title}
        if self._doc_store:
            doc = self._doc_store.create(type="plotly", name=_title, data=plotly_json)
            result["document_id"] = doc.id
            result["__inline_doc__"] = {"type": "plotly", "name": _title, "data": plotly_json, "id": doc.id}
        return json.dumps(result)

    # ── Plot 3D ───────────────────────────────────────────────

    async def _plot_3d(self, args: dict) -> str:
        functions = args["functions"]
        x_range = args.get("x_range", [-5, 5])
        y_range = args.get("y_range", [-5, 5])
        title = args.get("title", "")
        plot_type = args.get("plot_type", "surface")
        u_range = args.get("u_range", [0, 2 * math.pi])
        v_range = args.get("v_range", [0, math.pi])
        n_points = min(args.get("points", 40), 80)
        colorscale = args.get("colorscale", "Viridis")

        def _compute():
            import sympy as sp
            env = _safe_sympy_env(self._variables)
            traces = []

            if plot_type == "surface":
                x_vals = np.linspace(x_range[0], x_range[1], n_points)
                y_vals = np.linspace(y_range[0], y_range[1], n_points)
                X, Y = np.meshgrid(x_vals, y_vals)

                for fn_str in functions:
                    expr = _sympy_eval(fn_str, self._variables)
                    x_sym, y_sym = env['x'], env['y']
                    f_np = sp.lambdify((x_sym, y_sym), expr, modules=['numpy'])
                    try:
                        Z = f_np(X, Y)
                        Z = np.where(np.isfinite(Z), Z, None)
                    except Exception:
                        Z = np.full_like(X, None)
                    traces.append({
                        "x": _rnd(x_vals),
                        "y": _rnd(y_vals),
                        "z": _rnd(Z),
                        "type": "surface",
                        "colorscale": colorscale,
                        "name": fn_str,
                    })

            elif plot_type == "parametric_surface":
                u_vals = np.linspace(u_range[0], u_range[1], n_points)
                v_vals = np.linspace(v_range[0], v_range[1], n_points)
                U, V = np.meshgrid(u_vals, v_vals)

                # Expect 3 expressions: x(u,v), y(u,v), z(u,v)
                if len(functions) >= 3:
                    u_sym, v_sym = sp.Symbol('u'), sp.Symbol('v')
                    env['u'] = u_sym
                    env['v'] = v_sym
                    x_expr = _sympy_eval(functions[0], self._variables)
                    y_expr = _sympy_eval(functions[1], self._variables)
                    z_expr = _sympy_eval(functions[2], self._variables)
                    fx = sp.lambdify((u_sym, v_sym), x_expr, modules=['numpy'])
                    fy = sp.lambdify((u_sym, v_sym), y_expr, modules=['numpy'])
                    fz = sp.lambdify((u_sym, v_sym), z_expr, modules=['numpy'])
                    traces.append({
                        "x": _rnd(fx(U, V)),
                        "y": _rnd(fy(U, V)),
                        "z": _rnd(fz(U, V)),
                        "type": "surface",
                        "colorscale": colorscale,
                    })

            elif plot_type == "parametric_curve":
                t_vals = np.linspace(u_range[0], u_range[1], n_points * 10)
                t_sym = env['t']

                if len(functions) >= 3:
                    x_expr = _sympy_eval(functions[0], self._variables)
                    y_expr = _sympy_eval(functions[1], self._variables)
                    z_expr = _sympy_eval(functions[2], self._variables)
                    fx = sp.lambdify(t_sym, x_expr, modules=['numpy'])
                    fy = sp.lambdify(t_sym, y_expr, modules=['numpy'])
                    fz = sp.lambdify(t_sym, z_expr, modules=['numpy'])
                    traces.append({
                        "x": _rnd(fx(t_vals)),
                        "y": _rnd(fy(t_vals)),
                        "z": _rnd(fz(t_vals)),
                        "type": "scatter3d",
                        "mode": "lines",
                        "line": {"color": "#6366f1", "width": 3},
                    })

            layout = {
                "title": {"text": title or ", ".join(functions[:2]) + ("..." if len(functions) > 2 else "")},
                "scene": {
                    "xaxis": {"title": {"text": "x"}},
                    "yaxis": {"title": {"text": "y"}},
                    "zaxis": {"title": {"text": "z"}},
                },
            }

            return {"data": traces, "layout": layout}

        plotly_json = await self._run_in_thread(_compute)
        _title = title or "3D Plot"
        result = {"status": "chart_created", "name": _title}
        if self._doc_store:
            doc = self._doc_store.create(type="plotly", name=_title, data=plotly_json)
            result["document_id"] = doc.id
            result["__inline_doc__"] = {"type": "plotly", "name": _title, "data": plotly_json, "id": doc.id}
        return json.dumps(result)

    # ── Variables ─────────────────────────────────────────────

    async def _set_variable(self, args: dict) -> str:
        import keyword
        import re as _re
        name = args["name"]
        if not _re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            return "Error: variable name must be alphanumeric"
        if name.startswith('__') or name.endswith('__'):
            return "Error: dunder names are not allowed"
        if keyword.iskeyword(name) or name in ('exec', 'eval', 'import', 'compile', 'getattr', 'setattr', 'delattr', 'globals', 'locals', 'vars', 'dir', 'type', 'object'):
            return "Error: reserved name"
        value_str = args["value"]

        def _compute():
            val = _sympy_eval(value_str, self._variables)
            self._variables[name] = val
            return _expr_to_str(val), _expr_to_latex(val)

        pretty, latex = await self._run_in_thread(_compute)
        return self._result(
            title=f"Variable '{name}' set",
            latex=f"{name} = {latex}",
            result_text=pretty,
            extra_info={"Variables stored": len(self._variables)},
        )

    async def _get_variables(self) -> str:
        if not self._variables:
            return json.dumps({"message": "No variables stored in this session."})

        items = {}
        for k, v in self._variables.items():
            items[k] = _expr_to_str(v)

        return json.dumps({
            "variables": items,
            "count": len(items),
        })
