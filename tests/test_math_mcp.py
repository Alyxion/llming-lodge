"""Tests for the MathMCP in-process server.

Covers all helper functions and all 13 tool endpoints.
"""

import asyncio
import json
import math

import pytest

from llming_models.tools.math_mcp import (
    MathMCP,
    _parse_latex_to_sympy,
    _safe_sympy_env,
    _sympy_eval,
    _expr_to_latex,
    _expr_to_str,
    _build_rich_result,
    _build_text_result,
    _js_escape,
    _html_escape,
)


# ── Helpers ───────────────────────────────────────────────────────

def _parse(result_str: str) -> dict:
    """Parse a tool result JSON string and return the __rich_mcp__ dict."""
    data = json.loads(result_str)
    if "__rich_mcp__" in data:
        return data["__rich_mcp__"]
    return data


def _run(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


@pytest.fixture
def mcp():
    """MathMCP with rich_output=True for testing rich card rendering."""
    return MathMCP(timeout=10.0, rich_output=True)


@pytest.fixture
def mcp_text():
    """MathMCP with rich_output=False (default) — plain markdown+LaTeX."""
    return MathMCP(timeout=10.0)


# ══════════════════════════════════════════════════════════════════
# _parse_latex_to_sympy
# ══════════════════════════════════════════════════════════════════


class TestParseLatex:

    def test_frac(self):
        assert "(3)/(7)" in _parse_latex_to_sympy(r"\frac{3}{7}")

    def test_nested_frac(self):
        result = _parse_latex_to_sympy(r"\frac{\frac{1}{2}}{3}")
        # Inner frac first, then outer
        assert "(" in result and "/" in result

    def test_sqrt(self):
        assert "sqrt(x)" in _parse_latex_to_sympy(r"\sqrt{x}")

    def test_trig_functions(self):
        assert "sin" in _parse_latex_to_sympy(r"\sin(x)")
        assert "cos" in _parse_latex_to_sympy(r"\cos(x)")
        assert "tan" in _parse_latex_to_sympy(r"\tan(x)")

    def test_pi(self):
        assert "pi" in _parse_latex_to_sympy(r"\pi")

    def test_infinity(self):
        assert "oo" in _parse_latex_to_sympy(r"\infty")

    def test_cdot_and_times(self):
        assert "*" in _parse_latex_to_sympy(r"3 \cdot 4")
        assert "*" in _parse_latex_to_sympy(r"3 \times 4")

    def test_left_right_stripped(self):
        result = _parse_latex_to_sympy(r"\left(\frac{1}{2}\right)")
        assert r"\left" not in result
        assert r"\right" not in result

    def test_power_braces(self):
        result = _parse_latex_to_sympy(r"x^{2}")
        assert "**(2)" in result

    def test_implicit_multiplication(self):
        result = _parse_latex_to_sympy("2x")
        assert "2*x" in result

    def test_dollar_signs_stripped(self):
        result = _parse_latex_to_sympy("$x + 1$")
        assert "$" not in result
        assert "x" in result

    def test_double_dollar_signs(self):
        result = _parse_latex_to_sympy("$$x + 1$$")
        assert "$" not in result

    def test_plain_expression_unchanged(self):
        assert _parse_latex_to_sympy("x**2 + 1") == "x**2 + 1"


# ══════════════════════════════════════════════════════════════════
# _safe_sympy_env
# ══════════════════════════════════════════════════════════════════


class TestSafeEnv:

    def test_contains_common_symbols(self):
        env = _safe_sympy_env({})
        assert "x" in env
        assert "y" in env
        assert "z" in env
        assert "pi" in env
        assert "sin" in env

    def test_user_variables_override(self):
        env = _safe_sympy_env({"x": 42})
        assert env["x"] == 42

    def test_integer_symbols(self):
        import sympy as sp
        env = _safe_sympy_env({})
        assert env["n"].is_integer is True
        assert env["k"].is_integer is True


# ══════════════════════════════════════════════════════════════════
# _sympy_eval
# ══════════════════════════════════════════════════════════════════


class TestSympyEval:

    def test_basic_arithmetic(self):
        import sympy as sp
        result = _sympy_eval("2 + 3", {})
        assert result == 5

    def test_symbolic(self):
        import sympy as sp
        result = _sympy_eval("x**2 + 1", {})
        x = sp.Symbol("x")
        assert result == x**2 + 1

    def test_with_variables(self):
        import sympy as sp
        result = _sympy_eval("a + b", {"a": sp.Integer(10), "b": sp.Integer(20)})
        assert result == 30

    def test_trig(self):
        import sympy as sp
        result = _sympy_eval("sin(pi/2)", {})
        assert result == 1

    def test_latex_input(self):
        import sympy as sp
        result = _sympy_eval(r"\frac{1}{2} + \frac{1}{3}", {})
        assert result == sp.Rational(5, 6)

    def test_sqrt(self):
        import sympy as sp
        result = _sympy_eval("sqrt(4)", {})
        assert result == 2


# ══════════════════════════════════════════════════════════════════
# _expr_to_latex / _expr_to_str
# ══════════════════════════════════════════════════════════════════


class TestExprConversion:

    def test_to_latex(self):
        import sympy as sp
        x = sp.Symbol("x")
        result = _expr_to_latex(x**2)
        assert "x" in result
        assert "2" in result

    def test_to_str(self):
        import sympy as sp
        result = _expr_to_str(sp.Rational(1, 2))
        assert "1" in result and "2" in result

    def test_to_latex_numeric(self):
        result = _expr_to_latex(42)
        assert "42" in result


# ══════════════════════════════════════════════════════════════════
# _build_rich_result
# ══════════════════════════════════════════════════════════════════


class TestBuildRichResult:

    def test_basic_structure(self):
        result = json.loads(_build_rich_result("Test", "x^2", "x squared"))
        rich = result["__rich_mcp__"]
        assert rich["version"] == "1.0"
        assert rich["render"]["type"] == "html_sandbox"
        assert "Test" in rich["llm_summary"]

    def test_with_steps(self):
        steps = [{"title": "Step 1", "latex": "x", "note": "start"}]
        result = json.loads(_build_rich_result("Test", "", "", steps=steps))
        js = result["__rich_mcp__"]["render"]["js"]
        assert "Step 1" in js

    def test_with_plotly(self):
        plotly = {"data": [{"x": [1, 2], "y": [3, 4], "type": "scatter"}], "layout": {}}
        result = json.loads(_build_rich_result("Test", "", "", plotly_json=plotly))
        render = result["__rich_mcp__"]["render"]
        assert "plotly" in render["vendor_libs"]
        assert "Plotly.newPlot" in render["js"]

    def test_vendor_libs_always_include_katex(self):
        result = json.loads(_build_rich_result("Test", "x", "x"))
        vendor = result["__rich_mcp__"]["render"]["vendor_libs"]
        assert "katex_js" in vendor
        assert "katex_css" in vendor

    def test_with_extra_info(self):
        result = json.loads(_build_rich_result("Test", "", "", extra_info={"Type": "int"}))
        js = result["__rich_mcp__"]["render"]["js"]
        assert "Type" in js
        assert "int" in js


# ══════════════════════════════════════════════════════════════════
# _js_escape / _html_escape
# ══════════════════════════════════════════════════════════════════


class TestEscaping:

    def test_js_escape_quotes(self):
        assert "\\'" in _js_escape("it's")
        assert '\\"' in _js_escape('say "hi"')

    def test_js_escape_html_tags(self):
        assert "&lt;" in _js_escape("<script>")
        assert "&gt;" in _js_escape("</script>")

    def test_js_escape_newlines(self):
        assert "\\n" in _js_escape("line1\nline2")

    def test_html_escape(self):
        assert "&amp;" in _html_escape("a & b")
        assert "&lt;" in _html_escape("<div>")
        assert "&gt;" in _html_escape("</div>")
        assert "&quot;" in _html_escape('"quoted"')


# ══════════════════════════════════════════════════════════════════
# MathMCP.list_tools
# ══════════════════════════════════════════════════════════════════


class TestListTools:

    def test_returns_13_tools(self, mcp):
        tools = _run(mcp.list_tools())
        assert len(tools) == 13

    def test_all_tools_have_required_fields(self, mcp):
        tools = _run(mcp.list_tools())
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_tool_names(self, mcp):
        tools = _run(mcp.list_tools())
        names = {t["name"] for t in tools}
        expected = {
            "math_evaluate", "math_solve", "math_differentiate", "math_integrate",
            "math_simplify", "math_limit", "math_series", "math_matrix",
            "math_statistics", "math_plot_2d", "math_plot_3d",
            "math_set_variable", "math_get_variables",
        }
        assert names == expected

    def test_all_have_display_metadata(self, mcp):
        tools = _run(mcp.list_tools())
        for tool in tools:
            assert "displayName" in tool, f"{tool['name']} missing displayName"
            assert "icon" in tool, f"{tool['name']} missing icon"


# ══════════════════════════════════════════════════════════════════
# MathMCP.call_tool — dispatching
# ══════════════════════════════════════════════════════════════════


class TestCallToolDispatch:

    def test_unknown_tool(self, mcp):
        result = json.loads(_run(mcp.call_tool("nonexistent", {})))
        assert "error" in result

    def test_error_handling(self, mcp):
        result = json.loads(_run(mcp.call_tool("math_evaluate", {"expression": "???!!!"})))
        assert "error" in result


# ══════════════════════════════════════════════════════════════════
# math_evaluate
# ══════════════════════════════════════════════════════════════════


class TestEvaluate:

    def test_basic_arithmetic(self, mcp):
        r = _parse(_run(mcp.call_tool("math_evaluate", {"expression": "2 + 3"})))
        assert "5" in r["llm_summary"]

    def test_symbolic_expression(self, mcp):
        r = _parse(_run(mcp.call_tool("math_evaluate", {"expression": "sqrt(2)"})))
        assert "render" in r
        assert r["render"]["type"] == "html_sandbox"

    def test_numeric_mode(self, mcp):
        r = _parse(_run(mcp.call_tool("math_evaluate", {"expression": "pi", "numeric": True})))
        # Numeric value appears in the rendered JS (extra_info), not in llm_summary
        assert "3.14159" in r["render"]["js"]

    def test_latex_input(self, mcp):
        r = _parse(_run(mcp.call_tool("math_evaluate", {"expression": r"\frac{1}{2} + \frac{1}{3}"})))
        # Should evaluate to 5/6
        assert "5" in r["llm_summary"]

    def test_sum_evaluates(self, mcp):
        """Sum(1/k**2, (k, 1, 10)) should evaluate to a rational, not stay unevaluated."""
        r = _parse(_run(mcp.call_tool("math_evaluate", {"expression": "Sum(1/k**2, (k, 1, 10))", "numeric": True})))
        # Should evaluate to 1968329/1270080 (a fraction), not unevaluated Sum
        assert "1968329" in r["llm_summary"]
        # Numeric approximation in rendered JS
        assert "1.549" in r["render"]["js"]

    def test_complex_numbers(self, mcp):
        r = _parse(_run(mcp.call_tool("math_evaluate", {"expression": "sqrt(-1)"})))
        # Should give I (imaginary unit)
        assert r["render"]["type"] == "html_sandbox"

    def test_factorial(self, mcp):
        r = _parse(_run(mcp.call_tool("math_evaluate", {"expression": "factorial(10)"})))
        assert "3628800" in r["llm_summary"]

    def test_user_variable_in_expression(self, mcp):
        _run(mcp.call_tool("math_set_variable", {"name": "val", "value": "42"}))
        r = _parse(_run(mcp.call_tool("math_evaluate", {"expression": "val + 8"})))
        assert "50" in r["llm_summary"]


# ══════════════════════════════════════════════════════════════════
# math_solve
# ══════════════════════════════════════════════════════════════════


class TestSolve:

    def test_quadratic(self, mcp):
        r = _parse(_run(mcp.call_tool("math_solve", {"equations": "x**2 - 5*x + 6 = 0"})))
        summary = r["llm_summary"]
        assert "2" in summary
        assert "3" in summary

    def test_linear_system(self, mcp):
        r = _parse(_run(mcp.call_tool("math_solve", {
            "equations": "x + y = 5; x - y = 1",
            "variables": "x, y",
        })))
        summary = r["llm_summary"]
        assert "3" in summary  # x = 3
        assert "2" in summary  # y = 2

    def test_no_equals_sign(self, mcp):
        """Expression without = is treated as expr = 0."""
        r = _parse(_run(mcp.call_tool("math_solve", {"equations": "x**2 - 4"})))
        summary = r["llm_summary"]
        assert "2" in summary

    def test_auto_detect_variables(self, mcp):
        """Should auto-detect x as the variable."""
        r = _parse(_run(mcp.call_tool("math_solve", {"equations": "2*x + 3 = 11"})))
        summary = r["llm_summary"]
        assert "4" in summary

    def test_has_steps(self, mcp):
        r = _parse(_run(mcp.call_tool("math_solve", {"equations": "x**2 = 9"})))
        js = r["render"]["js"]
        assert "Given" in js or "Solution" in js


# ══════════════════════════════════════════════════════════════════
# math_differentiate
# ══════════════════════════════════════════════════════════════════


class TestDifferentiate:

    def test_polynomial(self, mcp):
        r = _parse(_run(mcp.call_tool("math_differentiate", {"expression": "x**3"})))
        # d/dx(x^3) = 3x^2
        assert "3" in r["llm_summary"]

    def test_trig(self, mcp):
        r = _parse(_run(mcp.call_tool("math_differentiate", {"expression": "sin(x)"})))
        assert "cos" in r["llm_summary"]

    def test_higher_order(self, mcp):
        r = _parse(_run(mcp.call_tool("math_differentiate", {
            "expression": "x**4",
            "order": 2,
        })))
        # d²/dx²(x^4) = 12x^2
        assert "12" in r["llm_summary"]

    def test_different_variable(self, mcp):
        r = _parse(_run(mcp.call_tool("math_differentiate", {
            "expression": "t**2 + 3*t",
            "variable": "t",
        })))
        # d/dt(t^2 + 3t) = 2t + 3
        summary = r["llm_summary"]
        assert "2" in summary and "3" in summary

    def test_has_steps(self, mcp):
        r = _parse(_run(mcp.call_tool("math_differentiate", {"expression": "exp(x)"})))
        js = r["render"]["js"]
        assert "Original" in js or "Derivative" in js


# ══════════════════════════════════════════════════════════════════
# math_integrate
# ══════════════════════════════════════════════════════════════════


class TestIntegrate:

    def test_indefinite(self, mcp):
        r = _parse(_run(mcp.call_tool("math_integrate", {"expression": "x**2"})))
        # ∫x² dx = x³/3
        assert "3" in r["llm_summary"]

    def test_definite(self, mcp):
        r = _parse(_run(mcp.call_tool("math_integrate", {
            "expression": "x**2",
            "lower": "0",
            "upper": "1",
        })))
        # ∫₀¹ x² dx = 1/3
        summary = r["llm_summary"]
        assert "1" in summary and "3" in summary

    def test_trig_definite(self, mcp):
        r = _parse(_run(mcp.call_tool("math_integrate", {
            "expression": "sin(x)",
            "lower": "0",
            "upper": "pi",
        })))
        # ∫₀π sin(x) dx = 2
        assert "2" in r["llm_summary"]

    def test_has_steps(self, mcp):
        r = _parse(_run(mcp.call_tool("math_integrate", {"expression": "exp(x)"})))
        js = r["render"]["js"]
        assert "Integrand" in js or "Antiderivative" in js


# ══════════════════════════════════════════════════════════════════
# math_simplify
# ══════════════════════════════════════════════════════════════════


class TestSimplify:

    def test_cancel(self, mcp):
        r = _parse(_run(mcp.call_tool("math_simplify", {
            "expression": "(x**2 - 1)/(x - 1)",
            "mode": "cancel",
        })))
        assert "x + 1" in r["llm_summary"]

    def test_expand(self, mcp):
        r = _parse(_run(mcp.call_tool("math_simplify", {
            "expression": "(x + 1)**3",
            "mode": "expand",
        })))
        summary = r["llm_summary"]
        assert "x" in summary

    def test_factor(self, mcp):
        r = _parse(_run(mcp.call_tool("math_simplify", {
            "expression": "x**2 - 4",
            "mode": "factor",
        })))
        summary = r["llm_summary"]
        assert "x" in summary and "2" in summary

    def test_trigsimp(self, mcp):
        r = _parse(_run(mcp.call_tool("math_simplify", {
            "expression": "sin(x)**2 + cos(x)**2",
            "mode": "trigsimp",
        })))
        assert "1" in r["llm_summary"]

    def test_default_mode_is_simplify(self, mcp):
        r = _parse(_run(mcp.call_tool("math_simplify", {
            "expression": "x + x",
        })))
        assert "2" in r["llm_summary"]


# ══════════════════════════════════════════════════════════════════
# math_limit
# ══════════════════════════════════════════════════════════════════


class TestLimit:

    def test_sinc_limit(self, mcp):
        r = _parse(_run(mcp.call_tool("math_limit", {
            "expression": "sin(x)/x",
            "point": "0",
        })))
        assert "1" in r["llm_summary"]

    def test_limit_at_infinity(self, mcp):
        r = _parse(_run(mcp.call_tool("math_limit", {
            "expression": "(1 + 1/x)**x",
            "point": "oo",
        })))
        # Should be e
        summary = r["llm_summary"]
        assert "E" in summary or "e" in summary or "exp" in summary

    def test_has_steps(self, mcp):
        r = _parse(_run(mcp.call_tool("math_limit", {
            "expression": "1/x",
            "point": "oo",
        })))
        js = r["render"]["js"]
        assert "Expression" in js or "Limit" in js


# ══════════════════════════════════════════════════════════════════
# math_series
# ══════════════════════════════════════════════════════════════════


class TestSeries:

    def test_exp_maclaurin(self, mcp):
        r = _parse(_run(mcp.call_tool("math_series", {"expression": "exp(x)", "order": 4})))
        summary = r["llm_summary"]
        # Should contain 1 + x + x²/2 + x³/6 + O(x⁴)
        assert "1" in summary

    def test_sin_series(self, mcp):
        r = _parse(_run(mcp.call_tool("math_series", {"expression": "sin(x)", "order": 5})))
        summary = r["llm_summary"]
        assert "x" in summary

    def test_around_nonzero_point(self, mcp):
        r = _parse(_run(mcp.call_tool("math_series", {
            "expression": "log(x)",
            "point": "1",
            "order": 4,
        })))
        assert r["render"]["type"] == "html_sandbox"

    def test_has_steps(self, mcp):
        r = _parse(_run(mcp.call_tool("math_series", {"expression": "cos(x)"})))
        js = r["render"]["js"]
        assert "Function" in js or "Taylor" in js


# ══════════════════════════════════════════════════════════════════
# math_matrix
# ══════════════════════════════════════════════════════════════════


class TestMatrix:

    def test_determinant(self, mcp):
        r = _parse(_run(mcp.call_tool("math_matrix", {
            "matrix": [[1, 2], [3, 4]],
            "operation": "determinant",
        })))
        assert "-2" in r["llm_summary"]

    def test_inverse(self, mcp):
        r = _parse(_run(mcp.call_tool("math_matrix", {
            "matrix": [[1, 0], [0, 1]],
            "operation": "inverse",
        })))
        # Identity matrix inverse is itself
        assert "1" in r["llm_summary"]

    def test_eigenvalues(self, mcp):
        r = _parse(_run(mcp.call_tool("math_matrix", {
            "matrix": [[2, 0], [0, 3]],
            "operation": "eigenvalues",
        })))
        summary = r["llm_summary"]
        assert "2" in summary
        assert "3" in summary

    def test_rank(self, mcp):
        r = _parse(_run(mcp.call_tool("math_matrix", {
            "matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            "operation": "rank",
        })))
        assert "2" in r["llm_summary"]

    def test_rref(self, mcp):
        r = _parse(_run(mcp.call_tool("math_matrix", {
            "matrix": [[1, 2], [3, 4]],
            "operation": "rref",
        })))
        js = r["render"]["js"]
        assert "Row echelon" in js or "Pivot" in js

    def test_transpose(self, mcp):
        r = _parse(_run(mcp.call_tool("math_matrix", {
            "matrix": [[1, 2], [3, 4]],
            "operation": "transpose",
        })))
        assert r["render"]["type"] == "html_sandbox"

    def test_trace(self, mcp):
        r = _parse(_run(mcp.call_tool("math_matrix", {
            "matrix": [[1, 0], [0, 5]],
            "operation": "trace",
        })))
        assert "6" in r["llm_summary"]

    def test_multiply(self, mcp):
        r = _parse(_run(mcp.call_tool("math_matrix", {
            "matrix": [[1, 0], [0, 1]],
            "operation": "multiply",
            "matrix_b": [[5, 6], [7, 8]],
        })))
        summary = r["llm_summary"]
        assert "5" in summary

    def test_nullspace(self, mcp):
        r = _parse(_run(mcp.call_tool("math_matrix", {
            "matrix": [[1, 2], [2, 4]],
            "operation": "nullspace",
        })))
        js = r["render"]["js"]
        assert "Null" in js

    def test_eigenvectors(self, mcp):
        r = _parse(_run(mcp.call_tool("math_matrix", {
            "matrix": [[2, 0], [0, 3]],
            "operation": "eigenvectors",
        })))
        js = r["render"]["js"]
        assert "Eigenvalue" in js


# ══════════════════════════════════════════════════════════════════
# math_statistics
# ══════════════════════════════════════════════════════════════════


class TestStatistics:

    def test_basic_stats(self, mcp):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        r = _parse(_run(mcp.call_tool("math_statistics", {"data": data})))
        summary = r["llm_summary"]
        assert "Mean" in summary
        assert "5.5" in summary
        assert "Median" in summary
        assert "Std Dev" in summary

    def test_histogram_visualization(self, mcp):
        r = _parse(_run(mcp.call_tool("math_statistics", {
            "data": [1, 2, 3, 4, 5],
            "visualize": "histogram",
        })))
        assert "plotly" in r["render"]["vendor_libs"]
        assert "Plotly.newPlot" in r["render"]["js"]

    def test_boxplot(self, mcp):
        r = _parse(_run(mcp.call_tool("math_statistics", {
            "data": [1, 2, 3, 4, 5, 10, 20],
            "visualize": "boxplot",
        })))
        assert "plotly" in r["render"]["vendor_libs"]

    def test_correlation(self, mcp):
        r = _parse(_run(mcp.call_tool("math_statistics", {
            "data": [1, 2, 3, 4, 5],
            "data_y": [2, 4, 6, 8, 10],
        })))
        summary = r["llm_summary"]
        assert "Correlation" in summary

    def test_scatter_with_regression(self, mcp):
        r = _parse(_run(mcp.call_tool("math_statistics", {
            "data": [1, 2, 3, 4, 5],
            "data_y": [2, 4, 5, 4, 5],
            "visualize": "scatter",
        })))
        assert "plotly" in r["render"]["vendor_libs"]

    def test_no_visualization(self, mcp):
        r = _parse(_run(mcp.call_tool("math_statistics", {
            "data": [1, 2, 3],
            "visualize": "none",
        })))
        assert "Plotly.newPlot" not in r["render"]["js"]


# ══════════════════════════════════════════════════════════════════
# math_plot_2d
# ══════════════════════════════════════════════════════════════════


class TestPlot2D:

    def test_cartesian(self, mcp):
        r = _parse(_run(mcp.call_tool("math_plot_2d", {
            "functions": ["sin(x)"],
        })))
        assert "plotly" in r["render"]["vendor_libs"]
        assert "Plotly.newPlot" in r["render"]["js"]

    def test_multiple_functions(self, mcp):
        r = _parse(_run(mcp.call_tool("math_plot_2d", {
            "functions": ["sin(x)", "cos(x)"],
        })))
        js = r["render"]["js"]
        assert "sin(x)" in js
        assert "cos(x)" in js

    def test_custom_range(self, mcp):
        r = _parse(_run(mcp.call_tool("math_plot_2d", {
            "functions": ["x**2"],
            "x_range": [-2, 2],
        })))
        assert r["render"]["type"] == "html_sandbox"

    def test_polar(self, mcp):
        r = _parse(_run(mcp.call_tool("math_plot_2d", {
            "functions": ["1 + cos(theta)"],
            "plot_type": "polar",
        })))
        assert "plotly" in r["render"]["vendor_libs"]

    def test_parametric(self, mcp):
        r = _parse(_run(mcp.call_tool("math_plot_2d", {
            "functions": ["cos(t)", "sin(t)"],
            "plot_type": "parametric",
        })))
        assert "plotly" in r["render"]["vendor_libs"]

    def test_implicit(self, mcp):
        r = _parse(_run(mcp.call_tool("math_plot_2d", {
            "functions": ["x**2 + y**2 - 1"],
            "plot_type": "implicit",
            "x_range": [-2, 2],
            "y_range": [-2, 2],
        })))
        assert "plotly" in r["render"]["vendor_libs"]

    def test_with_title(self, mcp):
        r = _parse(_run(mcp.call_tool("math_plot_2d", {
            "functions": ["x"],
            "title": "My Plot",
        })))
        assert "My Plot" in r["render"]["js"]


# ══════════════════════════════════════════════════════════════════
# math_plot_3d
# ══════════════════════════════════════════════════════════════════


class TestPlot3D:

    def test_surface(self, mcp):
        r = _parse(_run(mcp.call_tool("math_plot_3d", {
            "functions": ["x**2 + y**2"],
            "points": 15,
        })))
        assert "plotly" in r["render"]["vendor_libs"]

    def test_parametric_surface(self, mcp):
        r = _parse(_run(mcp.call_tool("math_plot_3d", {
            "functions": ["cos(u)*sin(v)", "sin(u)*sin(v)", "cos(v)"],
            "plot_type": "parametric_surface",
            "u_range": [0, 6.28],
            "v_range": [0, 3.14],
            "points": 15,
        })))
        assert "plotly" in r["render"]["vendor_libs"]

    def test_parametric_curve(self, mcp):
        r = _parse(_run(mcp.call_tool("math_plot_3d", {
            "functions": ["cos(t)", "sin(t)", "t/10"],
            "plot_type": "parametric_curve",
            "u_range": [0, 20],
            "points": 15,
        })))
        assert "plotly" in r["render"]["vendor_libs"]

    def test_custom_colorscale(self, mcp):
        r = _parse(_run(mcp.call_tool("math_plot_3d", {
            "functions": ["sin(x)*cos(y)"],
            "colorscale": "Plasma",
            "points": 10,
        })))
        assert "Plasma" in r["render"]["js"]


# ══════════════════════════════════════════════════════════════════
# math_set_variable / math_get_variables
# ══════════════════════════════════════════════════════════════════


class TestVariables:

    def test_set_numeric(self, mcp):
        r = _parse(_run(mcp.call_tool("math_set_variable", {"name": "a", "value": "42"})))
        assert "42" in r["llm_summary"]

    def test_set_expression(self, mcp):
        r = _parse(_run(mcp.call_tool("math_set_variable", {"name": "f", "value": "x**2 + 1"})))
        assert "f" in r["render"]["js"]

    def test_get_empty(self, mcp):
        fresh = MathMCP()
        result = json.loads(_run(fresh.call_tool("math_get_variables", {})))
        assert result["message"] == "No variables stored in this session."

    def test_get_with_variables(self, mcp):
        _run(mcp.call_tool("math_set_variable", {"name": "a", "value": "1"}))
        _run(mcp.call_tool("math_set_variable", {"name": "b", "value": "2"}))
        result = json.loads(_run(mcp.call_tool("math_get_variables", {})))
        assert result["count"] == 2
        assert "a" in result["variables"]
        assert "b" in result["variables"]

    def test_variable_used_in_evaluate(self, mcp):
        _run(mcp.call_tool("math_set_variable", {"name": "myval", "value": "7"}))
        r = _parse(_run(mcp.call_tool("math_evaluate", {"expression": "myval * 6"})))
        assert "42" in r["llm_summary"]

    def test_variable_overwrite(self, mcp):
        _run(mcp.call_tool("math_set_variable", {"name": "v", "value": "10"}))
        _run(mcp.call_tool("math_set_variable", {"name": "v", "value": "20"}))
        result = json.loads(_run(mcp.call_tool("math_get_variables", {})))
        assert result["count"] == 1
        assert "20" in result["variables"]["v"]

    def test_set_matrix_variable(self, mcp):
        _run(mcp.call_tool("math_set_variable", {"name": "M", "value": "Matrix([[1,2],[3,4]])"}))
        result = json.loads(_run(mcp.call_tool("math_get_variables", {})))
        assert "M" in result["variables"]


# ══════════════════════════════════════════════════════════════════
# prompt_hints
# ══════════════════════════════════════════════════════════════════


class TestPromptHints:

    def test_returns_hints(self, mcp):
        hints = _run(mcp.get_prompt_hints())
        assert len(hints) > 0
        assert any("math" in h.lower() for h in hints)


# ══════════════════════════════════════════════════════════════════
# session isolation
# ══════════════════════════════════════════════════════════════════


class TestSessionIsolation:

    def test_separate_variable_scopes(self):
        mcp1 = MathMCP()
        mcp2 = MathMCP()
        _run(mcp1.call_tool("math_set_variable", {"name": "x_val", "value": "100"}))
        result = json.loads(_run(mcp2.call_tool("math_get_variables", {})))
        assert "message" in result  # mcp2 should have no variables


# ══════════════════════════════════════════════════════════════════
# timeout
# ══════════════════════════════════════════════════════════════════


class TestTimeout:

    def test_respects_max_timeout(self):
        mcp = MathMCP(timeout=999)
        assert mcp._timeout == 60.0  # capped at MAX_TIMEOUT

    def test_normal_timeout(self):
        mcp = MathMCP(timeout=5.0)
        assert mcp._timeout == 5.0


# ══════════════════════════════════════════════════════════════════
# _build_text_result
# ══════════════════════════════════════════════════════════════════


class TestBuildTextResult:

    def test_basic(self):
        result = _build_text_result("Test Title", "x^2", "x squared")
        assert "**Test Title**" in result
        assert "$" in result
        assert "x^2" in result
        assert "`x squared`" in result

    def test_with_steps(self):
        steps = [{"title": "Step 1", "latex": "x", "note": "start"}]
        result = _build_text_result("Test", "", "", steps=steps)
        assert "**Steps:**" in result
        assert "Step 1" in result
        assert "$x$" in result
        assert "*start*" in result

    def test_with_extra_info(self):
        result = _build_text_result("Test", "", "", extra_info={"Type": "int"})
        assert "**Type:** int" in result

    def test_no_latex(self):
        result = _build_text_result("Test", "", "42")
        assert "$$" not in result
        assert "`42`" in result


# ══════════════════════════════════════════════════════════════════
# Plain text mode (rich_output=False, default)
# ══════════════════════════════════════════════════════════════════


class TestPlainTextMode:

    def test_default_is_plain(self):
        mcp = MathMCP()
        assert mcp._rich is False

    def test_evaluate_returns_plain_text(self, mcp_text):
        result = _run(mcp_text.call_tool("math_evaluate", {"expression": "2 + 3"}))
        assert "__rich_mcp__" not in result
        assert "5" in result
        assert "**" in result  # markdown bold

    def test_solve_returns_plain_text(self, mcp_text):
        result = _run(mcp_text.call_tool("math_solve", {"equations": "x**2 - 4"}))
        assert "__rich_mcp__" not in result
        assert "2" in result

    def test_differentiate_returns_plain_text(self, mcp_text):
        result = _run(mcp_text.call_tool("math_differentiate", {"expression": "x**3"}))
        assert "__rich_mcp__" not in result
        assert "3" in result

    def test_integrate_returns_plain_text(self, mcp_text):
        result = _run(mcp_text.call_tool("math_integrate", {"expression": "x**2"}))
        assert "__rich_mcp__" not in result
        assert "Integration" in result

    def test_simplify_returns_plain_text(self, mcp_text):
        result = _run(mcp_text.call_tool("math_simplify", {"expression": "x + x"}))
        assert "__rich_mcp__" not in result
        assert "2" in result

    def test_limit_returns_plain_text(self, mcp_text):
        result = _run(mcp_text.call_tool("math_limit", {"expression": "sin(x)/x", "point": "0"}))
        assert "__rich_mcp__" not in result
        assert "1" in result

    def test_series_returns_plain_text(self, mcp_text):
        result = _run(mcp_text.call_tool("math_series", {"expression": "exp(x)", "order": 3}))
        assert "__rich_mcp__" not in result
        assert "Taylor" in result

    def test_matrix_returns_plain_text(self, mcp_text):
        result = _run(mcp_text.call_tool("math_matrix", {
            "matrix": [[1, 2], [3, 4]], "operation": "determinant",
        }))
        assert "__rich_mcp__" not in result
        assert "-2" in result

    def test_set_variable_returns_plain_text(self, mcp_text):
        result = _run(mcp_text.call_tool("math_set_variable", {"name": "a", "value": "42"}))
        assert "__rich_mcp__" not in result
        assert "42" in result

    def test_statistics_no_viz_returns_plain_text(self, mcp_text):
        result = _run(mcp_text.call_tool("math_statistics", {
            "data": [1, 2, 3, 4, 5], "visualize": "none",
        }))
        assert "__rich_mcp__" not in result
        assert "Mean" in result

    def test_statistics_with_viz_returns_rich(self, mcp_text):
        result = _run(mcp_text.call_tool("math_statistics", {
            "data": [1, 2, 3, 4, 5], "visualize": "histogram",
        }))
        data = json.loads(result)
        assert "__rich_mcp__" in data

    def test_plot_2d_always_rich(self, mcp_text):
        result = _run(mcp_text.call_tool("math_plot_2d", {"functions": ["sin(x)"]}))
        data = json.loads(result)
        assert "__rich_mcp__" in data

    def test_plot_3d_always_rich(self, mcp_text):
        result = _run(mcp_text.call_tool("math_plot_3d", {
            "functions": ["x**2 + y**2"], "points": 10,
        }))
        data = json.loads(result)
        assert "__rich_mcp__" in data
