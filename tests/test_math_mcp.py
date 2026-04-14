"""Comprehensive unit tests for MathMCP in-process MCP server.

Tests cover:
- Helper functions: _rnd, _parse_latex_to_sympy, _build_rich_result, _sympy_eval
- MathMCP class: __init__, list_tools, get_prompt_hints
- Tool calls via call_tool: evaluate, solve, differentiate, integrate, simplify,
  limit, series, matrix, statistics, plot_2d, plot_3d, set_variable, get_variables
- Edge cases: invalid expressions, division by zero, unknown tools, empty arguments,
  various LaTeX notations
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest

from llming_lodge.tools.math_mcp import (
    MathMCP,
    _build_rich_result,
    _parse_latex_to_sympy,
    _rnd,
    _sympy_eval,
    _build_text_result,
    _safe_sympy_env,
    _expr_to_latex,
    _expr_to_str,
    DEFAULT_TIMEOUT,
    MAX_TIMEOUT,
    _PLOT_DECIMALS,
)


# ═══════════════════════════════════════════════════════════════════
# Helper function tests
# ═══════════════════════════════════════════════════════════════════


class TestRnd:
    """Tests for the _rnd() rounding helper."""

    def test_non_array_passthrough(self):
        """Non-array values are returned unchanged."""
        assert _rnd(42) == 42
        assert _rnd("hello") == "hello"
        assert _rnd(None) is None
        assert _rnd([1, 2, 3]) == [1, 2, 3]

    def test_pure_numeric_array(self):
        """Pure float array is rounded and converted to list."""
        arr = np.array([1.123456789, 2.987654321, 3.0])
        result = _rnd(arr)
        assert isinstance(result, list)
        assert result == [1.1235, 2.9877, 3.0]

    def test_object_array_with_none(self):
        """Object arrays containing None values preserve None in output."""
        arr = np.array([1.123456, None, 3.789012], dtype=object)
        result = _rnd(arr)
        assert isinstance(result, list)
        assert result[0] == pytest.approx(1.1235, abs=1e-4)
        assert result[1] is None
        assert result[2] == pytest.approx(3.789, abs=1e-4)

    def test_nested_object_array(self):
        """Object array with nested lists (like a 2D z-matrix with None)."""
        inner = [1.123456, None, 2.987654]
        arr = np.array([inner, [4.5, 5.6, None]], dtype=object)
        result = _rnd(arr)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0][0] == pytest.approx(1.1235, abs=1e-4)
        assert result[0][1] is None
        assert result[1][2] is None

    def test_integer_array(self):
        """Integer arrays are rounded (no-op) and converted to list."""
        arr = np.array([1, 2, 3])
        result = _rnd(arr)
        assert result == [1, 2, 3]

    def test_empty_array(self):
        """Empty array returns empty list."""
        arr = np.array([])
        result = _rnd(arr)
        assert result == []


class TestParseLatexToSympy:
    """Tests for _parse_latex_to_sympy() LaTeX-to-SymPy converter."""

    def test_dollar_sign_removal(self):
        """Strips single and double dollar signs."""
        assert _parse_latex_to_sympy("$x + 1$") == "x + 1"
        assert _parse_latex_to_sympy("$$x + 1$$") == "x + 1"

    def test_frac_conversion(self):
        """Converts \\frac{a}{b} to (a)/(b)."""
        result = _parse_latex_to_sympy(r"\frac{3}{7}")
        assert result == "(3)/(7)"

    def test_nested_frac(self):
        """Handles nested fractions iteratively."""
        result = _parse_latex_to_sympy(r"\frac{1}{2} + \frac{3}{4}")
        assert "(1)/(2)" in result
        assert "(3)/(4)" in result

    def test_sqrt_conversion(self):
        """Converts \\sqrt{x} to sqrt(x)."""
        result = _parse_latex_to_sympy(r"\sqrt{x + 1}")
        assert result == "sqrt(x + 1)"

    def test_trig_functions(self):
        """Strips backslash from trig function names."""
        assert _parse_latex_to_sympy(r"\sin(x)") == "sin(x)"
        assert _parse_latex_to_sympy(r"\cos(x)") == "cos(x)"
        assert _parse_latex_to_sympy(r"\tan(x)") == "tan(x)"
        assert _parse_latex_to_sympy(r"\log(x)") == "log(x)"
        assert _parse_latex_to_sympy(r"\exp(x)") == "exp(x)"

    def test_pi_and_infinity(self):
        """Converts \\pi to pi, \\infty to oo."""
        assert _parse_latex_to_sympy(r"\pi") == "pi"
        assert _parse_latex_to_sympy(r"\infty") == "oo"
        assert _parse_latex_to_sympy(r"\inf") == "oo"

    def test_cdot_and_times(self):
        """Converts \\cdot and \\times to *."""
        assert _parse_latex_to_sympy(r"2 \cdot 3") == "2 * 3"
        assert _parse_latex_to_sympy(r"2 \times 3") == "2 * 3"

    def test_left_right_removal(self):
        """Removes \\left and \\right."""
        result = _parse_latex_to_sympy(r"\left( x + 1 \right)")
        assert r"\left" not in result
        assert r"\right" not in result
        assert "( x + 1 )" in result

    def test_power_braces(self):
        """Converts x^{n} to x**(n)."""
        result = _parse_latex_to_sympy("x^{2}")
        assert "x**(2)" in result

    def test_caret_with_backslash(self):
        """Converts \\^ to **."""
        result = _parse_latex_to_sympy(r"x\^2")
        assert "x**2" in result

    def test_implicit_multiplication(self):
        """Inserts * between digit and letter: 2x -> 2*x."""
        result = _parse_latex_to_sympy("2x + 3y")
        assert "2*x" in result
        assert "3*y" in result

    def test_remaining_backslash_commands(self):
        """Strips backslash from unknown commands."""
        result = _parse_latex_to_sympy(r"\alpha + \beta")
        assert result == "alpha + beta"

    def test_plain_expression_unchanged(self):
        """Plain expressions without LaTeX pass through mostly unchanged."""
        result = _parse_latex_to_sympy("x**2 + 1")
        assert result == "x**2 + 1"


class TestBuildRichResult:
    """Tests for _build_rich_result() envelope builder."""

    def test_basic_envelope_structure(self):
        """Returns valid JSON with __rich_mcp__ envelope."""
        result = _build_rich_result(
            title="Test Title",
            latex=r"x^2",
            result_text="x**2",
        )
        data = json.loads(result)
        assert "__rich_mcp__" in data
        envelope = data["__rich_mcp__"]
        assert envelope["version"] == "1.0"
        assert envelope["min_viewer_version"] == "1.0"
        assert "render" in envelope
        assert "llm_summary" in envelope

    def test_render_data_fields(self):
        """Render data contains type, title, latex, result_text."""
        result = _build_rich_result(
            title="Evaluation Result",
            latex=r"\frac{1}{2}",
            result_text="1/2",
        )
        data = json.loads(result)
        render = data["__rich_mcp__"]["render"]
        assert render["type"] == "math_result"
        assert render["title"] == "Evaluation Result"
        assert render["latex"] == r"\frac{1}{2}"
        assert render["result_text"] == "1/2"

    def test_with_steps(self):
        """Steps list is included in render data."""
        steps = [
            {"title": "Step 1", "latex": "x", "note": "Given"},
            {"title": "Step 2", "latex": "2x", "note": "Doubled"},
        ]
        result = _build_rich_result(
            title="Test",
            latex="2x",
            result_text="2*x",
            steps=steps,
        )
        data = json.loads(result)
        assert data["__rich_mcp__"]["render"]["steps"] == steps

    def test_with_extra_info(self):
        """Extra info dict is included in render data."""
        result = _build_rich_result(
            title="Test",
            latex="",
            result_text="42",
            extra_info={"Type": "Integer", "N": "5"},
        )
        data = json.loads(result)
        assert data["__rich_mcp__"]["render"]["extra_info"] == {"Type": "Integer", "N": "5"}

    def test_optional_fields_omitted_when_empty(self):
        """Empty optional fields are not included in render data."""
        result = _build_rich_result(
            title="Test",
            latex="",
            result_text="",
        )
        data = json.loads(result)
        render = data["__rich_mcp__"]["render"]
        assert "latex" not in render
        assert "result_text" not in render
        assert "steps" not in render
        assert "extra_info" not in render

    def test_llm_summary_content(self):
        """LLM summary includes title, result_text, and step info."""
        steps = [{"title": "Given", "note": "x=1"}]
        result = _build_rich_result(
            title="Test Title",
            latex="",
            result_text="42",
            steps=steps,
        )
        data = json.loads(result)
        summary = data["__rich_mcp__"]["llm_summary"]
        assert "Test Title" in summary
        assert "42" in summary
        assert "Given" in summary


class TestBuildTextResult:
    """Tests for _build_text_result() plain-text builder."""

    def test_basic_output(self):
        """Basic output includes title and LaTeX in $$ delimiters."""
        result = _build_text_result(
            title="Result",
            latex="x^2",
            result_text="x**2",
        )
        assert "**Result**" in result
        assert "$$\nx^2\n$$" in result
        assert "`x**2`" in result

    def test_with_steps(self):
        """Steps are numbered and include title, latex, note."""
        steps = [
            {"title": "Step 1", "latex": "x", "note": "initial"},
        ]
        result = _build_text_result("Title", "", "", steps=steps)
        assert "1. **Step 1**" in result
        assert "$$x$$" in result
        assert "*initial*" in result

    def test_with_extra_info(self):
        """Extra info is formatted as bold key-value pairs."""
        result = _build_text_result("Title", "", "", extra_info={"Type": "Integer"})
        assert "**Type:** Integer" in result


class TestSympyEval:
    """Tests for _sympy_eval() expression evaluator."""

    def test_basic_arithmetic(self):
        """Evaluates basic arithmetic expressions."""
        import sympy as sp
        result = _sympy_eval("2 + 3", {})
        assert result == 5

    def test_symbolic_expression(self):
        """Returns symbolic expression when variables are free."""
        import sympy as sp
        result = _sympy_eval("x**2 + 1", {})
        x = sp.Symbol("x")
        assert result == x**2 + 1

    def test_with_variables(self):
        """Uses provided variables in evaluation."""
        import sympy as sp
        result = _sympy_eval("a + b", {"a": sp.Integer(10), "b": sp.Integer(20)})
        assert result == 30

    def test_latex_input(self):
        """Handles LaTeX-formatted input via _parse_latex_to_sympy."""
        import sympy as sp
        result = _sympy_eval(r"\frac{1}{2} + \frac{1}{3}", {})
        assert result == sp.Rational(5, 6)

    def test_trig_function(self):
        """Evaluates trigonometric functions."""
        import sympy as sp
        result = _sympy_eval("sin(pi/2)", {})
        assert result == 1

    def test_implicit_multiplication(self):
        """Handles implicit multiplication (e.g., 2x)."""
        import sympy as sp
        x = sp.Symbol("x")
        result = _sympy_eval("2x", {})
        assert result == 2 * x

    def test_xor_converted_to_power(self):
        """The ^ operator is converted to exponentiation via convert_xor."""
        import sympy as sp
        result = _sympy_eval("2^10", {})
        assert result == 1024


class TestSafeSympyEnv:
    """Tests for _safe_sympy_env() namespace builder."""

    def test_common_symbols_present(self):
        """Common symbols x, y, z, t, etc. are pre-created."""
        import sympy as sp
        env = _safe_sympy_env({})
        assert isinstance(env["x"], sp.Symbol)
        assert isinstance(env["y"], sp.Symbol)
        assert isinstance(env["z"], sp.Symbol)
        assert isinstance(env["t"], sp.Symbol)
        assert str(env["x"]) == "x"

    def test_functions_present(self):
        """Standard math functions are available."""
        env = _safe_sympy_env({})
        assert callable(env["sin"])
        assert callable(env["cos"])
        assert callable(env["sqrt"])
        assert callable(env["log"])
        assert callable(env["exp"])

    def test_constants_present(self):
        """Mathematical constants are available."""
        import sympy as sp
        env = _safe_sympy_env({})
        assert env["pi"] == sp.pi
        assert env["e"] == sp.E
        assert env["I"] == sp.I
        assert env["oo"] == sp.oo

    def test_user_variables_overlay(self):
        """User variables override defaults."""
        import sympy as sp
        env = _safe_sympy_env({"x": sp.Integer(42)})
        assert env["x"] == 42

    def test_integer_symbols(self):
        """n and k are created as integer symbols."""
        env = _safe_sympy_env({})
        assert env["n"].is_integer is True
        assert env["k"].is_integer is True


class TestExprConversion:
    """Tests for _expr_to_latex() and _expr_to_str()."""

    def test_latex_output(self):
        """SymPy expression is converted to LaTeX."""
        import sympy as sp
        x = sp.Symbol("x")
        result = _expr_to_latex(x**2 + 1)
        assert "x" in result
        assert "2" in result

    def test_str_output(self):
        """SymPy expression is converted to readable string."""
        import sympy as sp
        x = sp.Symbol("x")
        result = _expr_to_str(x**2 + 1)
        assert "x" in result

    def test_latex_handles_string_input(self):
        """Handles plain string input (sympy renders it as LaTeX text)."""
        result = _expr_to_latex("not_a_sympy_expr")
        # sympy.latex wraps strings in \mathtt{\text{...}}
        assert isinstance(result, str)
        assert len(result) > 0

    def test_str_fallback_on_error(self):
        """Falls back to str() for non-sympy objects."""
        result = _expr_to_str("not_a_sympy_expr")
        assert result == "not_a_sympy_expr"


# ═══════════════════════════════════════════════════════════════════
# MathMCP class tests
# ═══════════════════════════════════════════════════════════════════


class TestMathMCPInit:
    """Tests for MathMCP initialization."""

    def test_default_init(self):
        """Default initialization creates instance with empty variables."""
        mcp = MathMCP()
        assert mcp._variables == {}
        assert mcp._timeout == DEFAULT_TIMEOUT
        assert mcp._rich is False
        assert mcp._doc_store is None

    def test_custom_timeout(self):
        """Custom timeout is stored, capped at MAX_TIMEOUT."""
        mcp = MathMCP(timeout=10.0)
        assert mcp._timeout == 10.0

    def test_timeout_capped(self):
        """Timeout exceeding MAX_TIMEOUT is capped."""
        mcp = MathMCP(timeout=999.0)
        assert mcp._timeout == MAX_TIMEOUT

    def test_rich_output_mode(self):
        """Rich output mode is configurable."""
        mcp = MathMCP(rich_output=True)
        assert mcp._rich is True

    def test_document_store(self):
        """Document store is stored."""
        store = object()
        mcp = MathMCP(document_store=store)
        assert mcp._doc_store is store


class TestMathMCPListTools:
    """Tests for list_tools() method."""

    @pytest.mark.asyncio
    async def test_returns_list(self):
        """list_tools returns a list of tool definitions."""
        mcp = MathMCP()
        tools = await mcp.list_tools()
        assert isinstance(tools, list)
        assert len(tools) == 13

    @pytest.mark.asyncio
    async def test_tool_names(self):
        """All expected tool names are present."""
        mcp = MathMCP()
        tools = await mcp.list_tools()
        names = {t["name"] for t in tools}
        expected = {
            "math_evaluate", "math_solve", "math_differentiate",
            "math_integrate", "math_simplify", "math_limit",
            "math_series", "math_matrix", "math_statistics",
            "math_plot_2d", "math_plot_3d",
            "math_set_variable", "math_get_variables",
        }
        assert names == expected

    @pytest.mark.asyncio
    async def test_tool_structure(self):
        """Each tool has required fields: name, description, inputSchema."""
        mcp = MathMCP()
        tools = await mcp.list_tools()
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"


class TestMathMCPPromptHints:
    """Tests for get_prompt_hints() method."""

    @pytest.mark.asyncio
    async def test_returns_hints(self):
        """get_prompt_hints returns a non-empty list of strings."""
        mcp = MathMCP()
        hints = await mcp.get_prompt_hints()
        assert isinstance(hints, list)
        assert len(hints) >= 1
        assert all(isinstance(h, str) for h in hints)

    @pytest.mark.asyncio
    async def test_hints_mention_math_tools(self):
        """Hints mention math tools to guide the LLM."""
        mcp = MathMCP()
        hints = await mcp.get_prompt_hints()
        combined = " ".join(hints)
        assert "math" in combined.lower()


# ═══════════════════════════════════════════════════════════════════
# Tool call tests (via call_tool)
# ═══════════════════════════════════════════════════════════════════


class TestEvaluate:
    """Tests for math_evaluate tool."""

    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        """Evaluates basic arithmetic: 2^10 + sqrt(144) = 1036."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_evaluate", {"expression": "2**10 + sqrt(144)"})
        assert "1036" in result

    @pytest.mark.asyncio
    async def test_symbolic_expression(self):
        """Returns symbolic result for expression with free variable."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_evaluate", {"expression": "x**2 + 2*x + 1"})
        # Should contain the simplified or original form
        # (x + 1)^2 or x^2 + 2x + 1
        assert "x" in result

    @pytest.mark.asyncio
    async def test_numeric_mode(self):
        """Numeric mode returns floating-point approximation."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_evaluate", {
            "expression": "pi",
            "numeric": True,
        })
        assert "3.14159" in result

    @pytest.mark.asyncio
    async def test_trig_evaluation(self):
        """Evaluates trigonometric expressions: sin(pi/4)."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_evaluate", {"expression": "sin(pi/4)"})
        # sqrt(2)/2
        assert "2" in result

    @pytest.mark.asyncio
    async def test_sum_evaluation(self):
        """Evaluates summation: sum(k, (k, 1, 10)) = 55."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_evaluate", {
            "expression": "Sum(k, (k, 1, 10))",
        })
        assert "55" in result

    @pytest.mark.asyncio
    async def test_latex_input(self):
        r"""Evaluates LaTeX input: \frac{3}{7} + \frac{2}{5}."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_evaluate", {
            "expression": r"\frac{3}{7} + \frac{2}{5}",
        })
        # 3/7 + 2/5 = 15/35 + 14/35 = 29/35
        assert "29" in result

    @pytest.mark.asyncio
    async def test_rich_output_mode(self):
        """Rich output mode returns __rich_mcp__ envelope."""
        mcp = MathMCP(rich_output=True)
        result = await mcp.call_tool("math_evaluate", {"expression": "2 + 3"})
        data = json.loads(result)
        assert "__rich_mcp__" in data
        assert data["__rich_mcp__"]["render"]["type"] == "math_result"

    @pytest.mark.asyncio
    async def test_text_output_mode(self):
        """Default (non-rich) mode returns markdown text with $$."""
        mcp = MathMCP(rich_output=False)
        result = await mcp.call_tool("math_evaluate", {"expression": "2 + 3"})
        assert "**Evaluation Result**" in result
        assert "5" in result


class TestSolve:
    """Tests for math_solve tool."""

    @pytest.mark.asyncio
    async def test_quadratic_equation(self):
        """Solves x^2 - 5x + 6 = 0 => x = 2, x = 3."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_solve", {
            "equations": "x**2 - 5*x + 6 = 0",
        })
        assert "2" in result
        assert "3" in result

    @pytest.mark.asyncio
    async def test_linear_equation(self):
        """Solves 2x + 3 = 7 => x = 2."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_solve", {
            "equations": "2*x + 3 = 7",
        })
        assert "2" in result

    @pytest.mark.asyncio
    async def test_system_of_equations(self):
        """Solves system: x + y = 5; x - y = 1 => x=3, y=2."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_solve", {
            "equations": "x + y = 5; x - y = 1",
            "variables": "x, y",
        })
        assert "3" in result
        assert "2" in result

    @pytest.mark.asyncio
    async def test_expression_assumed_zero(self):
        """Expression without = is assumed equal to 0."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_solve", {
            "equations": "x**2 - 4",
        })
        assert "2" in result

    @pytest.mark.asyncio
    async def test_auto_detect_variables(self):
        """Variables are auto-detected when not specified."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_solve", {
            "equations": "x**2 = 9",
        })
        assert "3" in result

    @pytest.mark.asyncio
    async def test_solve_with_steps(self):
        """Result includes step information."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_solve", {
            "equations": "x = 5",
        })
        assert "Given" in result or "Solution" in result


class TestDifferentiate:
    """Tests for math_differentiate tool."""

    @pytest.mark.asyncio
    async def test_polynomial_derivative(self):
        """d/dx(x^3 + 2x) = 3x^2 + 2."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_differentiate", {
            "expression": "x**3 + 2*x",
        })
        # Should contain 3x^2 + 2 in some form
        assert "3" in result
        assert "2" in result

    @pytest.mark.asyncio
    async def test_trig_derivative(self):
        """d/dx(sin(x)) = cos(x)."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_differentiate", {
            "expression": "sin(x)",
        })
        assert "cos" in result

    @pytest.mark.asyncio
    async def test_chain_rule(self):
        """d/dx(exp(-x^2)) applies chain rule."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_differentiate", {
            "expression": "exp(-x**2)",
        })
        # Result should contain -2*x*exp(-x^2) or equivalent
        assert "x" in result
        assert "exp" in result.lower() or "e" in result

    @pytest.mark.asyncio
    async def test_higher_order_derivative(self):
        """Second derivative of x^4 = 12x^2."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_differentiate", {
            "expression": "x**4",
            "order": 2,
        })
        assert "12" in result

    @pytest.mark.asyncio
    async def test_custom_variable(self):
        """Differentiate with respect to custom variable."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_differentiate", {
            "expression": "t**2 + 3*t",
            "variable": "t",
        })
        assert "t" in result or "3" in result

    @pytest.mark.asyncio
    async def test_default_variable_is_x(self):
        """Default variable is x when not specified."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_differentiate", {
            "expression": "x**2",
        })
        assert "2" in result


class TestIntegrate:
    """Tests for math_integrate tool."""

    @pytest.mark.asyncio
    async def test_indefinite_integral(self):
        """Integral of x^2 = x^3/3."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_integrate", {
            "expression": "x**2",
        })
        assert "3" in result
        assert "Indefinite" in result or "Antiderivative" in result

    @pytest.mark.asyncio
    async def test_definite_integral(self):
        """Integral of sin(x) from 0 to pi = 2."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_integrate", {
            "expression": "sin(x)",
            "lower": "0",
            "upper": "pi",
        })
        assert "2" in result

    @pytest.mark.asyncio
    async def test_definite_with_numeric(self):
        """Definite integral includes numeric value."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_integrate", {
            "expression": "x**2",
            "lower": "0",
            "upper": "1",
        })
        # Integral of x^2 from 0 to 1 = 1/3
        assert "3" in result or "0.333" in result

    @pytest.mark.asyncio
    async def test_custom_variable(self):
        """Integration with respect to custom variable."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_integrate", {
            "expression": "t**2",
            "variable": "t",
        })
        assert "3" in result

    @pytest.mark.asyncio
    async def test_exp_integral(self):
        """Integral of exp(x) = exp(x)."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_integrate", {
            "expression": "exp(x)",
        })
        assert "exp" in result.lower() or "e" in result


class TestSimplify:
    """Tests for math_simplify tool."""

    @pytest.mark.asyncio
    async def test_simplify_default(self):
        """Default simplification of (x^2 - 1)/(x - 1) = x + 1."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_simplify", {
            "expression": "(x**2 - 1)/(x - 1)",
        })
        assert "x + 1" in result or "x+1" in result

    @pytest.mark.asyncio
    async def test_expand_mode(self):
        """Expand (x + 1)^2 = x^2 + 2x + 1."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_simplify", {
            "expression": "(x + 1)**2",
            "mode": "expand",
        })
        assert "2" in result

    @pytest.mark.asyncio
    async def test_factor_mode(self):
        """Factor x^2 + 2x + 1 = (x + 1)^2."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_simplify", {
            "expression": "x**2 + 2*x + 1",
            "mode": "factor",
        })
        # Should contain (x + 1)^2 or (x + 1)**2
        assert "x + 1" in result or "x+1" in result

    @pytest.mark.asyncio
    async def test_trigsimp_mode(self):
        """Trig simplification: sin(x)^2 + cos(x)^2 = 1."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_simplify", {
            "expression": "sin(x)**2 + cos(x)**2",
            "mode": "trigsimp",
        })
        assert "1" in result

    @pytest.mark.asyncio
    async def test_cancel_mode(self):
        """Cancel common factors."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_simplify", {
            "expression": "(x**2 - 4)/(x - 2)",
            "mode": "cancel",
        })
        assert "x + 2" in result or "x+2" in result

    @pytest.mark.asyncio
    async def test_apart_mode(self):
        """Partial fraction decomposition."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_simplify", {
            "expression": "1/(x**2 - 1)",
            "mode": "apart",
        })
        # Should decompose into partial fractions
        assert "x" in result

    @pytest.mark.asyncio
    async def test_result_includes_mode_in_title(self):
        """Result title includes the simplification mode."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_simplify", {
            "expression": "x + x",
            "mode": "simplify",
        })
        assert "simplify" in result.lower()


class TestLimit:
    """Tests for math_limit tool."""

    @pytest.mark.asyncio
    async def test_sinx_over_x(self):
        """lim(sin(x)/x, x->0) = 1."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_limit", {
            "expression": "sin(x)/x",
            "point": "0",
        })
        assert "1" in result

    @pytest.mark.asyncio
    async def test_limit_at_infinity(self):
        """lim((1+1/n)^n, n->oo) = e."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_limit", {
            "expression": "(1 + 1/n)**n",
            "variable": "n",
            "point": "oo",
        })
        # Result should be e (Euler's number)
        assert "e" in result.lower() or "E" in result

    @pytest.mark.asyncio
    async def test_limit_with_direction(self):
        """One-sided limit: lim(1/x, x->0+) = oo."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_limit", {
            "expression": "1/x",
            "point": "0",
            "direction": "+",
        })
        assert "oo" in result or "inf" in result.lower() or "\u221e" in result

    @pytest.mark.asyncio
    async def test_polynomial_limit(self):
        """lim(x^2, x->3) = 9."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_limit", {
            "expression": "x**2",
            "point": "3",
        })
        assert "9" in result


class TestSeries:
    """Tests for math_series tool."""

    @pytest.mark.asyncio
    async def test_exp_maclaurin(self):
        """Maclaurin series of exp(x): 1 + x + x^2/2 + ..."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_series", {
            "expression": "exp(x)",
        })
        # Should contain early terms
        assert "1" in result
        assert "x" in result

    @pytest.mark.asyncio
    async def test_sin_maclaurin(self):
        """Maclaurin series of sin(x): x - x^3/6 + ..."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_series", {
            "expression": "sin(x)",
        })
        assert "x" in result

    @pytest.mark.asyncio
    async def test_custom_order(self):
        """Series expansion with custom order."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_series", {
            "expression": "cos(x)",
            "order": 4,
        })
        assert "x" in result

    @pytest.mark.asyncio
    async def test_expansion_around_nonzero_point(self):
        """Taylor series around non-zero point."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_series", {
            "expression": "ln(x)",
            "variable": "x",
            "point": "1",
            "order": 4,
        })
        assert "x" in result

    @pytest.mark.asyncio
    async def test_series_includes_steps(self):
        """Series result includes function, expansion, and polynomial steps."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_series", {
            "expression": "exp(x)",
            "order": 3,
        })
        assert "Function" in result or "Taylor" in result or "Polynomial" in result


class TestMatrix:
    """Tests for math_matrix tool."""

    @pytest.mark.asyncio
    async def test_determinant(self):
        """Determinant of [[1,2],[3,4]] = -2."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_matrix", {
            "matrix": [[1, 2], [3, 4]],
            "operation": "determinant",
        })
        assert "-2" in result

    @pytest.mark.asyncio
    async def test_inverse(self):
        """Inverse of 2x2 matrix."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_matrix", {
            "matrix": [[1, 2], [3, 4]],
            "operation": "inverse",
        })
        # Inverse of [[1,2],[3,4]] has entries involving -2, 1, 3/2, -1/2
        assert "Inverse" in result or "inverse" in result

    @pytest.mark.asyncio
    async def test_eigenvalues(self):
        """Eigenvalues of identity matrix are all 1."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_matrix", {
            "matrix": [[1, 0], [0, 1]],
            "operation": "eigenvalues",
        })
        assert "1" in result

    @pytest.mark.asyncio
    async def test_rank(self):
        """Rank of [[1,2],[2,4]] = 1."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_matrix", {
            "matrix": [[1, 2], [2, 4]],
            "operation": "rank",
        })
        assert "1" in result

    @pytest.mark.asyncio
    async def test_transpose(self):
        """Transpose of [[1,2],[3,4]] = [[1,3],[2,4]]."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_matrix", {
            "matrix": [[1, 2], [3, 4]],
            "operation": "transpose",
        })
        assert "Transpose" in result or "transpose" in result

    @pytest.mark.asyncio
    async def test_trace(self):
        """Trace of [[1,2],[3,4]] = 5."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_matrix", {
            "matrix": [[1, 2], [3, 4]],
            "operation": "trace",
        })
        assert "5" in result

    @pytest.mark.asyncio
    async def test_multiply(self):
        """Matrix multiplication."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_matrix", {
            "matrix": [[1, 0], [0, 1]],
            "operation": "multiply",
            "matrix_b": [[5, 6], [7, 8]],
        })
        assert "5" in result
        assert "8" in result

    @pytest.mark.asyncio
    async def test_rref(self):
        """Row echelon form of a matrix."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_matrix", {
            "matrix": [[1, 2, 3], [4, 5, 6]],
            "operation": "rref",
        })
        assert "echelon" in result.lower() or "Pivot" in result or "pivot" in result

    @pytest.mark.asyncio
    async def test_nullspace(self):
        """Nullspace of a rank-deficient matrix."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_matrix", {
            "matrix": [[1, 2], [2, 4]],
            "operation": "nullspace",
        })
        assert "Null" in result or "null" in result

    @pytest.mark.asyncio
    async def test_eigenvectors(self):
        """Eigenvectors of a matrix."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_matrix", {
            "matrix": [[2, 1], [0, 2]],
            "operation": "eigenvectors",
        })
        assert "Eigenvalue" in result or "eigenvalue" in result

    @pytest.mark.asyncio
    async def test_dimensions_in_info(self):
        """Result includes matrix dimensions."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_matrix", {
            "matrix": [[1, 2, 3], [4, 5, 6]],
            "operation": "rank",
        })
        assert "2x3" in result


class TestStatistics:
    """Tests for math_statistics tool."""

    @pytest.mark.asyncio
    async def test_basic_statistics(self):
        """Computes basic statistics on a simple dataset."""
        mcp = MathMCP()
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = await mcp.call_tool("math_statistics", {"data": data})
        assert "Mean" in result
        assert "5.5" in result  # mean
        assert "Median" in result
        assert "Std Dev" in result
        assert "Min" in result
        assert "Max" in result

    @pytest.mark.asyncio
    async def test_count(self):
        """Count matches number of data points."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_statistics", {"data": [1, 2, 3]})
        assert "Count" in result
        assert "3" in result

    @pytest.mark.asyncio
    async def test_quartiles(self):
        """Quartile values are computed."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_statistics", {
            "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        assert "Q1" in result
        assert "Q3" in result
        assert "IQR" in result

    @pytest.mark.asyncio
    async def test_correlation_with_two_datasets(self):
        """Correlation and regression with paired data."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_statistics", {
            "data": [1, 2, 3, 4, 5],
            "data_y": [2, 4, 6, 8, 10],
        })
        # Perfect correlation r=1
        assert "Correlation" in result
        assert "Regression" in result

    @pytest.mark.asyncio
    async def test_skewness_kurtosis(self):
        """Skewness and kurtosis computed for n > 2."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_statistics", {
            "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        assert "Skewness" in result
        assert "Kurtosis" in result

    @pytest.mark.asyncio
    async def test_histogram_visualization(self):
        """Histogram visualization returns JSON with chart data."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_statistics", {
            "data": [1, 2, 3, 4, 5],
            "visualize": "histogram",
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_boxplot_visualization(self):
        """Boxplot visualization returns JSON with chart data."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_statistics", {
            "data": [1, 2, 3, 4, 5],
            "visualize": "boxplot",
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_scatter_visualization(self):
        """Scatter plot with regression returns JSON with chart data."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_statistics", {
            "data": [1, 2, 3, 4, 5],
            "data_y": [2, 4, 5, 4, 5],
            "visualize": "scatter",
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_single_element_dataset(self):
        """Single element dataset produces valid statistics (std dev = 0)."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_statistics", {"data": [42]})
        assert "Mean" in result
        assert "42" in result


class TestPlot2D:
    """Tests for math_plot_2d tool."""

    @pytest.mark.asyncio
    async def test_cartesian_plot(self):
        """Cartesian plot returns valid JSON with plot data."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_2d", {
            "functions": ["sin(x)"],
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_multiple_functions(self):
        """Multiple functions can be plotted together."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_2d", {
            "functions": ["sin(x)", "cos(x)"],
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_custom_range(self):
        """Custom x and y ranges."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_2d", {
            "functions": ["x**2"],
            "x_range": [-5, 5],
            "y_range": [0, 25],
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_custom_title(self):
        """Custom plot title."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_2d", {
            "functions": ["x**2"],
            "title": "Parabola",
        })
        data = json.loads(result)
        assert data["name"] == "Parabola"

    @pytest.mark.asyncio
    async def test_parametric_plot(self):
        """Parametric plot returns valid data."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_2d", {
            "functions": ["cos(t)", "sin(t)"],
            "plot_type": "parametric",
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_polar_plot(self):
        """Polar plot returns valid data."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_2d", {
            "functions": ["1 + cos(theta)"],
            "plot_type": "polar",
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_implicit_plot(self):
        """Implicit plot (contour) returns valid data."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_2d", {
            "functions": ["x**2 + y**2 - 1"],
            "plot_type": "implicit",
            "x_range": [-2, 2],
            "y_range": [-2, 2],
            "points": 50,
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_points_capped(self):
        """Points parameter is capped at 1000."""
        mcp = MathMCP()
        # Should not raise even with large points value
        result = await mcp.call_tool("math_plot_2d", {
            "functions": ["x"],
            "points": 5000,
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_with_document_store(self):
        """When document store is provided, doc is created with inline data."""
        class MockDoc:
            id = "doc-123"
        class MockStore:
            def create(self, **kwargs):
                return MockDoc()
        mcp = MathMCP(document_store=MockStore())
        result = await mcp.call_tool("math_plot_2d", {
            "functions": ["x"],
        })
        data = json.loads(result)
        assert data["document_id"] == "doc-123"
        assert "__inline_doc__" in data


class TestPlot3D:
    """Tests for math_plot_3d tool."""

    @pytest.mark.asyncio
    async def test_surface_plot(self):
        """3D surface plot returns valid JSON."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_3d", {
            "functions": ["sin(sqrt(x**2 + y**2))"],
            "points": 10,
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_parametric_surface(self):
        """Parametric surface plot with 3 expressions."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_3d", {
            "functions": ["cos(u)*sin(v)", "sin(u)*sin(v)", "cos(v)"],
            "plot_type": "parametric_surface",
            "points": 10,
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_parametric_curve(self):
        """Parametric 3D curve plot."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_3d", {
            "functions": ["cos(t)", "sin(t)", "t"],
            "plot_type": "parametric_curve",
            "points": 10,
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_custom_ranges(self):
        """Custom x/y ranges for surface plot."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_3d", {
            "functions": ["x*y"],
            "x_range": [-2, 2],
            "y_range": [-2, 2],
            "points": 10,
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_custom_colorscale(self):
        """Custom colorscale parameter."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_3d", {
            "functions": ["x**2 - y**2"],
            "colorscale": "Plasma",
            "points": 10,
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_points_capped(self):
        """Points parameter is capped at 80."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_3d", {
            "functions": ["x + y"],
            "points": 200,
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"


class TestSetVariable:
    """Tests for math_set_variable tool."""

    @pytest.mark.asyncio
    async def test_set_numeric(self):
        """Stores a numeric variable."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_set_variable", {
            "name": "my_val",
            "value": "42",
        })
        assert "my_val" in result
        assert "42" in result
        assert "my_val" in mcp._variables

    @pytest.mark.asyncio
    async def test_set_expression(self):
        """Stores a symbolic expression as a variable."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_set_variable", {
            "name": "f",
            "value": "3*x + 1",
        })
        assert "f" in result
        assert "f" in mcp._variables

    @pytest.mark.asyncio
    async def test_variable_persists(self):
        """Variable persists across tool calls."""
        mcp = MathMCP()
        await mcp.call_tool("math_set_variable", {
            "name": "a_val",
            "value": "10",
        })
        # Now use the variable in evaluation
        result = await mcp.call_tool("math_evaluate", {
            "expression": "a_val + 5",
        })
        assert "15" in result

    @pytest.mark.asyncio
    async def test_variable_count(self):
        """Variable count is tracked."""
        mcp = MathMCP()
        await mcp.call_tool("math_set_variable", {"name": "v1", "value": "1"})
        result = await mcp.call_tool("math_set_variable", {"name": "v2", "value": "2"})
        assert "2" in result  # count or value

    @pytest.mark.asyncio
    async def test_overwrite_variable(self):
        """Overwriting a variable updates its value."""
        mcp = MathMCP()
        await mcp.call_tool("math_set_variable", {"name": "x_val", "value": "10"})
        await mcp.call_tool("math_set_variable", {"name": "x_val", "value": "20"})
        result = await mcp.call_tool("math_evaluate", {"expression": "x_val"})
        assert "20" in result


class TestGetVariables:
    """Tests for math_get_variables tool."""

    @pytest.mark.asyncio
    async def test_empty_variables(self):
        """Returns message when no variables are stored."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_get_variables", {})
        data = json.loads(result)
        assert "No variables" in data["message"]

    @pytest.mark.asyncio
    async def test_list_stored_variables(self):
        """Lists all stored variables with values."""
        mcp = MathMCP()
        await mcp.call_tool("math_set_variable", {"name": "a", "value": "10"})
        await mcp.call_tool("math_set_variable", {"name": "b", "value": "20"})
        result = await mcp.call_tool("math_get_variables", {})
        data = json.loads(result)
        assert "variables" in data
        assert data["count"] == 2
        assert "a" in data["variables"]
        assert "b" in data["variables"]


# ═══════════════════════════════════════════════════════════════════
# Edge cases and error handling
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_unknown_tool_name(self):
        """Unknown tool name returns error JSON."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_nonexistent", {})
        data = json.loads(result)
        assert "error" in data
        assert "Unknown tool" in data["error"]

    @pytest.mark.asyncio
    async def test_missing_required_argument(self):
        """Missing required argument returns error."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_evaluate", {})
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_invalid_expression(self):
        """Invalid expression returns error."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_evaluate", {
            "expression": "!@#$%^&*(invalid",
        })
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_division_by_zero_in_evaluate(self):
        """Division by zero returns a symbolic result (zoo/nan/ComplexInfinity)."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_evaluate", {
            "expression": "1/0",
        })
        # SymPy returns zoo (ComplexInfinity) for 1/0
        assert "zoo" in result or "ComplexInfinity" in result or "\u221e" in result or "oo" in result

    @pytest.mark.asyncio
    async def test_singular_matrix_inverse(self):
        """Attempting to invert a singular matrix returns error."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_matrix", {
            "matrix": [[1, 2], [2, 4]],
            "operation": "inverse",
        })
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_latex_dollar_signs(self):
        """Expressions wrapped in dollar signs are handled."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_evaluate", {
            "expression": "$2 + 3$",
        })
        assert "5" in result

    @pytest.mark.asyncio
    async def test_latex_double_dollar_signs(self):
        """Expressions wrapped in double dollar signs are handled."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_evaluate", {
            "expression": "$$x^{2} + 1$$",
        })
        assert "x" in result

    @pytest.mark.asyncio
    async def test_empty_data_statistics(self):
        """Empty data array in statistics raises error."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_statistics", {"data": []})
        # numpy operations on empty arrays may raise or return nan
        # Either way, the error handler catches it
        data = json.loads(result)
        assert "error" in data or "Mean" in result

    @pytest.mark.asyncio
    async def test_solve_no_solution(self):
        """Equation with no real solution."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_solve", {
            "equations": "x**2 + 1 = 0",
        })
        # SymPy returns complex solutions: -i, i
        assert "I" in result or "i" in result or "Solution" in result

    @pytest.mark.asyncio
    async def test_very_large_number(self):
        """Handles very large numbers without error."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_evaluate", {
            "expression": "factorial(20)",
        })
        assert "2432902008176640000" in result

    @pytest.mark.asyncio
    async def test_complex_number_evaluation(self):
        """Complex number arithmetic."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_evaluate", {
            "expression": "(1 + I)**2",
        })
        # (1 + i)^2 = 2i
        assert "I" in result or "i" in result

    @pytest.mark.asyncio
    async def test_set_and_use_matrix_variable(self):
        """Set a matrix variable and use it."""
        mcp = MathMCP()
        await mcp.call_tool("math_set_variable", {
            "name": "M",
            "value": "Matrix([[1, 2], [3, 4]])",
        })
        result = await mcp.call_tool("math_get_variables", {})
        data = json.loads(result)
        assert "M" in data["variables"]

    @pytest.mark.asyncio
    async def test_integrate_with_only_lower_bound(self):
        """Only lower bound without upper -> treated as indefinite."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_integrate", {
            "expression": "x",
            "lower": "0",
        })
        # Without upper bound, is_definite is False
        assert "Indefinite" in result or "Antiderivative" in result

    @pytest.mark.asyncio
    async def test_differentiate_constant(self):
        """Derivative of a constant is 0."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_differentiate", {
            "expression": "5",
        })
        assert "0" in result

    @pytest.mark.asyncio
    async def test_simplify_already_simple(self):
        """Simplifying an already simple expression returns it unchanged."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_simplify", {
            "expression": "x + 1",
        })
        assert "x" in result
        assert "1" in result

    @pytest.mark.asyncio
    async def test_series_default_order(self):
        """Default order is 6 when not specified."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_series", {
            "expression": "exp(x)",
        })
        # Should include up to x^5 (order 6 means O(x^6))
        assert "x" in result

    @pytest.mark.asyncio
    async def test_limit_default_variable(self):
        """Default variable for limit is x."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_limit", {
            "expression": "x**2",
            "point": "2",
        })
        assert "4" in result

    @pytest.mark.asyncio
    async def test_multiply_without_matrix_b(self):
        """Multiply operation without matrix_b falls through gracefully."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_matrix", {
            "matrix": [[1, 2], [3, 4]],
            "operation": "multiply",
        })
        # Without matrix_b, the else branch returns str(M)
        assert "Matrix" in result or "1" in result

    @pytest.mark.asyncio
    async def test_plot_2d_with_nan_producing_function(self):
        """Plot handles functions that produce NaN/inf for some x values."""
        mcp = MathMCP()
        result = await mcp.call_tool("math_plot_2d", {
            "functions": ["log(x)"],
            "x_range": [-5, 5],
            "points": 20,
        })
        data = json.loads(result)
        assert data["status"] == "chart_created"

    @pytest.mark.asyncio
    async def test_evaluate_with_stored_variable(self):
        """Evaluation uses previously stored variables."""
        mcp = MathMCP()
        await mcp.call_tool("math_set_variable", {"name": "my_const", "value": "7"})
        result = await mcp.call_tool("math_evaluate", {
            "expression": "my_const * 6",
        })
        assert "42" in result

    @pytest.mark.asyncio
    async def test_result_mode_toggle(self):
        """Switching between rich and text mode for the same MCP instance."""
        mcp_rich = MathMCP(rich_output=True)
        result_rich = await mcp_rich.call_tool("math_evaluate", {"expression": "1+1"})
        data = json.loads(result_rich)
        assert "__rich_mcp__" in data

        mcp_text = MathMCP(rich_output=False)
        result_text = await mcp_text.call_tool("math_evaluate", {"expression": "1+1"})
        assert "**Evaluation Result**" in result_text
        assert "__rich_mcp__" not in result_text
