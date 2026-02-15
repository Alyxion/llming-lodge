# prompt_template.py
from __future__ import annotations
import re
import threading
import ast
from typing import Any, Dict, List, Tuple, Callable

##############################################################################
# 1. ──────────────  Safety helpers  ──────────────────────────────────────── #
##############################################################################
_ALLOWED_AST_NODES = {
    ast.Expression, ast.Name, ast.Load, ast.Constant,
    ast.UnaryOp, ast.UAdd, ast.USub, ast.Not,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.BoolOp, ast.And, ast.Or,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Call,  # Allow function calls, but check which ones below
}

def _check_ast_safety(node, max_repeat=10000, max_range=10000):
    """
    Recursively check AST for dangerous constructs like huge list/string multiplication or huge ranges.
    Raises ValueError if unsafe usage is detected.
    """
    # Block multiplying lists/strings by large ints
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, right = node.left, node.right
        # Check if left or right is a constant int and the other is a constant with a sequence type
        for side1, side2 in [(left, right), (right, left)]:
            if isinstance(side1, ast.Constant) and isinstance(side1.value, int):
                if side1.value > max_repeat:
                    raise ValueError(f"Multiplication by too large number: {side1.value}")
                if isinstance(side2, ast.Constant) and isinstance(side2.value, (str, list, tuple)):
                    if side1.value > max_repeat:
                        raise ValueError(f"Attempt to create object with {side1.value} elements")
    # Only allow calls to 'range', block all other calls
    if isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id == "range"):
            raise ValueError(f"Unsafe function call: only 'range' is allowed, got '{getattr(node.func, 'id', None)}'")
        # Try to extract the stop value for range
        if node.args:
            stop_arg = node.args[-1]
            if isinstance(stop_arg, ast.Constant) and isinstance(stop_arg.value, int):
                if stop_arg.value > max_range:
                    raise ValueError(f"range() with too large stop: {stop_arg.value}")
    # Block list/tuple with huge number of elements
    if isinstance(node, ast.Constant) and isinstance(node.value, (list, tuple)):
        if len(node.value) > max_repeat:
            raise ValueError(f"List/Tuple with too many elements: {len(node.value)}")
    # Recursively check all child nodes
    for child in ast.iter_child_nodes(node):
        _check_ast_safety(child, max_repeat=max_repeat, max_range=max_range)

def _safe_eval(expr: str, ctx: Dict[str, Any]) -> Any:
    """
    Evaluate *expr* using only safe Python operators and the provided context.
    """
    node = ast.parse(expr, mode="eval")
    if any(type(n) not in _ALLOWED_AST_NODES for n in ast.walk(node)):
        raise ValueError(f"Unsafe expression: {expr!r}")
    _check_ast_safety(node)
    return eval(compile(node, "<expr>", "eval"), {"__builtins__": {}, "range": range}, ctx)


##############################################################################
# 2. ──────────────  Template node representation  ────────────────────────── #
##############################################################################
class _Node:
    def render(self, ctx: Dict[str, Any], out: List[str]) -> None: ...

class _Text(_Node):
    __slots__ = ("text",)
    def __init__(self, text: str): self.text = text
    def render(self, ctx, out): out.append(self.text)

class _Var(_Node):
    __slots__ = ("expr", "filters")
    def __init__(self, expr: str, filters: List[Tuple[str, Tuple[str, ...]]]):
        self.expr, self.filters = expr.strip(), filters

    _filters_map: Dict[str, Callable[..., Any]] = {
        "upper": lambda v: str(v).upper(),
        "lower": lambda v: str(v).lower(),
        "clip":  lambda v, n="2": f"{float(v):.{int(n)}f}",
    }

    def render(self, ctx, out):
        val = _safe_eval(self.expr, ctx)
        for f_name, f_args in self.filters:
            if f_name not in self._filters_map:
                raise ValueError(f"Unknown filter: {f_name}")
            val = self._filters_map[f_name](val, *f_args)
        out.append(str(val))

class _If(_Node):
    __slots__ = ("tests", "else_block")
    def __init__(self):             # list of (condition:str | None, block:List[_Node])
        self.tests: List[Tuple[str | None, List[_Node]]] = []
        self.else_block: List[_Node] = []

    def render(self, ctx, out):
        for cond, block in self.tests:
            if cond is None or _safe_eval(cond, ctx):
                for n in block: n.render(ctx, out)
                return
        for n in self.else_block: n.render(ctx, out)

class _For(_Node):
    __slots__ = ("var", "iter_expr", "body")
    def __init__(self, var: str, iter_expr: str, body: List[_Node]):
        self.var, self.iter_expr, self.body = var.strip(), iter_expr.strip(), body

    def render(self, ctx, out):
        import time
        iterable = _safe_eval(self.iter_expr, ctx)
        start_time = ctx.get("_start_time")
        max_time_s = ctx.get("_max_time_s")
        for i, item in enumerate(iterable):
            # Check elapsed time every 50 iterations
            if i % 50 == 0 and start_time is not None and max_time_s is not None:
                elapsed = time.monotonic() - start_time
                if elapsed > max_time_s:
                    raise TimeoutError(f"Rendering exceeded {max_time_s:.4f}s limit (thread-aware)")
            ctx[self.var] = item
            for n in self.body:
                n.render(ctx, out)

##############################################################################
# 3. ──────────────  Template parser  ─────────────────────────────────────── #
##############################################################################
_TAG_RE = re.compile(r"({{.*?}}|{%-?.*?-%}|{%.*?%})", re.DOTALL)

def _parse(template: str) -> List[_Node]:
    tokens = _TAG_RE.split(template)
    root: List[_Node] = []
    stack: List[Tuple[str, _If | _For, List[_Node]]] = []  # (tag, node, cur_block)

    def current_block() -> List[_Node]:
        return root if not stack else stack[-1][2]

    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        if not tok or tok.startswith("\n") and tok.strip() == "":
            idx += 1; continue

        # Variable: {{ ... }}
        if tok.startswith("{{"):
            inner = tok[2:-2].strip()
            parts = [p.strip() for p in inner.split("|")]
            expr, filters = parts[0], parts[1:]
            filt_parsed: List[Tuple[str, Tuple[str, ...]]] = []
            for f in filters:
                if "(" in f:
                    name, arg_str = f.split("(", 1)
                    args = tuple(a.strip() for a in arg_str[:-1].split(","))  # drop ')'
                else:
                    name, args = f, ()
                filt_parsed.append((name, args))
            current_block().append(_Var(expr, filt_parsed))

        # Block tags: {% ... %}
        elif tok.startswith("{%"):
            tag_content = tok[2:-2].strip()
            parts = tag_content.split()
            keyword = parts[0]

            if keyword == "if":
                node = _If()
                node.tests.append((" ".join(parts[1:]), []))
                stack.append((keyword, node, node.tests[-1][1]))
            elif keyword == "elif":
                if not stack or stack[-1][0] != "if":
                    raise SyntaxError("elif without matching if")
                _kw, node, _ = stack[-1]
                node.tests.append((" ".join(parts[1:]), []))
                stack[-1] = (_kw, node, node.tests[-1][1])
            elif keyword == "else":
                if not stack or stack[-1][0] not in {"if", "for"}:
                    raise SyntaxError("else without matching block")
                if stack[-1][0] == "if":
                    _kw, node, _ = stack[-1]
                    node.else_block = []
                    stack[-1] = (_kw, node, node.else_block)
                else:  # 'for' blocks do not support else here
                    raise SyntaxError("Unexpected else in for‑block")
            elif keyword == "endif":
                if not stack or stack[-1][0] != "if":
                    raise SyntaxError("endif without matching if")
                _kw, node, _ = stack.pop()
                current_block().append(node)
            elif keyword == "for":
                if len(parts) < 4 or parts[2] != "in":
                    raise SyntaxError("Malformed for tag")
                var, iter_expr = parts[1], " ".join(parts[3:])
                node = _For(var, iter_expr, [])
                stack.append((keyword, node, node.body))
            elif keyword == "endfor":
                if not stack or stack[-1][0] != "for":
                    raise SyntaxError("endfor without matching for")
                _kw, node, _ = stack.pop()
                current_block().append(node)
            else:
                raise SyntaxError(f"Unknown tag: {keyword}")

        # Plain text
        else:
            current_block().append(_Text(tok))

        idx += 1

    if stack:
        raise SyntaxError("Unclosed block in template")
    return root

##############################################################################
# 4. ──────────────  Public API  ──────────────────────────────────────────── #
##############################################################################
class PromptTemplate:
    """
    Lightweight Jinja‑like template limited to:
        • {{ expr | upper / lower / clip(n) }}
        • {% if / elif / else / endif %}
        • {% for X in Y %} ... {% endfor %}
    No external libs, only `ast`, `re`, `threading`.
    """

    def __init__(self, source: str):
        if not isinstance(source, str):
            raise TypeError("Template source must be a string")
        self._nodes = _parse(source)
        self._error: str | None = None

    # ------------------------------------------------------------------ #
    def render(
        self,
        *,
        max_time_s: float = 0.25,
        max_output_size: int = 8192,
        parameters: dict | None = None
    ) -> str:
        parameters = parameters or {}
        # 1️⃣  Validate argument types
        for k, v in parameters.items():
            if not isinstance(v, (str, bool, int, float, list, tuple)):
                raise ValueError(
                    f"Argument '{k}' has invalid type {type(v).__name__}; "
                    "allowed: str, bool, int, float, list, tuple"
                )

        # 2️⃣  Render in a background thread to enforce wall‑clock limit
        result: List[str] = []
        self._error = None

        import time
        def _run() -> None:
            ctx = parameters.copy()  # local copy; loops may mutate
            ctx["_start_time"] = time.monotonic()
            ctx["_max_time_s"] = max_time_s
            try:
                for n in self._nodes:
                    n.render(ctx, result)
            except Exception as e:
                self._error = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(max_time_s)

        if self._error:
            if isinstance(self._error, TimeoutError):
                raise self._error
            raise ValueError(f"Failed to render prompt: {self._error}")

        if thread.is_alive():
            raise TimeoutError(f"Rendering exceeded {max_time_s:.2f}s limit")

        output = "".join(result)
        if len(output) > max_output_size:
            raise ValueError(
                f"Rendered output too large ({len(output)} chars > {max_output_size})"
            )
        return output

##############################################################################
# 5. ──────────────  Example  ─────────────────────────────────────────────── #
##############################################################################
if __name__ == "__main__":
    tpl = PromptTemplate("""
Hello {{ user|upper }}!

{% if balance > 0 %}
  Dein Kontostand: {{ balance|clip(2) }} €.
{% elif balance == 0 %}
  Konto ausgeglichen.
{% else %}
  Du schuldest {{ -balance|clip }} €.
{% endif %}

{% if language == "dutch" %}
 Dutch
{% endif %}

Letzte Vorgänge:
{% for row in history %}
  • {{ row|lower }}
{% endfor %}
""")

    msg = tpl.render(
        parameters={
            "user": "Michael",
            "balance": -3.499,
            "language": "german",
            "history": ["EINZAHLUNG 10", "RÜCKERSTATTUNG 2.5"],
        },
        max_time_s= 0.2,
        max_output_size= 4000
    )
    print(msg)
