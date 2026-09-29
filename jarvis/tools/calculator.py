"""
jarvis/tools/calculator.py
──────────────────────────
Tool: calculator

Evaluates basic mathematical expressions safely using Python's AST parser.
Does not use `eval()` to ensure strict sandboxing against arbitrary code execution.
"""

import ast
import operator

from jarvis.tools.base import BaseTool, CachePolicy
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

# Map AST nodes to Python operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_ast(node: ast.AST) -> float | int:
    """Recursively evaluates the AST node. Raises ValueError on unsafe elements."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    elif isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _OPERATORS:
            raise ValueError(f"Unsupported operator: {op.__name__}")
        
        left = _safe_eval_ast(node.left)
        right = _safe_eval_ast(node.right)
        
        # Guard against absurd power operations (e.g. 99**99999) that hang the thread
        if op is ast.Pow:
            if right > 1000 or right < -1000:
                raise ValueError("Exponent too large")
                
        return _OPERATORS[op](left, right)

    elif isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op.__name__}")
            
        operand = _safe_eval_ast(node.operand)
        return _OPERATORS[op](operand)

    elif isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)

    else:
        raise ValueError(f"Unsupported math structure: {type(node).__name__}")


def safe_math_eval(expr: str) -> float | int:
    """Parse string to AST and evaluate it securely."""
    try:
        tree = ast.parse(expr, mode='eval')
        return _safe_eval_ast(tree)
    except ZeroDivisionError:
        raise ValueError("Division by zero")
    except SyntaxError:
        raise ValueError("Invalid mathematical syntax")
    except Exception as e:
        # Catch-all for any other AST weirdness
        raise ValueError(str(e))


def canonical_expression_form(expr: str) -> str | None:
    """
    v0.24 (Part M): deterministic canonical form of a calculator expression,
    for CACHE-KEY equivalence only — '2+2', '2 + 2', '(2+2)', and even
    '5-1' map to the same cache entry when they evaluate to the same value.

    Uses THIS module's existing deterministic AST parser (no new math parser
    is invented — Part M). Because the calculator is a PURE FUNCTION of its
    expression, the evaluated value itself is the strongest canonical form:
    two expressions share a cache entry exactly when the tool would return
    the same result, which is the definition of safe reuse.

    Returns None when the expression cannot be parsed/evaluated safely — the
    caller then falls back to the generic (verbatim-whitespace) key, so a
    malformed expression is never confused with a valid one. NEVER used to
    decide execution; the real dispatch always runs the actual expression.
    """
    try:
        value = safe_math_eval(expr)
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return repr(float(value))


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Compute the EXACT result of one arithmetic expression. "
        "PURPOSE: precise arithmetic (+, -, *, /, **, parentheses). "
        "WHEN TO USE: whenever the answer requires calculating with numbers "
        "in the request or derived from it — sums, products, quotients, "
        "percentages, unit or day-count spans — even when the math looks "
        "easy, because mental arithmetic produces wrong answers. "
        "WHEN NOT TO USE: conceptual math questions (definitions, proofs, "
        "estimates, explanations) that do not ask for an exact value. "
        "INPUT: one expression, e.g. '12 * (4 + 3) / 2.5'. "
        "OUTPUT: 'Result: <number>' or an ERROR string (invalid syntax, "
        "division by zero, exponent too large)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The math expression to evaluate (e.g. '12 * (4 + 3) / 2.5').",
            },
        },
        "required": ["expression"],
    }
    # v0.24 (Part B, Class 2): a PURE deterministic function of its argument.
    # Cross-turn reuse is always safe; freshness is trivial (its result can't
    # decay), and the entry cap is the only eviction. Key normalization
    # proves expression equivalence with the same AST parser that executes
    # it (Part M): 2+2, 2 + 2 and (2+2) share one cache entry.
    cache_policy = CachePolicy(
        scope="global",
        freshness="ttl",
        normalizer="calculator_expression",
    )

    def run(self, expression: str, **kwargs) -> str:
        log.info("calculator", expression=expression)
        try:
            result = safe_math_eval(expression)
            
            # Format clean output (10.0 -> 10)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
                
            return f"Result: {result}"
        except Exception as e:
            log.warning("calculator_error", expression=expression, error=str(e))
            return f"ERROR: {e}"
