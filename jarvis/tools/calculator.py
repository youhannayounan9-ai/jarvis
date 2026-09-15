"""
jarvis/tools/calculator.py
──────────────────────────
Tool: calculator

Evaluates basic mathematical expressions safely using Python's AST parser.
Does not use `eval()` to ensure strict sandboxing against arbitrary code execution.
"""

import ast
import operator

from jarvis.tools.base import BaseTool
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


class CalculatorTool(BaseTool):
    name = "calculator"
    description = (
        "Evaluate a mathematical expression safely. "
        "Supports +, -, *, /, **, and parentheses. "
        "ONLY use this tool for exact calculations you cannot answer yourself."
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
