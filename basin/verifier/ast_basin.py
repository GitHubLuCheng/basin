"""
basin/verifier/ast_basin.py
================================
Symbolic (AST-normalized) reasoning-basin key for code generation.

Mirrors the philosophy of Game of 24's exact symbolic basin key
(operator sequence + remaining values): two code candidates are in the
same basin iff they implement the same *structural* algorithm, regardless
of variable/function naming, docstring wording, or formatting.

This is deliberately NOT based on execution outcome (pass/fail/exception
type) — clustering by outcome would make the redundancy-gap analysis
circular (basin membership would already encode correctness). It is also
not semantic/NLI-based like MuSR's hypothesis clustering — code has an
exact, free structural invariant available (its parse tree), so we use it
directly, the same way Game24 uses exact operator sequences instead of an
approximate/learned notion of arithmetic-strategy similarity.

Mechanism: parse the candidate, rename all identifiers (function name,
parameters, local variables) to canonical placeholders in order of first
appearance, strip docstrings, then dump the normalized tree. Two
candidates that differ only in naming/formatting collapse to the same
key; candidates with different control flow, different operations, or a
different algorithmic approach produce different keys.

Unparseable candidates fall back to a single ``"PARSE_ERROR"`` basin.
"""

from __future__ import annotations

import ast


class _Canonicalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self._map: dict[str, str] = {}
        self._n = 0

    def _canon(self, name: str) -> str:
        if name not in self._map:
            self._map[name] = f"v{self._n}"
            self._n += 1
        return self._map[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._canon(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._canon(node.arg)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self._canon(node.name)
        # Drop a leading docstring — it's copied prose, not algorithm structure.
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:]
        self.generic_visit(node)
        return node


def structural_basin_key(code: str) -> str:
    """Return an exact structural basin key for a candidate code string.

    Returns ``"PARSE_ERROR"`` if the code does not parse — a single shared
    basin for malformed candidates, analogous to how malformed states are
    handled elsewhere in the codebase (e.g. NLI extraction failures).
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return "PARSE_ERROR"
    try:
        normalized = _Canonicalizer().visit(tree)
        ast.fix_missing_locations(normalized)
        return ast.dump(normalized, annotate_fields=False)
    except Exception:  # noqa: BLE001 - any AST-walk oddity -> treat as unparseable
        return "PARSE_ERROR"
