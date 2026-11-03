"""Load selected functions from the frozen v1.47 script without executing the whole file."""
from __future__ import annotations

import ast
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V147 = os.path.join(ROOT, "legacy", "v1_47_paper", "agente_egipsio_langgraph_v1_47.py")


def load_v147(*names: str, constants: tuple = ()) -> dict:
    tree = ast.parse(open(V147, encoding="utf-8").read())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            keep.append(node)
    ns = {"re": re, "List": list, "Dict": dict}
    exec(compile(ast.Module(body=keep, type_ignores=[]), V147, "exec"), ns)
    return ns
