"""Static checks on a submission before any of its code runs.

These are a defense layer, not a sandbox. Each rule exists because of a
concrete exploit (see redteam/ and README "Threat model").
"""
import ast

ALLOWED_IMPORTS = {"torch", "math", "typing", "functools", "collections", "dataclasses"}

BANNED_NAMES = {
    # introspection / escaping the harness to find the held-out split or labels
    "inspect", "gc", "sys", "os", "subprocess", "builtins", "importlib",
    "__import__", "eval", "exec", "compile", "open", "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
    # modular arithmetic: a neural net never needs it; a symbolic cheat always does
    "divmod", "pow",
}

BANNED_ATTRS = {
    # capturing dL/dlogits leaks per-example labels to the submission
    "register_hook", "register_full_backward_hook", "register_backward_hook",
    "register_full_backward_pre_hook", "register_forward_hook", "register_forward_pre_hook",
    "Function",  # torch.autograd.Function: custom backward = label leak
    # modular arithmetic primitives
    "remainder", "fmod", "floor_divide", "pow",
    # frame walking
    "f_back", "f_locals", "f_globals", "gi_frame",
}

ALLOWED_DUNDERS = {"__init__", "__name__"}


def check(source: str) -> list[str]:
    errors = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    # Bare string statements (docstrings) have no runtime effect; don't flag them.
    # (Found by red-teaming: a docstring mentioning a dunder was a false positive.)
    inert = {id(n.value) for n in ast.walk(tree)
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}

    for node in ast.walk(tree):
        if id(node) in inert:
            continue
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    errors.append(f"line {node.lineno}: import of '{a.name}' not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                errors.append(f"line {node.lineno}: import from '{node.module}' not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errors.append(f"line {node.lineno}: use of '{node.id}' not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_ATTRS:
                errors.append(f"line {node.lineno}: attribute '.{node.attr}' not allowed")
            if node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS:
                errors.append(f"line {node.lineno}: dunder attribute '.{node.attr}' not allowed")
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, (ast.Mod, ast.FloorDiv, ast.Pow)):
            # '%' on strings is also blocked; use f-strings
            errors.append(f"line {node.lineno}: operator '{type(node.op).__name__}' not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "__" in node.value:
            errors.append(f"line {node.lineno}: string containing '__' not allowed")
    return errors


if __name__ == "__main__":
    import sys
    errs = check(open(sys.argv[1]).read())
    print("\n".join(errs) or "ok")
