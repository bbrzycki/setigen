from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "setigen"
EXEMPT_DUNDERS = {
    "__len__",
    "__iter__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__str__",
    "__repr__",
    "__getstate__",
}


@dataclass(frozen=True)
class CallableAudit:
    """Represent one top-level function or direct class method under audit."""

    module_path: Path
    qualname: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    is_property: bool
    uses_copy_docstring: bool


def _iter_runtime_modules() -> list[Path]:
    """Return runtime-package modules that should participate in the audit."""
    return sorted(
        path for path in RUNTIME_ROOT.rglob("*.py")
        if path.name != "__init__.py"
    )


def _iter_direct_callables(module_path: Path) -> tuple[list[type[ast.ClassDef]], list[CallableAudit]]:
    """Collect direct classes, top-level functions, and direct methods from one module.

    Args:
        module_path: Runtime module to inspect.

    Returns:
        Tuple of class nodes and callable audit records.
    """
    tree = ast.parse(module_path.read_text())
    classes: list[type[ast.ClassDef]] = []
    callables: list[CallableAudit] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append(node)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    decorator_names = {
                        ast.unparse(decorator) for decorator in child.decorator_list
                    }
                    callables.append(
                        CallableAudit(
                            module_path=module_path,
                            qualname=f"{node.name}.{child.name}",
                            node=child,
                            is_property="property" in decorator_names,
                            uses_copy_docstring=any(
                                "_copy_docstring" in name for name in decorator_names
                            ),
                        )
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            decorator_names = {
                ast.unparse(decorator) for decorator in node.decorator_list
            }
            callables.append(
                CallableAudit(
                    module_path=module_path,
                    qualname=node.name,
                    node=node,
                    is_property=False,
                    uses_copy_docstring=any(
                        "_copy_docstring" in name for name in decorator_names
                    ),
                )
            )

    return classes, callables


def _has_full_signature_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether a function signature is fully annotated.

    Args:
        node: Function or method AST node.

    Returns:
        Whether all relevant parameters and the return value are annotated.
    """
    positional = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    parameters = [arg for arg in positional if arg.arg not in {"self", "cls"}]

    if any(arg.annotation is None for arg in parameters):
        return False
    if node.args.vararg is not None and node.args.vararg.annotation is None:
        return False
    if node.args.kwarg is not None and node.args.kwarg.annotation is None:
        return False
    return node.returns is not None


def _docstring_requires_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Determine whether a callable docstring should have an ``Args`` section.

    Args:
        node: Function or method AST node.

    Returns:
        Whether the callable takes any parameters beyond ``self`` or ``cls``.
    """
    positional = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    parameters = [arg for arg in positional if arg.arg not in {"self", "cls"}]
    return bool(parameters or node.args.vararg or node.args.kwarg)


def _docstring_requires_returns(record: CallableAudit) -> bool:
    """Determine whether a callable docstring should have a return section.

    Args:
        record: Callable audit record.

    Returns:
        Whether the callable should document a return or yielded value.
    """
    if record.node.name == "__init__" or record.is_property:
        return False
    if isinstance(record.node.returns, ast.Constant) and record.node.returns.value is None:
        return False
    return True


def test_runtime_classes_and_signatures_are_typed_and_documented() -> None:
    """Audit runtime modules for typed top-level signatures and Google docstrings."""
    failures: list[str] = []

    for module_path in _iter_runtime_modules():
        classes, callables = _iter_direct_callables(module_path)
        relpath = module_path.relative_to(RUNTIME_ROOT)

        for class_node in classes:
            if not ast.get_docstring(class_node):
                failures.append(f"{relpath}:{class_node.lineno} missing class docstring for {class_node.name}")

        for record in callables:
            node = record.node
            if not _has_full_signature_annotations(node):
                failures.append(f"{relpath}:{node.lineno} missing full type annotations for {record.qualname}")

            if node.name in EXEMPT_DUNDERS or record.uses_copy_docstring:
                continue

            docstring = ast.get_docstring(node)
            if not docstring:
                failures.append(f"{relpath}:{node.lineno} missing docstring for {record.qualname}")
                continue

            if _docstring_requires_args(node) and "Args:" not in docstring:
                failures.append(f"{relpath}:{node.lineno} missing Google-style Args section for {record.qualname}")
            if _docstring_requires_returns(record) and all(section not in docstring for section in ("Returns:", "Yields:")):
                failures.append(f"{relpath}:{node.lineno} missing Google-style Returns/Yields section for {record.qualname}")

    assert not failures, "Docstring/type audit failures:\n" + "\n".join(failures)
