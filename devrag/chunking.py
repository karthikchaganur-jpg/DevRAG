from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


MAX_CHUNK_LINES = 48
OVERLAP_LINES = 8


def _make_chunk(path: Path, lines: list[str], start: int, end: int, **metadata: Any) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "start": start,
        "end": end,
        "line_numbers": list(range(start, end + 1)),
        "text": "\n".join(lines),
        **metadata,
    }


def _trimmed_chunk(path: Path, source_lines: list[str], start_index: int, end_index: int, **metadata: Any) -> dict[str, Any] | None:
    first, last = start_index, end_index
    while first < last and not source_lines[first].strip():
        first += 1
    while last > first and not source_lines[last - 1].strip():
        last -= 1
    if first == last:
        return None
    return _make_chunk(path, source_lines[first:last], first + 1, last, **metadata)


def _line_chunks(path: Path, lines: list[str], start: int, end: int, *, symbol: str | None = None) -> list[dict[str, Any]]:
    output = []
    cursor = start
    while cursor < end:
        stop = min(cursor + MAX_CHUNK_LINES, end)
        chunk = _trimmed_chunk(path, lines, cursor, stop, symbol=symbol) if symbol else _trimmed_chunk(path, lines, cursor, stop)
        if chunk:
            output.append(chunk)
        if stop == end:
            break
        cursor = max(cursor + 1, stop - OVERLAP_LINES)
    return output


def _python_chunks(path: Path, lines: list[str]) -> list[dict[str, Any]]:
    source = "\n".join(lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    def node_start(node: ast.AST) -> int:
        decorators = getattr(node, "decorator_list", [])
        return min([node.lineno, *(decorator.lineno for decorator in decorators)])
    if not definitions:
        return []

    chunks: list[dict[str, Any]] = []
    preamble_end = node_start(definitions[0]) - 1
    chunks.extend(_line_chunks(path, lines, 0, preamble_end, symbol="module context"))
    definition_ranges: list[tuple[int, int]] = []
    for node in definitions:
        start = node_start(node) - 1
        stop = getattr(node, "end_lineno", node.lineno)
        definition_ranges.append((start, stop))
        if isinstance(node, ast.ClassDef):
            methods = [child for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if methods:
                class_header_end = node_start(methods[0]) - 1
                if class_header_end > start:
                    chunks.extend(_line_chunks(path, lines, start, class_header_end, symbol=node.name))
                previous_stop = class_header_end
                for method in methods:
                    method_start = node_start(method) - 1
                    method_stop = getattr(method, "end_lineno", method.lineno)
                    symbol = f"{node.name}.{method.name}"
                    if method_start > previous_stop:
                        chunks.extend(_line_chunks(path, lines, previous_stop, method_start, symbol=node.name))
                    chunks.extend(_line_chunks(path, lines, method_start, method_stop, symbol=symbol))
                    previous_stop = method_stop
                if previous_stop < stop:
                    chunks.extend(_line_chunks(path, lines, previous_stop, stop, symbol=node.name))
            else:
                chunks.extend(_line_chunks(path, lines, start, stop, symbol=node.name))
        else:
            chunks.extend(_line_chunks(path, lines, start, stop, symbol=node.name))

    for (_, previous_stop), (next_start, _) in zip(definition_ranges, definition_ranges[1:]):
        if next_start > previous_stop:
            chunks.extend(_line_chunks(path, lines, previous_stop, next_start, symbol="module context"))
    if definition_ranges and definition_ranges[-1][1] < len(lines):
        chunks.extend(_line_chunks(path, lines, definition_ranges[-1][1], len(lines), symbol="module context"))
    return chunks


def _markdown_chunks(path: Path, lines: list[str]) -> list[dict[str, Any]]:
    heading_indexes = [i for i, line in enumerate(lines) if re.match(r"^\s{0,3}#{1,6}\s+", line)]
    if not heading_indexes:
        return []
    chunks = _line_chunks(path, lines, 0, heading_indexes[0], symbol="README introduction") if heading_indexes[0] else []
    for index, start in enumerate(heading_indexes):
        stop = heading_indexes[index + 1] if index + 1 < len(heading_indexes) else len(lines)
        heading = re.sub(r"^\s{0,3}#{1,6}\s+", "", lines[start]).strip()
        chunks.extend(_line_chunks(path, lines, start, stop, symbol=heading or "Markdown section"))
    return chunks


def _config_chunks(path: Path, lines: list[str]) -> list[dict[str, Any]]:
    ext = path.suffix.lower()
    if ext == ".toml":
        starts = [i for i, line in enumerate(lines) if re.match(r"^\s*\[[^]]+\]\s*$", line)]
    elif ext in {".yaml", ".yml"}:
        starts = [i for i, line in enumerate(lines) if re.match(r"^[A-Za-z0-9_.-]+\s*:\s*(?:#.*)?$", line)]
    else:
        return []
    if not starts:
        return []
    chunks = _line_chunks(path, lines, 0, starts[0], symbol="configuration header") if starts[0] else []
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(lines)
        label = lines[start].strip().strip("[]:") or "configuration section"
        chunks.extend(_line_chunks(path, lines, start, stop, symbol=label))
    return chunks


def split_chunks(path: Path, text: str) -> list[dict[str, Any]]:
    """Split source by symbols/sections first; use overlapped line windows as fallback."""
    lines = text.splitlines()
    if not lines:
        return []
    ext = path.suffix.lower()
    chunks: list[dict[str, Any]] = []
    if ext == ".py":
        chunks = _python_chunks(path, lines)
    elif ext == ".md":
        chunks = _markdown_chunks(path, lines)
    elif ext in {".toml", ".yaml", ".yml"}:
        chunks = _config_chunks(path, lines)
    return chunks or _line_chunks(path, lines, 0, len(lines))
