from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(os.getenv("DEVRAG_REPO", ".")).resolve()
STATIC = Path(__file__).parent / "static"
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build", ".next"}
TEXT_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".h", ".cpp", ".cs", ".rb", ".php", ".swift", ".kt", ".sql", ".md", ".yaml", ".yml", ".toml", ".json", ".html", ".css", ".sh"}
MANIFEST_NAMES = {"requirements.txt", "package.json", "pyproject.toml", "Pipfile", "Pipfile.lock", "poetry.lock", "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "build.gradle.kts"}
MAX_FILE_BYTES = 500_000
REGISTRY_PATH = Path.home() / ".devrag" / "repos.json"
chunks: list[dict[str, Any]] = []
indexed_root: Path | None = None
repo_files: list[dict[str, Any]] = []
active_repo_id: str | None = None
app = FastAPI(title="DevRAG", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class IndexRequest(BaseModel):
    path: str | None = None


class AskRequest(BaseModel):
    question: str
    limit: int = 6


class RepoSelectRequest(BaseModel):
    id: str


def load_registry() -> dict[str, Any]:
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("repos"), list):
            return {"repos": data["repos"], "current_id": data.get("current_id")}
    except (OSError, json.JSONDecodeError):
        pass
    return {"repos": [], "current_id": None}


registry = load_registry()


def save_registry() -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = REGISTRY_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    temp_path.replace(REGISTRY_PATH)


def repository_id(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def split_chunks(path: Path, text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    out = []
    start = 0
    while start < len(lines):
        end = min(start + 48, len(lines))
        # Keep chunks near function/class boundaries where possible.
        if end < len(lines):
            for candidate in range(end, max(start + 24, end - 12), -1):
                if re.match(r"\s*(def |async def |class |function |export (?:async )?function )", lines[candidate - 1]):
                    end = candidate - 1
                    break
        chunk_lines = lines[start:end]
        first = 0
        last = len(chunk_lines)
        while first < last and not chunk_lines[first].strip():
            first += 1
        while last > first and not chunk_lines[last - 1].strip():
            last -= 1
        body = "\n".join(chunk_lines[first:last])
        if body:
            out.append({"path": path.as_posix(), "start": start + first + 1,
                        "end": start + last, "line_numbers": list(range(start + first + 1, start + last + 1)),
                        "text": body})
        start = max(end, start + 1)
    return out


def index_repository(root: Path) -> dict[str, Any]:
    global chunks, indexed_root, repo_files
    if not root.exists() or not root.is_dir():
        raise HTTPException(400, f"Folder not found: {root}")
    found: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    files = 0
    for folder, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith(".")]
        for name in names:
            path = Path(folder) / name
            if path.suffix.lower() not in TEXT_EXTS and name.lower() not in MANIFEST_NAMES:
                continue
            try:
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    continue
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if not content.strip():
                continue
            files += 1
            relative = path.relative_to(root).as_posix()
            catalog.append({"path": relative, "lines": len(content.splitlines()), "bytes": path.stat().st_size})
            found.extend(split_chunks(Path(relative), content))
    chunks = found
    indexed_root = root
    repo_files = sorted(catalog, key=lambda item: item["path"].lower())
    return {"files": files, "chunks": len(found), "root": str(root)}


def terms(text: str) -> list[str]:
    def normalize(word: str) -> str:
        word = word.lower()
        if word.startswith(("dedup", "deduplicat", "duplicate", "duplicat")):
            return "duplicate"
        if len(word) > 5 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 5 and word.endswith("ing"):
            root = word[:-3]
            if len(root) > 2 and root[-1] == root[-2]:
                root = root[:-1]
            return root
        if len(word) > 4 and word.endswith("ied"):
            return word[:-3] + "y"
        if len(word) > 4 and word.endswith("ed"):
            root = word[:-2]
            return root + "e" if root.endswith("at") else root
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word

    return [normalize(w) for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text) if len(w) > 2]


def question_kind(question: str) -> str | None:
    query = set(terms(question))
    if query & {"library", "libraries", "package", "packages", "dependency", "dependencies", "framework", "frameworks"}:
        return "libraries"
    if query & {"api", "apis", "endpoint", "endpoints", "route", "routes"}:
        return "apis"
    if query & {"function", "functions", "method", "methods"}:
        return "functions"
    return None


API_PATTERN = re.compile(
    r"(?:@(?:app|router|bp|blueprint)\.(?:get|post|put|patch|delete|route)|"
    r"\b(?:app|router)\.(?:get|post|put|patch|delete)\s*\(|"
    r"\b(?:fetch|axios\.(?:get|post|put|patch|delete)|requests\.(?:get|post|put|patch|delete)|httpx\.(?:get|post|put|patch|delete))\s*\(|"
    r"\bpath\s*\()"
)


def _source_line(source: dict[str, Any], offset: int) -> int:
    line_numbers = source.get("line_numbers", [])
    return line_numbers[offset - 1] if 0 < offset <= len(line_numbers) else source["start"] + offset - 1


def extract_api_facts(sources: list[dict[str, Any]]) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        for offset, line in enumerate(source["text"].splitlines()):
            cleaned = line.strip()
            if cleaned and API_PATTERN.search(cleaned):
                ref = f"{source['path']}:{_source_line(source, offset + 1)}"
                fact = (cleaned[:180], ref)
                if fact not in seen:
                    seen.add(fact)
                    facts.append(fact)
    return facts


def extract_function_facts(sources: list[dict[str, Any]]) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    declarations = (
        re.compile(r"^\s*(?:export\s+)?(?:(?:async\s+)?function\s+|(?:async\s+)?(?:def|func|fn)\s+)([A-Za-z_$][\w$]*)\s*\(([^)]*)\)"),
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>"),
        re.compile(r"^\s*(?:(?:public|private|protected|static|async)\s+)*([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*\{"),
    )
    for source in sources:
        lines = source["text"].splitlines()
        parsed_python = False
        if Path(source["path"]).suffix.lower() == ".py":
            try:
                tree = ast.parse(source["text"])
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        doc = ast.get_docstring(node, clean=True)
                        purpose = " ".join(doc.split()).split(".")[0] if doc else "No docstring; behavior needs to be checked in the cited code."
                        ref = f"{source['path']}:{_source_line(source, node.lineno)}"
                        fact = (f"`{node.name}()` — {purpose[:140]}", ref)
                        if fact not in seen:
                            seen.add(fact)
                            facts.append(fact)
                parsed_python = True
            except SyntaxError:
                pass
        if parsed_python:
            continue
        for offset, line in enumerate(lines):
            match = next((pattern.match(line) for pattern in declarations if pattern.match(line)), None)
            if not match:
                continue
            name = match.group(1)
            if name in {"if", "for", "while", "switch", "catch", "with", "return"}:
                continue
            ref = f"{source['path']}:{_source_line(source, offset + 1)}"
            purpose = re.sub(r"[_$]+", " ", name).strip()
            fact = (f"`{name}()` — purpose is not documented; the name suggests {purpose}.", ref)
            if fact not in seen:
                seen.add(fact)
                facts.append(fact)
    return facts


def third_party_imports(source: dict[str, Any]) -> list[tuple[int, str]]:
    """Extract unique third-party libraries with chunk-relative line offsets."""
    path = Path(source["path"])
    name = path.name.lower()
    body = source["text"]
    found: list[tuple[int, str]] = []

    if path.suffix.lower() == ".py":
        try:
            tree = ast.parse(body)
        except SyntaxError:
            tree = None
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    found.extend((node.lineno, alias.name.split(".", 1)[0]) for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    found.append((node.lineno, node.module.split(".", 1)[0]))
    elif name.startswith("requirements"):
        for offset, line in enumerate(body.splitlines(), 1):
            entry = line.split("#", 1)[0].split(";", 1)[0].strip()
            if not entry or entry.startswith("-"):
                continue
            match = re.match(r"([A-Za-z0-9][A-Za-z0-9_.-]*)", entry)
            if match:
                found.append((offset, match.group(1)))
    elif name == "package.json":
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict):
            for group in ("dependencies", "devDependencies", "peerDependencies"):
                packages = data.get(group, {})
                if isinstance(packages, dict):
                    for package in packages:
                        match = re.search(rf"['\"]{re.escape(package)}['\"]\s*:", body)
                        offset = body.count("\n", 0, match.start()) + 1 if match else 1
                        found.append((offset, package))
    elif name == "pyproject.toml":
        section = ""
        in_project_dependencies = False
        for offset, line in enumerate(body.splitlines(), 1):
            stripped = line.strip()
            section_match = re.match(r"\[([^]]+)\]", stripped)
            if section_match:
                section = section_match.group(1).lower()
                in_project_dependencies = False
                continue
            if section == "project" and stripped.startswith("dependencies") and "=" in stripped:
                in_project_dependencies = "[" in stripped and "]" not in stripped
                values = re.findall(r"['\"]([A-Za-z0-9_.-]+)(?:\[[^]]*\])?(?:[<>=!~].*)?['\"]", stripped)
                found.extend((offset, package) for package in values)
            elif in_project_dependencies:
                if "]" in stripped:
                    in_project_dependencies = False
                values = re.findall(r"['\"]([A-Za-z0-9_.-]+)(?:\[[^]]*\])?(?:[<>=!~].*)?['\"]", stripped)
                found.extend((offset, package) for package in values)
            elif section == "tool.poetry.dependencies" or (section.endswith(".dependencies") and section.startswith("tool.poetry.group.")):
                match = re.match(r"([A-Za-z0-9_.-]+)\s*=", stripped)
                if match and match.group(1).lower() != "python":
                    found.append((offset, match.group(1)))
    elif path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx"}:
        node_builtins = {"assert", "buffer", "child_process", "crypto", "events", "fs", "http", "https", "os", "path", "stream", "url", "util"}
        pattern = re.compile(r"(?:from\s*|require\s*\(|import\s*\()['\"]([^'\"]+)['\"]")
        for offset, line in enumerate(body.splitlines(), 1):
            for package in pattern.findall(line):
                if package.startswith(".") or package.startswith("/"):
                    continue
                package_root = "/".join(package.split("/")[:2]) if package.startswith("@") else package.split("/", 1)[0]
                if package_root.lower() not in node_builtins:
                    found.append((offset, package_root))
    elif name == "pipfile":
        section = ""
        for offset, line in enumerate(body.splitlines(), 1):
            header = re.match(r"\s*\[([^]]+)\]", line)
            if header:
                section = header.group(1).lower()
            elif section in {"packages", "dev-packages"}:
                match = re.match(r"\s*['\"]?([A-Za-z0-9_.-]+)['\"]?\s*=", line)
                if match:
                    found.append((offset, match.group(1)))

    aliases = {"django": "Django", "rest_framework": "Django REST Framework", "djangorestframework": "Django REST Framework"}
    unique: dict[str, tuple[int, str]] = {}
    for offset, package in found:
        if package.lower() in sys.stdlib_module_names:
            continue
        display = aliases.get(package.lower(), package)
        unique.setdefault(re.sub(r"[-_.]+", "", display).casefold(), (offset, display))
    return sorted(unique.values(), key=lambda item: item[1].casefold())


def extract_library_facts(sources: list[dict[str, Any]]) -> list[tuple[str, str]]:
    facts: dict[str, tuple[str, set[str]]] = {}
    for source in sources:
        for offset, package in third_party_imports(source):
            key = re.sub(r"[-_.]+", "", package).casefold()
            if key not in facts:
                facts[key] = (package, set())
            facts[key][1].add(f"{source['path']}:{_source_line(source, offset)}")
    return [(package, ", ".join(sorted(refs))) for package, refs in sorted(facts.values(), key=lambda item: item[0].casefold())]


def rank(question: str, limit: int = 6) -> list[dict[str, Any]]:
    query = set(terms(question))
    if not query:
        return []
    kind = question_kind(question)
    path_terms_by_file: dict[str, set[str]] = {}
    for item in chunks:
        path_terms_by_file.setdefault(item["path"], set()).update(terms(item["path"]))
    path_frequency: dict[str, int] = {}
    for file_terms in path_terms_by_file.values():
        for word in file_terms:
            path_frequency[word] = path_frequency.get(word, 0) + 1
    common_path_terms = {
        word for word, count in path_frequency.items()
        if count >= max(2, (len(path_terms_by_file) + 3) // 4)
    }
    scored = []
    for item in chunks:
        body = item["text"].lower()
        path_words = set(terms(item["path"]))
        words = set(terms(body))
        overlap = query & words
        is_manifest = Path(item["path"]).name.lower() in MANIFEST_NAMES
        has_import = bool(re.search(r"(?m)^\s*(?:import\s+|from\s+\S+\s+import\s+|require\s*\(|.*\bfrom\s*['\"])", item["text"]))
        has_api = bool(API_PATTERN.search(item["text"]))
        has_function = bool(re.search(r"(?m)^\s*(?:(?:async\s+)?def\s+|(?:export\s+)?(?:async\s+)?function\s+|(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)", item["text"]))
        if kind == "libraries" and (is_manifest or has_import):
            score = 12 if is_manifest else 5
        elif kind == "apis" and has_api:
            score = 10
        elif kind == "functions" and has_function:
            score = 10
        elif kind == "libraries" or kind == "apis" or kind == "functions":
            continue
        else:
            score = 0
        if not overlap and kind not in {"libraries", "apis", "functions"}:
            continue
        if kind is None:
            score = len(overlap)
            score += sum(1.5 for word in query & path_words if word not in common_path_terms)
            normalized_words = terms(item["text"])
            score += sum(0.5 for word in overlap if normalized_words.count(word) > 1)
        ranked_item = {**item, "match_lines": []}
        for offset, line in enumerate(item["text"].splitlines()):
            if query & set(terms(line)) and offset < len(item.get("line_numbers", [])):
                ranked_item["match_lines"].append(item["line_numbers"][offset])
        if kind == "apis":
            api_lines = [item["line_numbers"][offset] for offset, line in enumerate(item["text"].splitlines())
                         if API_PATTERN.search(line) and offset < len(item.get("line_numbers", []))]
            if api_lines:
                ranked_item["match_lines"] = api_lines
        if kind == "functions":
            function_lines = [int(ref.rsplit(":", 1)[1]) for _, ref in extract_function_facts([item])]
            if function_lines:
                ranked_item["match_lines"] = function_lines
        if kind == "libraries":
            import_lines = [item["line_numbers"][offset - 1] for offset, _ in third_party_imports(item)
                            if offset - 1 < len(item.get("line_numbers", []))]
            if import_lines:
                ranked_item["match_lines"] = import_lines
        if not ranked_item["match_lines"]:
            ranked_item["match_lines"] = [item["start"]]
        scored.append((score / (len(query) ** 0.45), ranked_item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**item, "score": round(score, 2)} for score, item in scored[:limit]]


def symbols_for(source: dict[str, Any]) -> list[str]:
    text, ext = source["text"], Path(source["path"]).suffix.lower()
    names: list[str] = []
    if ext == ".py":
        try:
            tree = ast.parse(text)
            names = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        except SyntaxError:
            pass
    if not names:
        names = re.findall(r"(?:function|class|def|const|let|var)\s+([A-Za-z_$][\w$]*)", text)
    return names[:12]


def local_notes(source: dict[str, Any], question: str) -> list[str]:
    """Build cautious, code-grounded notes without sending code to a hosted model."""
    text = source["text"]
    path = source["path"].lower()
    notes: list[str] = []
    roles = [
        (("auth", "login", "security", "permission", "token"), "authentication or access control"),
        (("database", "db", "repository", "query", "model"), "data storage or database access"),
        (("route", "api", "controller", "endpoint", "view"), "request handling"),
        (("service",), "application service logic"),
        (("parser", "ingest", "loader", "index"), "parsing or data ingestion"),
        (("task", "worker", "job", "queue"), "background task processing"),
        (("config", "setting"), "application configuration"),
        (("test", "spec"), "automated tests"),
    ]
    for words, label in roles:
        if any(word in path for word in words):
            notes.append(f"The file path suggests this code handles {label}.")
            break

    # Prefer real Python docstrings; they are the clearest offline description.
    if Path(source["path"]).suffix.lower() == ".py":
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    doc = ast.get_docstring(node, clean=True)
                    if doc:
                        one_line = " ".join(doc.split())
                        notes.append(f"`{node.name}` is documented as: {one_line[:220]}")
                        if len(notes) >= 3:
                            break
        except SyntaxError:
            pass

    names = symbols_for(source)
    if names:
        label = "Functions or classes in this excerpt: "
        if not any(note.startswith(label) for note in notes):
            notes.append(label + ", ".join(f"`{name}`" for name in names[:5]) + ".")

    if not any("documented as:" in note for note in notes):
        # Surface operations as evidence rather than guessing their purpose.
        patterns = re.compile(r"\b(await|return|raise|yield|execute|commit|fetch|request|connect|send|publish|save|delete|insert|select|authenticate|login|verify)\b", re.I)
        operations = []
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned and not cleaned.startswith(("#", "//", "*")) and patterns.search(cleaned):
                operations.append(cleaned)
            if len(operations) == 2:
                break
        if operations:
            notes.append("Visible operations include " + "; ".join(f"`{line[:110]}`" for line in operations) + ".")
    return notes[:3]


def answer_text(question: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "I couldn’t find a strong match in the indexed files. Try a specific function, module, or domain term, or re-index the repository."
    kind = question_kind(question)
    if kind == "apis":
        facts = extract_api_facts(sources)
        if facts:
            return "I found API routes and external API calls in the indexed code:\n\n" + "\n".join(f"- `{fact}` — `{ref}`" for fact, ref in facts[:30])
        return "I didn’t find recognizable API routes or HTTP calls in the retrieved code. Try asking about endpoints, routes, or a specific HTTP client."
    if kind == "functions":
        facts = extract_function_facts(sources)
        if facts:
            return "Here are functions I found and what the code says about them:\n\n" + "\n".join(f"- {fact} — `{ref}`" for fact, ref in facts[:30])
        return "I didn’t find function definitions in the retrieved code. Try naming a module or asking for a specific function."
    if kind == "libraries":
        facts = extract_library_facts(sources)
        if facts:
            lines = ["I found these libraries or packages in the indexed repository:", ""]
            for package, refs in facts:
                lines.append(f"- **`{package}`** — " + ", ".join(f"`{ref}`" for ref in refs.split(", ")))
            return "\n".join(lines)
        return "I found import statements, but no dependency manifest in the retrieved files confirmed the libraries. Check the references below or index the repository again."
    # If configured, use OpenAI for synthesis while keeping all citations grounded in retrieved chunks.
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            context = "\n\n".join(f"[{s['path']}:{s['start']}-{s['end']}]\n{s['text']}" for s in sources)
            result = OpenAI(api_key=api_key).chat.completions.create(
                model=os.getenv("DEVRAG_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "Answer using only the supplied code excerpts. Cite each claim as [path:line]. If the excerpts do not establish something, say what is missing. Keep the answer concise."},
                    {"role": "user", "content": f"Question: {question}\n\nCode excerpts:\n{context}"},
                ], temperature=0.2,
            )
            return result.choices[0].message.content or "Relevant code was found, but no answer was generated."
        except Exception:
            pass
    lines = ["Here’s a local explanation based on the matching code. It uses names, comments, and visible operations in the files:", ""]
    for i, source in enumerate(sources[:4], 1):
        lines.append(f"{i}. **`{source['path']}:{source['start']}`**")
        notes = local_notes(source, question)
        lines.extend(f"   - {note}" for note in notes)
        if not notes:
            lines.append("   - This excerpt contains code matching the terms in your question; inspect the reference below for its exact behavior.")
    return "\n".join(lines)


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {"indexed": indexed_root is not None, "root": str(indexed_root) if indexed_root else None, "repo_id": active_repo_id, "files": len(repo_files), "chunks": len(chunks), "llm": bool(os.getenv("OPENAI_API_KEY"))}


@app.get("/api/repos")
def list_repositories():
    return {"repos": registry["repos"], "current_id": registry.get("current_id")}


@app.post("/api/repos")
def add_repository(req: IndexRequest):
    global active_repo_id
    path = Path(req.path).expanduser().resolve() if req.path else ROOT
    info = index_repository(path)
    repo_id = repository_id(path)
    repo = {"id": repo_id, "name": path.name or str(path), "path": str(path)}
    registry["repos"] = [item for item in registry["repos"] if item.get("id") != repo_id]
    registry["repos"].append(repo)
    registry["current_id"] = repo_id
    active_repo_id = repo_id
    save_registry()
    return {"repo": repo, "info": info, "repos": registry["repos"], "current_id": repo_id}


@app.post("/api/repos/select")
def select_repository(req: RepoSelectRequest):
    global active_repo_id
    repo = next((item for item in registry["repos"] if item.get("id") == req.id), None)
    if not repo:
        raise HTTPException(404, "Saved repository not found.")
    path = Path(repo["path"]).expanduser().resolve()
    info = index_repository(path)
    registry["current_id"] = repo["id"]
    active_repo_id = repo["id"]
    save_registry()
    return {"repo": repo, "info": info}


@app.get("/api/files")
def list_source_files():
    if indexed_root is None:
        raise HTTPException(400, "Select a repository first.")
    return {"root": str(indexed_root), "files": repo_files}


@app.get("/api/files/content")
def read_source_file(path: str):
    if indexed_root is None:
        raise HTTPException(400, "Select a repository first.")
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(400, "Invalid source file path.")
    normalized = relative.as_posix()
    if not any(item["path"] == normalized for item in repo_files):
        raise HTTPException(404, "Source file not found in the current repository.")
    resolved = (indexed_root / relative).resolve()
    if not resolved.is_relative_to(indexed_root):
        raise HTTPException(400, "Invalid source file path.")
    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(404, "Could not read source file.") from exc
    return {"path": normalized, "content": content}


@app.post("/api/index")
def index(req: IndexRequest):
    result = add_repository(req)
    return result["info"]


@app.post("/api/ask")
def ask(req: AskRequest):
    if not chunks:
        raise HTTPException(400, "Index a repository before asking a question.")
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Enter a question first.")
    sources = rank(question, max(1, min(req.limit, 10)))
    return {"answer": answer_text(question, sources), "sources": sources, "question": question}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=True)
