from __future__ import annotations

import ast
import hashlib
import json
import os
import re
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
        body = "\n".join(lines[start:end]).strip()
        if body:
            out.append({"path": path.as_posix(), "start": start + 1, "end": end, "text": body})
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
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in names:
            path = Path(folder) / name
            if path.suffix.lower() not in TEXT_EXTS and name.lower() not in MANIFEST_NAMES:
                continue
            try:
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
    return [w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text) if len(w) > 2]


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
    r"(?i)(?:@(?:app|router|bp|blueprint)\.(?:get|post|put|patch|delete|route)|"
    r"\b(?:app|router)\.(?:get|post|put|patch|delete)\s*\(|"
    r"\b(?:fetch|axios\.(?:get|post|put|patch|delete)|requests\.(?:get|post|put|patch|delete)|httpx\.(?:get|post|put|patch|delete))\s*\(|"
    r"\bpath\s*\()"
)


def extract_api_facts(sources: list[dict[str, Any]]) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        for offset, line in enumerate(source["text"].splitlines()):
            cleaned = line.strip()
            if cleaned and API_PATTERN.search(cleaned):
                ref = f"{source['path']}:{source['start'] + offset}"
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
                        ref = f"{source['path']}:{source['start'] + node.lineno - 1}"
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
            ref = f"{source['path']}:{source['start'] + offset}"
            purpose = re.sub(r"[_$]+", " ", name).strip()
            fact = (f"`{name}()` — purpose is not documented; the name suggests {purpose}.", ref)
            if fact not in seen:
                seen.add(fact)
                facts.append(fact)
    return facts


def rank(question: str, limit: int = 6) -> list[dict[str, Any]]:
    query = set(terms(question))
    if not query:
        return []
    kind = question_kind(question)
    scored = []
    for item in chunks:
        body = item["text"].lower()
        path = item["path"].lower()
        words = set(terms(body))
        overlap = query & words
        is_manifest = Path(item["path"]).name.lower() in MANIFEST_NAMES
        has_import = bool(re.search(r"(?m)^\s*(?:import\s+|from\s+\S+\s+import\s+|require\s*\(|.*\bfrom\s*['\"])", item["text"]))
        has_api = bool(API_PATTERN.search(item["text"]))
        has_function = bool(re.search(r"(?m)^\s*(?:(?:async\s+)?def\s+|(?:export\s+)?(?:async\s+)?function\s+|(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)", item["text"]))
        if kind == "libraries" and (is_manifest or has_import):
            score = 12 if is_manifest else 5
            scored.append((score, item))
            continue
        if kind == "apis" and has_api:
            scored.append((10, item))
            continue
        if kind == "functions" and has_function:
            scored.append((10, item))
            continue
        if not overlap:
            continue
        score = sum(2.5 if word in path else 1 for word in overlap)
        score += sum(1.5 for word in overlap if body.count(word) > 1)
        scored.append((score / (len(query) ** 0.45), item))
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
        found: dict[str, set[str]] = {}
        for source in sources:
            path = source["path"]
            body = source["text"]
            name = Path(path).name.lower()
            packages: list[str] = []
            if name == "package.json":
                try:
                    data = json.loads(body)
                    for group in ("dependencies", "devDependencies", "peerDependencies"):
                        packages.extend(data.get(group, {}).keys())
                except (json.JSONDecodeError, AttributeError):
                    pass
            elif name == "requirements.txt" or name == "pipfile":
                packages.extend(re.findall(r"(?m)^\s*([A-Za-z0-9_.-]+)\s*(?:[<>=!~].*)?$", body))
            elif name in {"pyproject.toml", "poetry.lock", "pipfile.lock", "cargo.toml", "go.mod", "pom.xml", "build.gradle", "build.gradle.kts"}:
                packages.extend(re.findall(r"(?m)^\s*([A-Za-z][\w.-]+)\s*(?:=|:|\s+)", body))
            else:
                packages.extend(re.findall(r"(?m)^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+)|.*?\bfrom\s*['\"]([^'\"]+)['\"])", body))
                packages = [next((part for part in match if part), "").split(".")[0] for match in packages]
            for package in packages:
                clean = package.strip().strip("\"'")
                if clean and clean.lower() not in {"dependencies", "devdependencies", "peerdependencies", "name", "version", "python", "node", "true", "false"}:
                    found.setdefault(clean, set()).add(f"{path}:{source['start']}")
        if found:
            lines = ["I found these libraries or packages in the indexed repository:", ""]
            for package, refs in sorted(found.items(), key=lambda item: item[0].lower()):
                lines.append(f"- **`{package}`** — " + ", ".join(f"`{ref}`" for ref in sorted(refs)))
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
