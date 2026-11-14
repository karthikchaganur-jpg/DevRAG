from __future__ import annotations

import ast
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
MAX_FILE_BYTES = 500_000
chunks: list[dict[str, Any]] = []
indexed_root: Path | None = None
app = FastAPI(title="DevRAG", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class IndexRequest(BaseModel):
    path: str | None = None


class AskRequest(BaseModel):
    question: str
    limit: int = 6


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
    global chunks, indexed_root
    if not root.exists() or not root.is_dir():
        raise HTTPException(400, f"Folder not found: {root}")
    found: list[dict[str, Any]] = []
    files = 0
    for folder, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in names:
            path = Path(folder) / name
            if path.suffix.lower() not in TEXT_EXTS:
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
            found.extend(split_chunks(path.relative_to(root), content))
    chunks = found
    indexed_root = root
    return {"files": files, "chunks": len(found), "root": str(root)}


def terms(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,}", text) if len(w) > 2]


def rank(question: str, limit: int = 6) -> list[dict[str, Any]]:
    query = set(terms(question))
    if not query:
        return []
    scored = []
    for item in chunks:
        body = item["text"].lower()
        path = item["path"].lower()
        words = set(terms(body))
        overlap = query & words
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
    return {"indexed": indexed_root is not None, "root": str(indexed_root) if indexed_root else None, "files": len({c['path'] for c in chunks}), "chunks": len(chunks), "llm": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/api/index")
def index(req: IndexRequest):
    path = Path(req.path).expanduser().resolve() if req.path else ROOT
    return index_repository(path)


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
