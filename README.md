# DevRAG

DevRAG indexes local source repositories, retrieves relevant code excerpts, and answers questions with file and line references. It combines keyword ranking with local semantic embeddings, stores chunks and vectors in SQLite, and avoids re-embedding files whose timestamps and sizes have not changed. You can save multiple repository paths, switch between them, and browse the current repository's source files in the UI. Source code and vectors stay on your machine. Saved paths are kept in `~/.devrag/repos.json`; the persistent index is `~/.devrag/index.sqlite3`. Gemini or OpenAI API keys are optional and used only to synthesize answers from retrieved excerpts.

Code is split into focused modules: `devrag/chunking.py` handles function and section chunks, `devrag/indexer.py` updates the repository index, `devrag/embeddings.py` creates local vectors, `devrag/store.py` persists the index, `devrag/retrieval.py` combines semantic and keyword results, and `devrag/generator.py` handles optional Gemini/OpenAI synthesis. The FastEmbed model runs locally on CPU. Its first use downloads model files (about 67 MB); after that, embedding and search work locally without an API key or per-query charges. If the model cannot be loaded, DevRAG keeps working in keyword-only mode.

## Run locally

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000. Enter the repository path and click **Index repository**. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` to enable generated explanations with Gemini. DevRAG chooses Gemini automatically when both provider keys are configured. You can choose a provider with `DEVRAG_PROVIDER=gemini`, `openai`, `none`, or `auto`; without an API key, it returns its local code-based explanation. Only retrieved excerpts are sent to the selected AI provider. Check its data-use terms before sending private code; Google's free tier is limited and may use submitted content to improve services.

## Run tests

```powershell
py -m unittest discover -s tests -v
```

## Run tests

```powershell
py -m unittest discover -s tests -v
```

You can also ask about API routes and HTTP calls, or ask what functions are defined and what they do. Function summaries use docstrings when available; otherwise DevRAG reports the function name and points to its source so you can verify its behavior.

## API

- `POST /api/index` with `{ "path": "C:/path/to/repository" }`
- `GET /api/repos` and `POST /api/repos` to list and save local repositories
- `POST /api/repos/select` with `{ "id": "..." }` to switch the active repository
- `GET /api/files` and `GET /api/files/content?path=...` to browse indexed source files
- `POST /api/ask` with `{ "question": "Where is authentication handled?", "limit": 6 }`
- `GET /api/status`

Chunks, file-to-line maps, and local vectors persist in SQLite. When you select a saved repository after restarting DevRAG, it checks file timestamps and sizes and only re-reads changed files; unchanged chunks and vectors are reused. Supported source/config formats include Python, TypeScript, JavaScript, Java, Go, Rust, C/C++, C#, Ruby, PHP, Swift, Kotlin, SQL, Markdown, and common config files.
