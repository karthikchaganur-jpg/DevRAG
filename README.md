# DevRAG

DevRAG indexes local source repositories, retrieves relevant code excerpts, and answers questions with file and line references. You can save multiple repository paths, switch between them, and browse the current repository's source files in the UI. It runs locally; source files stay on your machine. Saved paths are kept in `~/.devrag/repos.json`. An OpenAI API key is optional and only used to synthesize answers from the retrieved excerpts.

## Run locally

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://127.0.0.1:8000. Enter the repository path and click **Index repository**. Set `OPENAI_API_KEY` in your environment to enable generated explanations; without it, DevRAG returns ranked code locations and symbol hints. Use a trusted local repository because enabling an LLM sends retrieved excerpts to the configured OpenAI API.

## API

- `POST /api/index` with `{ "path": "C:/path/to/repository" }`
- `GET /api/repos` and `POST /api/repos` to list and save local repositories
- `POST /api/repos/select` with `{ "id": "..." }` to switch the active repository
- `GET /api/files` and `GET /api/files/content?path=...` to browse indexed source files
- `POST /api/ask` with `{ "question": "Where is authentication handled?", "limit": 6 }`
- `GET /api/status`

The index is held in memory and rebuilt when the server restarts. Supported source/config formats include Python, TypeScript, JavaScript, Java, Go, Rust, C/C++, C#, Ruby, PHP, Swift, Kotlin, SQL, Markdown, and common config files.
