from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "dist", "build", ".next"}
TEXT_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".h", ".cpp", ".cs", ".rb", ".php",
    ".swift", ".kt", ".sql", ".md", ".yaml", ".yml", ".toml", ".json", ".html", ".css", ".sh",
}
MANIFEST_NAMES = {
    "requirements.txt", "package.json", "pyproject.toml", "pipfile", "pipfile.lock", "poetry.lock", "cargo.toml",
    "go.mod", "pom.xml", "build.gradle", "build.gradle.kts",
}
MAX_FILE_BYTES = 500_000
INDEX_DIR = Path.home() / ".devrag"
DATABASE_PATH = INDEX_DIR / "index.sqlite3"
REGISTRY_PATH = INDEX_DIR / "repos.json"
EMBEDDING_MODEL = os.getenv("DEVRAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_MODEL_CACHE = INDEX_DIR / "models"
