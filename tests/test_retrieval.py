import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

import main
from devrag.generator import configured_provider


class StubEmbedder:
    model_name = "test-model"

    def __init__(self):
        self.passage_calls = 0

    def embed_passages(self, texts):
        self.passage_calls += len(texts)
        return [[1.0, 0.0] if "duplicate" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [0.0, 1.0]


class RetrievalFixTests(unittest.TestCase):
    def setUp(self):
        self.original_chunks = main.chunks

    def tearDown(self):
        main.chunks = self.original_chunks

    def test_duplicate_word_forms_match_deduplication(self):
        self.assertEqual(
            main.terms("duplicate duplicates dedup deduplicating deduplication"),
            ["duplicate"] * 5,
        )

    def test_provider_auto_prefers_gemini_and_can_be_selected(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "g-key", "OPENAI_API_KEY": "o-key"}, clear=True):
            self.assertEqual(configured_provider(), "gemini")
            os.environ["DEVRAG_PROVIDER"] = "openai"
            self.assertEqual(configured_provider(), "openai")
            os.environ["DEVRAG_PROVIDER"] = "none"
            self.assertIsNone(configured_provider())

    def test_provider_without_keys_uses_local_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(configured_provider())

    def test_common_directory_name_does_not_create_matches(self):
        main.chunks = [
            {"path": f"findings/{name}.py", "start": 1, "end": 1, "line_numbers": [1], "text": "def run(): pass"}
            for name in ("apps", "views", "tasks")
        ]
        self.assertEqual(main.rank("findings"), [])

    def test_duplicate_question_ranks_deduplication_logic(self):
        target = {
            "path": "findings/ingest/service.py", "start": 20, "end": 22,
            "line_numbers": [20, 21, 22],
            "text": "def remove_duplicates(items):\n    # Deduplicate findings by asset\n    return unique(items)",
        }
        distractor = {
            "path": "findings/apps.py", "start": 1, "end": 1,
            "line_numbers": [1], "text": "class FindingsConfig: pass",
        }
        main.chunks = [distractor, target]
        ranked = main.rank("How are duplicate findings removed?")
        self.assertTrue(ranked)
        self.assertEqual(ranked[0]["path"], target["path"])
        self.assertEqual(ranked[0]["match_lines"], [21])

    def test_pathlib_path_is_not_reported_as_api_route(self):
        source = {"path": "config/settings.py", "start": 1, "line_numbers": [1],
                  "text": "ROOT = Path(__file__).resolve().parent"}
        self.assertEqual(main.extract_api_facts([source]), [])
        self.assertIn("didn’t find recognizable API routes", main.answer_text("what APIs are used", [source]))

    def test_python_routes_are_case_sensitive_and_keep_line_number(self):
        source = {"path": "urls.py", "start": 10, "line_numbers": [10, 11, 12],
                  "text": 'urlpatterns = [\n    path("api/v1/", view),\n]'}
        self.assertEqual(main.extract_api_facts([source]), [('path("api/v1/", view),', "urls.py:11")])
        main.chunks = [source]
        results = main.rank("show api routes")
        self.assertEqual(results[0]["match_lines"], [11])

    def test_dictionary_get_is_not_mistaken_for_route(self):
        source = {"path": "config.py", "start": 1,
                  "text": 'settings = {"home": "/"}\nvalue = settings.get("/home")'}
        self.assertEqual(main.extract_api_facts([source]), [])

    def test_library_extraction_filters_stdlib_and_deduplicates_names(self):
        source = {
            "path": "requirements.txt", "start": 1,
            "line_numbers": [1, 2, 3, 4, 5],
            "text": "os\npathlib\nDjango>=5\ndjango==4.2\nFastAPI>=0.100; python_version >= '3.10'",
        }
        self.assertEqual(
            main.third_party_imports(source),
            [(3, "Django"), (5, "FastAPI")],
        )

    def test_library_extraction_reads_pyproject_dependencies(self):
        source = {
            "path": "pyproject.toml", "start": 1,
            "text": '[project]\ndependencies = [\n  "fastapi>=0.1",\n  "uvicorn[standard]",\n]\n',
        }
        self.assertEqual(
            main.third_party_imports(source),
            [(3, "fastapi"), (4, "uvicorn")],
        )

    def test_python_import_extraction_excludes_stdlib_and_deduplicates_case(self):
        source = {
            "path": "app.py", "start": 8,
            "line_numbers": [8, 9, 10, 11, 12, 13, 14],
            "text": "import os\nimport sys\nimport pathlib\nimport django\nimport Django\nfrom rest_framework import viewsets\nfrom . import local_module",
        }
        self.assertEqual(
            main.third_party_imports(source),
            [(4, "Django"), (6, "Django REST Framework")],
        )

    def test_chunk_line_map_ignores_trimmed_blank_lines(self):
        chunks = main.split_chunks(Path("sample.py"), "\n\nvalue = 1\n\n")
        self.assertEqual(chunks[0]["line_numbers"], [3])
        self.assertEqual((chunks[0]["start"], chunks[0]["end"]), (3, 3))

    def test_python_chunks_follow_function_boundaries(self):
        text = "import os\n\n\ndef login(user):\n    return user.is_valid()\n\n\ndef logout(user):\n    return user.close()\n"
        chunks = main.split_chunks(Path("auth.py"), text)
        functions = {chunk["symbol"]: chunk for chunk in chunks if chunk.get("symbol") in {"login", "logout"}}
        self.assertEqual(set(functions), {"login", "logout"})
        self.assertEqual(functions["login"]["start"], 4)
        self.assertIn("return user.is_valid()", functions["login"]["text"])
        self.assertNotIn("logout", functions["login"]["text"])

    def test_markdown_chunks_follow_headings(self):
        chunks = main.split_chunks(Path("README.md"), "# Overview\nIntro\n\n## Install\nRun setup\n\n## Usage\nAsk a question")
        sections = {chunk["symbol"] for chunk in chunks}
        self.assertTrue({"Overview", "Install", "Usage"}.issubset(sections))

    def test_hybrid_search_can_retrieve_a_semantic_match_without_keyword_overlap(self):
        main.chunks = [
            {"id": "keyword", "path": "findings.py", "start": 1, "end": 1,
             "line_numbers": [1], "text": "def deduplicate_findings(items): return items"},
            {"id": "semantic", "path": "storage.py", "start": 4, "end": 4,
             "line_numbers": [4], "text": "Store records once using their stable identity."},
        ]
        main.chunk_vectors = {"keyword": [1.0, 0.0], "semantic": [0.0, 1.0]}
        with patch("main.get_embedder", return_value=StubEmbedder()):
            ranked = main.rank("How are duplicate entries consolidated?")
        self.assertEqual(ranked[0]["id"], "semantic")


class ApplicationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.saved_state = (main.chunks, main.chunk_vectors, main.indexed_root, main.repo_files,
                            main.active_repo_id, main.embedding_error, main.registry, main.REGISTRY_PATH,
                            main.index_store)
        main.index_store = main.IndexStore(Path(self.temp.name) / "index.sqlite3")
        main.chunks, main.chunk_vectors, main.indexed_root, main.repo_files = [], {}, None, []
        main.active_repo_id, main.embedding_error = None, None

    def tearDown(self):
        (main.chunks, main.chunk_vectors, main.indexed_root, main.repo_files,
         main.active_repo_id, main.embedding_error, main.registry, main.REGISTRY_PATH,
         main.index_store) = self.saved_state
        self.temp.cleanup()

    def test_index_skips_generated_hidden_and_oversized_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text("print('ok')\n", encoding="utf-8")
            (root / "Node_Modules").mkdir()
            (root / "Node_Modules" / "bundle.js").write_text("ignored", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "config.py").write_text("ignored", encoding="utf-8")
            (root / "large.py").write_text("x" * (main.MAX_FILE_BYTES + 1), encoding="utf-8")
            info = main.index_repository(root)
            self.assertEqual(info["files"], 1)
            self.assertEqual([entry["path"] for entry in main.repo_files], ["app.py"])
            self.assertFalse(info["semantic_search"])
            self.assertIn("Install fastembed", info["embedding_error"])

    def test_sqlite_index_persists_vectors_and_reuses_unchanged_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            source = root / "logic.py"
            source.write_text("def deduplicate(items):\n    return items\n", encoding="utf-8")
            stub = StubEmbedder()
            store_path = Path(directory) / "index.sqlite3"
            main.index_store = main.IndexStore(store_path)
            with patch("main.get_embedder", return_value=stub):
                first = main.index_repository(root)
                first_calls = stub.passage_calls
                main.index_store = main.IndexStore(store_path)
                second = main.index_repository(root)
                self.assertEqual(second["unchanged"], 1)
                self.assertEqual(second["added_or_changed"], 0)
                self.assertEqual(stub.passage_calls, first_calls)
                self.assertTrue(second["semantic_search"])
                self.assertEqual(len(main.chunk_vectors), first["chunks"])
                source.write_text("def deduplicate(items):\n    return set(items)\n", encoding="utf-8")
                third = main.index_repository(root)
                self.assertEqual(third["added_or_changed"], 1)
                self.assertEqual(third["unchanged"], 0)

    def test_incomplete_scan_keeps_previously_indexed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            source = root / "app.py"
            source.write_text("def login(): return True\n", encoding="utf-8")
            main.index_repository(root)
            def failed_walk(path, onerror):
                onerror(PermissionError("directory access denied"))
                yield str(path), [], []
            with patch("devrag.indexer.os.walk", side_effect=failed_walk):
                info = main.index_repository(root)
            self.assertTrue(info["scan_incomplete"])
            self.assertEqual(info["removed"], 0)
            self.assertEqual([entry["path"] for entry in main.repo_files], ["app.py"])

    def test_index_skips_file_symlinks_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root, outside_root = Path(directory), Path(outside)
            secret = outside_root / "secret.py"
            secret.write_text("API_KEY = 'should not be indexed'", encoding="utf-8")
            (root / "linked.py").write_text(secret.read_text(encoding="utf-8"), encoding="utf-8")
            (root / "inside.py").write_text("value = 1", encoding="utf-8")
            original_is_symlink = Path.is_symlink
            with patch.object(Path, "is_symlink", autospec=True,
                              side_effect=lambda path: path.name == "linked.py" or original_is_symlink(path)):
                main.index_repository(root)
            self.assertEqual([entry["path"] for entry in main.repo_files], ["inside.py"])

    def test_repository_add_save_select_and_file_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "app.py").write_text("def login():\n    return True\n", encoding="utf-8")
            main.REGISTRY_PATH = Path(directory) / "registry.json"
            main.registry = {"repos": [], "current_id": None}
            added = main.add_repository(main.IndexRequest(path=str(root)))
            self.assertEqual(added["info"]["files"], 1)
            self.assertTrue(main.REGISTRY_PATH.exists())
            repeated = main.add_repository(main.IndexRequest(path=str(root)))
            self.assertEqual(len(repeated["repos"]), 1)
            selected = main.select_repository(main.RepoSelectRequest(id=added["current_id"]))
            self.assertEqual(selected["info"]["root"], str(root.resolve()))
            content = main.read_source_file("app.py")
            self.assertIn("def login", content["content"])
            with self.assertRaises(HTTPException) as raised:
                main.read_source_file("../secret.py")
            self.assertEqual(raised.exception.status_code, 400)
            with self.assertRaises(HTTPException) as raised:
                main.read_source_file("missing.py")
            self.assertEqual(raised.exception.status_code, 404)
            with self.assertRaises(HTTPException) as raised:
                main.select_repository(main.RepoSelectRequest(id="not-saved"))
            self.assertEqual(raised.exception.status_code, 404)

    def test_file_list_requires_an_active_repository(self):
        main.indexed_root = None
        main.repo_files = []
        with self.assertRaises(HTTPException) as raised:
            main.list_source_files()
        self.assertEqual(raised.exception.status_code, 400)

    def test_missing_repo_and_empty_question_return_client_errors(self):
        main.chunks = []
        with self.assertRaises(HTTPException) as raised:
            main.ask(main.AskRequest(question="hello"))
        self.assertEqual(raised.exception.status_code, 400)
        main.chunks = [{"path": "app.py", "start": 1, "end": 1,
                        "line_numbers": [1], "text": "value = 1"}]
        with self.assertRaises(HTTPException) as raised:
            main.ask(main.AskRequest(question="   "))
        self.assertEqual(raised.exception.status_code, 400)

    def test_offline_ask_returns_answer_and_matching_line(self):
        main.chunks = [{"path": "auth.py", "start": 10, "end": 11,
                        "line_numbers": [10, 11], "text": "def login():\n    authenticate(user)"}]
        with patch.dict(os.environ, {}, clear=True):
            response = main.ask(main.AskRequest(question="where does login authenticate"))
        self.assertIn("local explanation", response["answer"])
        self.assertEqual(response["sources"][0]["match_lines"], [10, 11])

    def test_corrupt_repository_registry_recovers_as_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            main.REGISTRY_PATH = Path(directory) / "repos.json"
            main.REGISTRY_PATH.write_text("{bad json", encoding="utf-8")
            self.assertEqual(main.load_registry(), {"repos": [], "current_id": None})


if __name__ == "__main__":
    unittest.main()
