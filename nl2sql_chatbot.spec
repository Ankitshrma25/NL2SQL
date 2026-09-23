# PyInstaller spec - build with:  pyinstaller nl2sql_chatbot.spec
# Model weights are NOT bundled; point NL2SQL_MODEL_CACHE at them at runtime.
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

datas = [
    ("app.py", "."),
    ("nl2sql", "nl2sql"),
    ("data/sample.db", "data"),
    ("data/schema_metadata.json", "data"),
]
datas += collect_data_files("streamlit")
datas += copy_metadata("streamlit")
for pkg in ("transformers", "tokenizers", "huggingface_hub", "safetensors", "torch", "sqlglot", "rapidfuzz",
            "SQLAlchemy", "pandas", "tqdm", "regex", "requests", "packaging", "filelock", "numpy", "pyyaml"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

hiddenimports = (
    collect_submodules("streamlit")
    + collect_submodules("nl2sql")
    + collect_submodules("sqlglot")
    + ["transformers.models.qwen2", "transformers.models.t5", "sqlalchemy.dialects.sqlite"]
)

a = Analysis(["run_app.py"], pathex=["."], datas=datas, hiddenimports=hiddenimports, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="nl2sql_chatbot", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="nl2sql_chatbot")
