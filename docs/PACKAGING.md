# Packaging

## Option A – plain Python (recommended)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python data/create_sample_db.py
python scripts/download_model.py                        # once, online
streamlit run app.py
```

## Option B – single-folder executable with PyInstaller

An executable is feasible, but torch and the weights make it large (roughly 1 GB for the app plus the model size), so
the weights are kept **outside** the bundle and found through `NL2SQL_MODEL_CACHE`.

```bash
pip install pyinstaller
pyinstaller nl2sql_chatbot.spec
# result: dist/nl2sql_chatbot/nl2sql_chatbot(.exe)
```

Run it:

```bash
# Linux / macOS
NL2SQL_MODEL_CACHE=/path/to/models ./dist/nl2sql_chatbot/nl2sql_chatbot
# Windows (PowerShell)
$env:NL2SQL_MODEL_CACHE="C:\path\to\models"; .\dist\nl2sql_chatbot\nl2sql_chatbot.exe
```

`run_app.py` is the entry point: it starts Streamlit programmatically on the bundled `app.py`, `data/sample.db` and
`data/schema_metadata.json`. Build on the target OS, because PyInstaller does not cross-compile.
Without weights the executable still works and answers through the template fallback.

The spec was written for this project but was **not built in the development environment**. Treat it as a starting
point: some torch/transformers versions need extra `hiddenimports`.

## Option C – Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 && python data/create_sample_db.py
# Mount the downloaded weights: -v $PWD/models:/app/models
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
```
