# Offline model setup

The chatbot runs `Qwen2.5-Coder-1.5B-Instruct` locally with Hugging Face Transformers + PyTorch. After a one-time
download it needs **no network**: weights load with `local_files_only=True` and `HF_HUB_OFFLINE=1`. Ollama or any other
runtime service is not used.

## 1. Install PyTorch

| Machine | Command |
|---|---|
| CPU only (works everywhere) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| NVIDIA GPU (e.g. RTX 40-series) | `pip install torch --index-url https://download.pytorch.org/whl/cu128` (or the command from pytorch.org) |

Then `pip install -r requirements.txt`. On Windows, a plain `pip install torch` gives the CPU build. To check:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 2. Download the weights (once, online)

```bash
python scripts/download_model.py
```

This downloads about 3 GB into `models/qwen2.5-coder-1.5b/`, which is where `nl2sql/config.py` looks by default. It
then loads the model with networking disabled and generates one query, proving the folder is complete. If the
download is interrupted, run the same command again and it resumes.

Equivalent without the script:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-Coder-1.5B-Instruct', local_dir='models/qwen2.5-coder-1.5b')"
```

## 3. Check it runs offline

Turn the network off, then:
```bash
python -m nl2sql -v --csv data/task2_bank_transactions_sample.csv --schema data/task2_bank_schema.txt --table transactions "How many debit transactions are there?"
```
You should see `Source: slm` and the answer 28. If it says `template_fallback`, the `-v` output shows why the model
didn't load (usually a wrong folder or an interrupted download).

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `NL2SQL_MODEL` | `models/qwen2.5-coder-1.5b` | model folder (or a cached Hugging Face id) |
| `NL2SQL_DEVICE` | `auto` | `auto` uses the GPU when available, otherwise the CPU; or force `cpu` / `cuda` |
| `NL2SQL_MAX_NEW_TOKENS` | `128` | maximum length of the generated SQL |
| `NL2SQL_TEMPERATURE` | `0.0` | 0 = greedy decoding (deterministic) |
| `NL2SQL_MAX_RETRIES` | `2` | repair attempts after the first generation (hard cap 5) |

## Resource use

| Device | Precision | Memory | Per question |
|---|---|---|---|
| NVIDIA GPU | fp16 | ~3.5 GB VRAM | ~1 s (RTX 4050 laptop) |
| CPU | fp32 | ~6–7 GB RAM | a few seconds (depends on the CPU) |

With 8 GB of RAM or less, use the 0.5B model instead:
`python scripts/download_model.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --out models/qwen2.5-coder-0.5b`
then set `NL2SQL_MODEL=models/qwen2.5-coder-0.5b`. It is weaker; the repair loop and fallback cover part of the gap.

## When the weights are missing

The backend reports that it is unavailable, the orchestrator skips the pointless retries, and the rule-based fallback
answers. The UI shows a warning in the sidebar. The app never crashes because of a missing model.