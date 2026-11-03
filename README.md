# Multi-Agent Framework for Determinative Detection in Egyptian Hieroglyphic Sequences

Code and benchmark for the manuscript:

> D. Cruz-Cavalieri, C. Gracia-Zamacona, J. Grotenhuis, S. E. Palazuelos-Cagigas, P. S. dos Santos-Junior.
> *Determinative Detection in Egyptian Hieroglyphic Sequences: A Sequence-Labelling Benchmark and
> Multi-Agent LLM Baseline.* Submitted to *Language Resources and Evaluation*.

Instituto Federal do Espírito Santo (IFES, Brazil) and Universidad de Alcalá (UAH, Spain).
Dataset archive: [10.5281/zenodo.21776840](https://doi.org/10.5281/zenodo.21776840).

## What is here

| Path | Content |
|---|---|
| `legacy/v1_47_paper/` | **Exact code behind the manuscript** (`agente_egipsio_langgraph_v1_47.py`), the original result files, and an offline reproduction script for the rule-only path |
| `data/raw/` | Corpus (533 MdC sequences with transliteration and translation) and the expert annotations of 99 sequences (determinatives marked in bold) |
| `data/gold/gold_v1.jsonl` | Same 99 annotations in machine-readable form, with the token position of every determinative (built by `python -m egydet.gold.build`) |
| `data/splits/v147_seed42.json` | The outer 5-fold and inner 5x2 cross-validation splits used for the manuscript, by sequence id |
| `kb/` | Knowledge base (sign entries, sign categories, semantic fields) as YAML |
| `docs/diagrams/` | Architecture and workflow diagrams |
| `legacy/v1_28_initial_github/` | The script previously published in this repository (superseded) |
| `configs/`, `src/egydet/`, `tests/` | Modular v2 implementation (in progress; the rule-only path is complete and reproduces v1.47) |

The git tag **`v1.47-paper`** marks the state of the code that produced the reported results.

## Reported results (nested 5-fold CV, inner 5x2 CV, 99 sequences)

| Metric | Macro | Micro |
|---|---|---|
| F1 | 81.94 ± 2.19 | 87.50 ± 2.50 |
| Precision | 87.20 ± 2.78 | 88.19 ± 0.51 |
| Recall | 86.72 ± 5.07 | 87.01 ± 5.19 |

LLM: OpenAI `gpt-5.4-nano`, `temperature=0`, `top_p=1`, `seed=42`. Source: `legacy/v1_47_paper/results/`.

## Running the paper version

```bash
pip install pandas numpy matplotlib seaborn openpyxl python-docx python-dotenv optuna \
            langgraph langchain-core langchain-openai langchain-anthropic
```

The script is a single Spyder-style file with hard-coded paths (`DATA_PATH`, `OUTPUT_DIR`, `DOCX_PATHS`,
around line 2946). Point them to `data/raw/`, put `OPENAI_API_KEY` in a `.env` file next to the script
(see `.env.example`), and run it. Without an API key it falls back to the rule-only path.

Offline check of the rule-only path against the archived results (no API key needed):

```bash
python legacy/v1_47_paper/repro_rule_path.py baseline   # seconds
python legacy/v1_47_paper/repro_rule_path.py fold1      # ~3 min, runs Optuna for outer fold 1
```

## Status

The v1.47 code is frozen at tag `v1.47-paper`. Further development happens in the modular v2 package.

```bash
pip install -e ".[dev]"
python -m egydet.gold.build    # rebuild data/gold/gold_v1.jsonl from data/raw
python -m egydet.eval.cv       # nested CV of the rule-only scorer (about 20 min), results in runs/
pytest -q
```
