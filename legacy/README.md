# legacy

`v1_47_paper/` is the frozen code and the original result files behind the LRE submission.
Do not edit anything here.

- `agente_egipsio_langgraph_v1_47.py`: md5 4cad3a4fd7d06d2bfb54d168cc6fb2f8, copied from
  `D:\estancia_espanha_agentes` on 2026-11-03.
- `results/`: nested CV and consensus CV outputs written on 2026-05-15 by that script.
- `repro_rule_path.py`: re-runs the rule-only path without API keys and compares with `results/`.

The full development history (v1 to v1.47, CrewAI notebooks, desktop app, RAG experiments) remains in
the archive folder `D:\estancia_espanha_agentes` and is not copied here.

`v1_28_initial_github/` holds the script that was published in this repository before 2026-11-03
(tag `v1.28-initial`). It predates LOO, Optuna and nested CV and does not reproduce the manuscript.
