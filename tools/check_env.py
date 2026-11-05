"""Check that every module the project needs can be imported in this Python environment.

    python tools/check_env.py

Prints one line per module and, for the ones that fail, a conda command to (re)install them. On
machines where Windows Application Control blocks DLLs from pip wheels, conda-forge builds are the
usual way out: uninstall the pip copy first, then install from conda-forge.
"""
from __future__ import annotations

import importlib
import sys

# module -> package name (pip / conda-forge)
MODULES = {
    "numpy": "numpy", "pandas": "pandas", "openpyxl": "openpyxl", "lxml.etree": "lxml",
    "docx": "python-docx", "yaml": "pyyaml", "optuna": "optuna", "matplotlib.pyplot": "matplotlib",
    "pydantic_core": "pydantic-core", "orjson": "orjson", "ormsgpack": "ormsgpack", "xxhash": "xxhash",
    "jiter": "jiter", "tiktoken": "tiktoken", "openai": "openai", "langchain_core": "langchain-core",
    "langchain_openai": "langchain-openai", "langgraph.graph": "langgraph", "dotenv": "python-dotenv",
}


def main() -> int:
    failed = []
    print(f"python: {sys.executable}")
    for mod, pkg in MODULES.items():
        try:
            importlib.import_module(mod)
            print(f"  ok      {mod}")
        except Exception as exc:
            msg = str(exc).splitlines()[0][:110]
            print(f"  FAILED  {mod}: {type(exc).__name__}: {msg}")
            failed.append(pkg)
    if failed:
        pkgs = " ".join(dict.fromkeys(failed))
        print("\nSuggested fix (run in PowerShell):")
        print(f'  & "{sys.executable}" -m pip uninstall -y {pkgs}')
        print(f'  & "$HOME\\miniconda3\\Scripts\\conda.exe" install -n <env> -c conda-forge {pkgs} -y')
        return 1
    print("\nall modules import correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
