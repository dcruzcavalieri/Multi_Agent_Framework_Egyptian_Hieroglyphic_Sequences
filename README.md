# Egyptian Hieroglyphic Analysis with LangGraph

This project implements a multi-agent system for deterministic and contextual analysis of Egyptian hieroglyphic sequences using LangGraph and LLMs (OpenAI/Anthropic).

## Prerequisites

Ensure you have Python installed (3.10+ recommended).

### Dependencies

Install the required libraries:

```bash
pip install pandas numpy matplotlib openpyxl python-dotenv langgraph langchain-openai langchain-anthropic langchain-core
```

## Configuration

1. Clone this repository or download the `agente_egipsio_langgraph_v1_28.py` file.
2. Create a file named `.env` in the same directory as the script.
3. Add your OpenAI (or Anthropic) API key to the `.env` file:

   ```env
   OPENAI_API_KEY=sk-...
   ```

4. **Note on Data Path**: By default, the script points to a specific Excel file path (`E:/estancia_espanha_agentes/content/extractrion_for_agents.xlsx`). Please update the `DATA_PATH` variable in the script to match your local file location.

## How to Run

Execute the main script:

```bash
python agente_egipsio_langgraph_v1_28.py
```

The script will perform rule-based analysis by default. If an API key is detected, it will automatically use LangGraph for a more sophisticated contextual analysis.
