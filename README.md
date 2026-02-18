# Análise de Jeroglíficos Egípcios com LangGraph

Este projeto implementa um sistema multi-agente para análise de determinativos em jeroglíficos egípcios, utilizando LangGraph e LLMs (OpenAI/Anthropic).

## Pré-requisitos

Certifique-se de ter o Python instalado (recomendado 3.10+).

### Dependências

Instale as bibliotecas necessárias:

```bash
pip install pandas numpy matplotlib openpyxl python-dotenv langgraph langchain-openai langchain-anthropic langchain-core
```

## Configuração

1. Clone o repositório ou baixe o arquivo `agente_egipsio_langgraph_v1_28.py`.
2. Crie um arquivo chamado `.env` na mesma pasta do código.
3. Adicione sua chave de API da OpenAI (ou Anthropic) no arquivo `.env`:

   ```env
   OPENAI_API_KEY=sk-...
   ```

4. Certifique-se de que o arquivo de dados `extractrion_for_agents.xlsx` esteja no caminho esperado pelo código ou ajuste a variável `DATA_PATH` no script.

## Como Executar

Execute o script principal:

```bash
python agente_egipsio_langgraph_v1_28.py
```

O script realizará a análise baseada em regras por padrão e, se uma chave de API for detectada, utilizará o LangGraph para uma análise contextual mais sofisticada.
