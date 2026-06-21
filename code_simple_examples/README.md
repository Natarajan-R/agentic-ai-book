# Agentic AI — Book Code Examples

All examples run **locally at zero cost** using Ollama + Qwen2.5.

> Full step-by-step setup (Ollama, model, virtual environment, both dependency
> sets, verification, and troubleshooting) is in **`00_getting_started.md`** at
> the top level of the book. The quick version is below.

## Quick setup

```bash
# 1. Install Ollama  (Windows: download from https://ollama.com/download)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the model (one-time, ~4.7 GB) — all 14 chapters use this one model
ollama pull qwen2.5:7b

# 3. Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1

# 4. Install the dependencies for these simple examples
pip install -r code_simple_examples/requirements.txt

# 5. Verify
ollama run qwen2.5:7b "Hello, are you working?"
python code_simple_examples/ch01_script_vs_agent.py
```

All 14 chapters run on `qwen2.5:7b`. If you have the hardware, pulling
`qwen2.5:14b` and changing `MODEL` at the top of a file gives somewhat more
reliable tool-calling on the Chapter 8–14 examples, but it is not required.

The production-grade package in `code_production/` has its own (larger)
dependency set — install it separately with
`pip install -r code_production/requirements.txt`.

## Code structure

| File | Chapter | Approach |
|---|---|---|
| ch01_script_vs_agent.py | 1 | Pure Python |
| ch02_eight_layers.py | 2 | Pure Python |
| ch03_memory.py | 3 | Pure Python + ChromaDB |
| ch04_react_loop.py | 4 | Pure Python |
| ch05_planning.py | 5 | Pure Python |
| ch06_tools.py | 6 | Pure Python |
| ch07_guardrails.py | 7 | Pure Python |
| ch08_reference_agent.py | 8 | **LangGraph** |
| ch09_multi_agent.py | 9 | **LangGraph** |
| ch10_knowledge_graph.py | 10 | **LangGraph + NetworkX** |
| ch11_advanced_planning.py | 11 | **LangGraph** |
| ch12_observability.py | 12 | **LangGraph + metrics** |
| ch13_security.py | 13 | **LangGraph + red-team tests** |
| ch14_self_refining.py | 14 | **LangGraph + history store** |

## Model selection

All 14 examples run on `qwen2.5:7b` and are engineered to behave reliably on
it. Each file defines `MODEL` (or `FRONTIER_MODEL`/`CHEAP_MODEL` in Chapter 12)
near the top — change it there if you want to try a larger model such as
`qwen2.5:14b`.
