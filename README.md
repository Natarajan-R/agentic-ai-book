# Agentic AI — A Hands-On Guide · Companion Code

Runnable code for the book **_Agentic AI — A Hands-On Guide_** by Natarajan
Ramasamy. Every example runs **locally at zero cost** on [Ollama](https://ollama.com)
with the `qwen2.5:7b` model — no API keys, no cloud bills, no GPU required.

📖 **Get it on Amazon Kindle:** [US](https://www.amazon.com/dp/B0H6R7SZZB) · [India](https://www.amazon.in/dp/B0H6R7SZZB)

---

## What's inside

| Folder | What it is |
|---|---|
| [`code_simple_examples/`](code_simple_examples/) | The concept examples for **Chapters 1–14** — short, focused scripts that each illustrate one idea. |
| [`code_production/`](code_production/) | A **production-grade** agent package (used from Chapter 8 onward): a complete LangGraph reference agent, multi-agent system, knowledge-graph agent, tree-of-thought planner, observability, security, self-refining agent — plus a `pytest` test suite. |

Chapter → file mapping is in [`code_simple_examples/README.md`](code_simple_examples/README.md).

---

## Setup

### 1. Install Ollama and pull the model

```bash
# macOS / Linux  (Windows: download from https://ollama.com/download)
curl -fsSL https://ollama.com/install.sh | sh

ollama pull qwen2.5:7b        # one-time, ~4.7 GB — every example uses this model
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
# for the simple examples (Chapters 1–14)
pip install -r code_simple_examples/requirements.txt

# for the production package (Chapter 8 onward)
pip install -r code_production/requirements.txt
```

### 4. Run

```bash
# a simple example
python code_simple_examples/ch01_script_vs_agent.py

# the production reference agent + test suite (run as modules from the package folder)
cd code_production
python -m agents.reference
pytest tests/test_suite.py -q
cd ..
```

All 14 chapters run on `qwen2.5:7b`. If you have the hardware and want slightly
more reliable tool-calling on Chapters 8–14, you can `ollama pull qwen2.5:14b`
and change the `MODEL` value at the top of those files — but it is not required.

---

## A note on small local models

A 7B model running locally is not perfectly deterministic — your output will
sometimes differ from the book's, and occasionally the model makes a reasoning
slip. The code is written defensively to behave reliably anyway; where the model
is the weak link, the book explains the engineering that makes agents dependable.

---

## More books by the author

Each one is a hands-on build with its code in the open.

| Book | Amazon | Code |
|---|---|---|
| **Enterprise AI Workflow Automation** | [US](https://www.amazon.com/dp/B0HCZC7VCC) · [IN](https://www.amazon.in/dp/B0HCZC7VCC) | [auto-sre-graph](https://github.com/Natarajan-R/auto-sre-graph) |
| **Building a Local AI Coding Agent** | [US](https://www.amazon.com/dp/B0H8B6QXXX) · [IN](https://www.amazon.in/dp/B0H8B6QXXX) | [local-ai-coding-agent](https://github.com/Natarajan-R/local-ai-coding-agent) |
| **GraphRAG: Building an Intelligent Research Assistant** | [US](https://www.amazon.com/dp/B0H3QXVSY4) · [IN](https://www.amazon.in/dp/B0H3QXVSY4) | [graphrag-book-code](https://github.com/Natarajan-R/graphrag-book-code) |

All titles → [Amazon author page](https://www.amazon.com/stores/author/B0H3T2MG83)

---

## License

Code released under the [MIT License](LICENSE). The book text is © 2026 Natarajan
Ramasamy and is **not** included in this repository.
