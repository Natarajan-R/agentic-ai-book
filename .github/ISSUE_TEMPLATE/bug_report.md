---
name: Bug report
about: An example doesn't run, crashes, or behaves incorrectly
title: "[Bug] "
labels: bug
---

**Which example?**
Chapter and file — e.g. `code_simple_examples/ch06_tools.py` or
`code_production/agents/reference.py`.

**What did you run?**
The exact command, e.g. `python code_simple_examples/ch04_react_loop.py`.

**What happened?**
Paste the full output / traceback (use a ``` code block ```). Describe what you
expected instead.

**Environment**
- OS: (macOS / Linux / Windows + version)
- Python version: (`python --version`)
- Ollama version: (`ollama --version`)
- Model used: (e.g. `qwen2.5:7b`)
- Installed which requirements? (simple examples / production / both)

**Checklist**
- [ ] Ollama is running and the model is pulled (`ollama run qwen2.5:7b "hi"`)
- [ ] My virtual environment is activated
- [ ] I installed the matching `requirements.txt`
- [ ] This is a real failure (crash / traceback / never completes), not just the
      7B model wording its answer differently
