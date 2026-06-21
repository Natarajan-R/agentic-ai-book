# Contributing & Feedback

Thanks for reading *Agentic AI — A Hands-On Guide* and for taking the time to
help improve the companion code.

## Reporting a problem

If an example doesn't run, or you spot a bug or an error in the book's code,
please [open an issue](../../issues/new/choose) and pick the matching template.
The **Bug report** template asks for a few details (chapter/file, the command you
ran, your OS, Python and Ollama versions, and the full error) — including these
makes it much faster to reproduce and fix.

Before opening an issue, a quick checklist that resolves most problems:

1. **Ollama is running** and the model is pulled: `ollama run qwen2.5:7b "hi"`.
2. Your **virtual environment is activated** (`ModuleNotFoundError` almost always
   means it isn't).
3. You installed the right requirements —
   `pip install -r code_simple_examples/requirements.txt` for the simple
   examples, `pip install -r code_production/requirements.txt` for the package.
4. Remember a **7B local model is not deterministic** — wording will vary between
   runs, and the model occasionally makes a reasoning slip. That on its own is
   not a bug; a crash, a traceback, or an example that never completes is.

## Suggesting an improvement or asking a question

Use the **Question / suggestion** template. Ideas that make the examples clearer
or more reliable on small local models are especially welcome.

## Pull requests

Small, focused PRs are welcome — typo fixes, clearer comments, reliability
improvements, or fixes for genuine bugs. Please keep each PR to one logical
change and describe what you ran to verify it. By contributing, you agree your
contribution is licensed under the repository's [MIT License](LICENSE).
