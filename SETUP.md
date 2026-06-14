# Setup notes

## GitHub

This repo is already on GitHub (created and pushed with the `gh` CLI authenticated
as `adamselzer`):

```
https://github.com/adamselzer/policy-manual-rag
```

To push further changes:

```bash
cd ~/code/policy-manual-rag
git add -A && git commit -m "..."
git push
```

## API key for generation and the LLM-judged eval

Retrieval, the ablation, and the tests run with no API key. Answer generation, the
Streamlit app's answer mode, and `eval/run_llm_eval.py` need an Anthropic key:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env   # .env is gitignored
```

The local commit identity is set to `Adam Selzer <hello@aselzer.com>` (the global
git config has malformed values, so it is set per-repo).

## Reproducing the corpus

The parsed sections in `data/manual_sections/` are committed. To re-download and
re-parse from the State of Michigan site:

```bash
python data/ingest.py
```

Raw PDFs land in `data/manual_pdfs/` (gitignored).
