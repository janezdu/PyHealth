# Rule: data safety

PyHealth works with clinical data, which is sensitive and usually **credentialed** (MIMIC,
eICU, EHRShot, etc. require a data-use agreement). The library ships *code*, never *data*.
This rule is non-negotiable and matters most for users who won't realize they've crossed a
line.

## Never commit

- Raw data files (CSV/parquet/records) — they don't belong in the repo, credentialed or not.
- Credentials, tokens, or data-access keys.
- **Hardcoded personal paths** to someone's data (`/home/me/mimic/...`, cluster scratch
  paths). Paths come from the user via `root=`/args/config, never baked into committed code.
- Model checkpoints or run outputs (they live under gitignored `_outputs/` etc.).

Before committing, scan the diff for paths, tokens, and stray data files. If you see one,
stop and flag it.

## Never expose

- Don't print patient-level rows into shared logs, notebooks committed to the repo, or any
  output that leaves the user's machine.
- Don't suggest uploading raw clinical records to an external API/service. If a workflow
  needs an LLM over clinical text, surface the data-governance question rather than silently
  sending records out.

## Defaults that keep users safe

- Start on **demo/dev data** (`dev=True`, demo subsets). Only touch the full dataset when the
  user explicitly asks and has the access.
- Keep data location **configurable** (`root=`), with the demo path as the gentle default.
- When a user points at a real dataset, assume it's governed: keep its contents on their
  machine and out of the repo.

## If a user is new to clinical data

Say the quiet part out loud: "MIMIC/eICU need a credentialed login and a data-use agreement;
here's the demo subset you can use today, and here's how to point PyHealth at the full data
once you're approved." That's part of making research accessible, not a tangent.
