# Rule: environment and tests

## This repo uses `pixi`, not bare pip

Development depends on a **pixi**-managed environment (see `pyproject.toml`, `pixi.lock`,
`makefile`). CI installs pixi and runs the test suite through it. Do **not** `pip install`
packages into the dev environment or hand-edit versions to "make it work" — that desyncs the
lockfile and breaks CI.

| Task | Command |
|------|---------|
| Set up / update the environment | `make init` (installs pixi, syncs the lock) |
| Run the core test suite | `make test` |
| Run the NLP test suite | `make testnlp` |
| Run everything (what CI runs) | `make testall` |

If `make`/`pixi` isn't available on the user's machine, that's an environment problem to
solve first — see the `dev-env-and-test` skill, and explain it in plain language rather than
silently switching to pip.

## Running the *right* tests

Tests live in `tests/`. After a change, run the **subset that covers what you touched**
(faster feedback) before running `make testall`:
- touched `pyhealth/nlp/...` or a text model → `make testnlp`
- touched a dataset/task/model/metric → the matching test under `tests/`, then `make test`

CI (`.github/workflows/test.yml`) runs on `pull_request` and on push to `master`, executes
`make testall` on Python 3.13 via pixi, and **ignores** doc/markdown-only changes. A green
local `make testall` is the bar for "ready to push".

## A note for less-technical users

Environment setup is the single most common place a newcomer gets stuck. If the user can't
get pixi working, prefer getting them to **one runnable example on demo data** over a perfect
dev setup — momentum first, full toolchain later.
