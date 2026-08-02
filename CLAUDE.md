# PyHealth — guidance for AI coding agents

This project ships agent guidance in a tool-agnostic [`.llms/`](.llms/README.md) folder.
The principles and rules below are **imported** from there so they load automatically in
Claude Code. The canonical copies live under `.llms/` — **edit them there**, not here.

New to the codebase? `.llms/rules/00-repo-map.md` (imported below) is the fastest orientation.

## Operating principles (read first)

@.llms/principles.md

## Rules

@.llms/rules/00-repo-map.md
@.llms/rules/01-env-and-tests.md
@.llms/rules/02-contributing.md
@.llms/rules/03-data-safety.md

## Skills

Step-by-step procedures for common jobs (train/evaluate a model, add a dataset/task/model,
set up the dev env, run on SLURM) live in `.llms/skills/<name>/SKILL.md`.

Claude Code only discovers skills under `.claude/skills/`, which is gitignored (local to each
clone). To activate the shipped skills as native Claude Code skills, run once from the repo
root:

```bash
bash .llms/install-claude-skills.sh
```

This symlinks `.llms/skills/*` into your local `.claude/skills/`. Restart Claude Code to pick
them up. See `.llms/README.md` for details and other tools.
