# PyHealth `.llms/` — agent guidance for working in this repo

This folder is a **tool-agnostic** home for the context an AI coding agent needs to be
useful in PyHealth: durable **rules** about how the codebase works, and step-by-step
**skills** for the things people actually do here. It is meant to ship in the repo so any
contributor's agent — Claude Code, Cursor, Copilot, etc. — starts oriented instead of
guessing.

## Who we build for

PyHealth's users are mostly **researchers**, and increasingly people for whom *coding is
not the main skill*: student researchers (undergrad → PhD), engineers giving a few evening
hours to healthcare research, and **clinicians who know the medicine but write little
code**. The accessibility goal is explicit — so this folder is written **clinician-first**.
Everything here assumes the human may not be able to debug a broken environment or tell a
real result from a silently broken one, and that the agent must close that gap.

Read `principles.md` first — it is the lens every skill inherits.

## Layout

```
.llms/
  README.md          ← you are here (index + how to wire it up)
  principles.md      ← clinician-first operating principles (apply to ALL skills)
  rules/             ← always-true facts about the repo (load these as context)
    00-repo-map.md       the Dataset→Task→Model→Trainer→Metrics pipeline + where things live
    01-env-and-tests.md  pixi env, how to run the right tests
    02-contributing.md   branch/PR flow, style, docs+tests requirements
    03-data-safety.md    never commit PHI/data/credentials; demo-data defaults
  skills/            ← procedures for specific jobs (one folder each, SKILL.md inside)
    train-eval/          load a dataset → task → model → train → evaluate (the core loop)
    scaffold-dataset/    add a new dataset (.py + YAML config + test)
    scaffold-task/       add a new prediction task (BaseTask subclass + test)
    scaffold-model/      add a new model (BaseModel subclass + test)
    dev-env-and-test/    set up pixi and run the correct test subset / read CI failures
    slurm-delta/         OPTIONAL, opt-in: run general-purpose jobs on Delta/SLURM
```

## Wiring it into a tool

The content here is plain Markdown so any tool can read it. The canonical copy stays in
`.llms/`; each tool gets a thin adapter so the guidance is never forked.

### Claude Code (adapter shipped)

- **Rules** load automatically via the repo-root `CLAUDE.md`, which `@`-imports
  `.llms/principles.md` and `.llms/rules/*`. Nothing to set up. (Edit the rules in `.llms/`,
  not in `CLAUDE.md`.)
- **Skills** must live under `.claude/skills/` to be discovered, and `.claude/` is
  **gitignored** (local to each clone) — so they are not shipped pre-activated. Activate them
  once per clone:

  ```bash
  bash .llms/install-claude-skills.sh
  ```

  This symlinks `.llms/skills/*` into your local `.claude/skills/`. Restart Claude Code to
  pick them up. (Windows: enable symlinks or copy the folders instead — the script prints how.)

### Cursor / other tools

Reference `.llms/rules/*` and `.llms/principles.md` from the tool's own rules file.

### Why skills aren't committed under `.claude/`

`.claude/*` is gitignored by design, which keeps personal/local config (and any per-user skill
activation) out of the public repo. The shippable, tool-agnostic source of truth is `.llms/`;
`.claude/skills/` is just a per-clone activation of it.

## House rules for this folder

- **Generic only.** This ships to every PyHealth user. No personal paths, accounts, dataset
  locations, or cluster specifics. Anything personal belongs in your own `~/.claude`, not
  here.
- **Local-first.** Default to demo/dev data and CPU/single-GPU. HPC is one optional skill,
  never assumed.
- **Keep it true.** If the codebase changes, fix the rule. A confidently wrong rule is worse
  than no rule, especially for a user who can't catch the error.
