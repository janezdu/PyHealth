# Rule: contributing (branch, PR, style)

From `CONTRIBUTING.md`. Get these right so a newcomer's first PR isn't bounced on mechanics.

## Branch flow (GitHub Flow with a `develop` integration branch)

- **Features** branch off **`develop`** and PR back into **`develop`** (needs 1 core
  approval).
- **Hotfixes** branch off **`master`** and PR into **`master`** (needs 2 core approvals).
- `master` is always production-ready; `develop` is the latest integrated work.
- Do **not** branch features off `master` or PR features straight into `master`.

## Every PR must include

1. The code change.
2. **Documentation** for it (docstrings + relevant `docs/` update).
3. **Unit tests** for it.

A change without tests + docs is not done. For non-trivial changes, open an issue to discuss
*before* implementing; trivial fixes can go straight to a PR.

## Style

- **PEP 8**, line length **88** characters.
- **Google-style docstrings** (Args / Returns / Raises / Examples). Public classes and
  functions get a docstring; for a clinical audience, a one-line plain-language summary of
  *what the thing is for* is worth more than restating the signature.

## Agent etiquette

- Create a branch; never commit straight to `master`/`develop`.
- Don't push or open a PR unless the user asks.
- When you finish a change, state plainly what you ran and whether tests passed — don't claim
  "done" on an unrun change.
