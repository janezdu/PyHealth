#!/usr/bin/env bash
#
# Activate the shipped .llms/ skills as native Claude Code skills.
#
# Claude Code only discovers skills under .claude/skills/, and .claude/ is gitignored
# (local to each clone), so this is a one-time per-clone setup step — it does not modify
# anything that ships in the repo. Re-run it after new skills are added under .llms/skills/.
#
# Usage (from anywhere):  bash .llms/install-claude-skills.sh
#
set -euo pipefail

# Resolve repo root as the parent of this script's directory.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

src_dir=".llms/skills"
dst_dir=".claude/skills"

if [ ! -d "${src_dir}" ]; then
  echo "error: ${src_dir} not found (run this from a PyHealth checkout)." >&2
  exit 1
fi

mkdir -p "${dst_dir}"

linked=0
for d in "${src_dir}"/*/; do
  [ -d "${d}" ] || continue
  name="$(basename "${d}")"
  target="${dst_dir}/${name}"
  # Replace any existing link/dir so re-runs are idempotent.
  rm -rf "${target}"
  # Relative symlink so the link is valid regardless of where the repo is cloned.
  ln -s "../../${src_dir}/${name}" "${target}"
  echo "linked ${target} -> ../../${src_dir}/${name}"
  linked=$((linked + 1))
done

echo "Activated ${linked} skill(s). Restart Claude Code to pick them up."
echo "Note: on Windows, symlinks need Developer Mode or 'git config core.symlinks true';"
echo "      otherwise copy the .llms/skills/<name> folders into .claude/skills/ instead."
