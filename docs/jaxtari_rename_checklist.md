# Jaxtari rename checklist

The local codemod is `scripts/rename_to_jaxtari.py`. It only mutates the
working tree. Everything below is **out of scope for the script** and must be
done by hand after the in-repo rename lands.

## What the script does (local only)

Documented here so reviewers can audit a dry-run against this list.

1. **Content rewrite** (text files only; skips binaries / `.git` / venvs / caches)
   - `jaxatari` → `jaxtari` (imports, paths, package refs)
   - `JAXAtari` / `JaxAtari` / `Jaxatari` → `Jaxtari` (prose + identifiers)
   - `JAXATARI_` → `JAXTARI_` (env vars / config keys)
   - Public API identifiers, longest-first (each gets a fallback alias):
     - `GymnasiumJaxAtariWrapper` → `GymnasiumJaxtariWrapper`
     - `JaxAtariInternalModPlugin` → `JaxtariInternalModPlugin`
     - `JaxAtariPostStepModPlugin` → `JaxtariPostStepModPlugin`
     - `JaxAtariModController` → `JaxtariModController`
     - `JaxAtariModWrapper` → `JaxtariModWrapper`
     - `JaxAtariFuncEnv` → `JaxtariFuncEnv`
     - `JAXAtariAction` / `JaxAtariAction` → `JaxtariAction`
     - `JaxatariWrapper` → `JaxtariWrapper`
2. **Protected strings** (must keep the old token)
   - `LEGACY_APP_NAME = "jaxatari"` (sprite path fallback for older installs)
   - `user_data_dir("jaxatari")` / `user_data_dir('jaxatari')` (legacy fallbacks / docs)
   - `~/.local/share/jaxatari` and `$HOME/.local/share/jaxatari` (legacy path mentions)
   - WandB `ENTITY: "jaxatari"` / `ENTITY: 'jaxatari'`
   - Canonical install path is already `user_data_dir("jaxtari")` / `~/.local/share/jaxtari`
3. **Explicit non-targets**
   - `HackAtari`, `OCAtari`, `OC_Atari`
   - Game modules named `jax_*.py` (JAX library prefix, not the package)
   - The codemod itself, its unit tests, and this checklist
4. **Package move**: `src/jaxatari/` → `src/jaxtari/` (`git mv` when possible)
5. **Other path renames** embedding the old name (e.g. `ppo_jaxatari_scan.py`,
   `pqn_jaxatari_*.yaml`, `JAXAtari_Poster.pptx`)
6. **Compatibility shim**: recreate `src/jaxatari/__init__.py` as a deprecated
   re-export of `jaxtari`
7. **Class aliases** appended on defining modules (`modification.py`,
   `wrappers.py`, `gym_wrapper.py`, `environment.py`) so the old public names
   still resolve for one release cycle

```bash
# Preview
python scripts/rename_to_jaxtari.py --dry-run --verbose

# Apply on the rename branch
python scripts/rename_to_jaxtari.py --apply --verbose
python scripts/rename_to_jaxtari.py --check
```

---

## Checklist — before the rename commit

- [ ] All in-flight env PRs that need live fork CI are merged or otherwise settled
- [ ] Contributors asked to commit / stash dirty trees
- [ ] Create `dev-prerename` from current `dev` tip (**before** the rename merges)
- [ ] Tag that tip, e.g. `jaxatari-final`
- [ ] Open / update rename branch from the same tip (e.g. `rename/jaxtari`)
- [ ] Unit tests green: `pytest tests/test_rename_to_jaxtari.py`
- [ ] Codemod dry-run reviewed against the step list above

## Checklist — land the rename

- [ ] Run `python scripts/rename_to_jaxtari.py --apply` on the rename branch
- [ ] Run `python scripts/rename_to_jaxtari.py --check` (must be clean)
- [ ] Reinstall editable package: `pip install -e ".[dev]"` (or project equivalent)
- [ ] Smoke: `python -c "import jaxtari; print(jaxtari.list_available_games()[:3])"`
- [ ] Smoke deprecated path: `python -c "import jaxatari"` (expect `DeprecationWarning`)
- [ ] CI path greps now use `src/jaxtari/games/` (script should have done this)
- [ ] Merge rename PR into `dev`
- [ ] Merge `dev` → `master`

## Checklist — after `dev` / `master` are renamed

- [ ] Rename the GitHub repository `JAXAtari` → `jaxtari` (redirects are automatic)
- [ ] Do **not** create a new repo named `JAXAtari` under the same owner later
- [ ] Update local remotes when convenient:
      `git remote set-url origin <new-url>`
      (optional — GitHub redirects old clone/fetch/push URLs automatically;
      updating only avoids relying on that redirect)
- [ ] Spot-check external links (lab site, paper/camera-ready BibTeX, badges) if they hardcode the old name
- [ ] Confirm GitHub Pages / ReadTheDocs project slug if used (Pages is not redirected like git URLs)

## Checklist — PyPI

- [ ] Claim / register **`jaxtari`** on PyPI
- [ ] Claim / register **`jaxatari`** on PyPI in the same window (redirect / shim distribution)
- [ ] Tag `v1.0.0` from renamed `master`
- [ ] Publish `jaxtari` 1.0.0 as the real package
- [ ] Publish `jaxatari` as a thin dependency on `jaxtari` (or identical shim package)

## Checklist — late / stale contributions

- [ ] Point late old-tree PRs at `dev-prerename` (after widening fork CI to allow that base)
- [ ] Port path: merge into `dev-prerename` → run the same codemod → PR onto `dev`
- [ ] In-repo feature branches: rebase onto renamed `dev`, re-run the codemod if needed

## Explicitly not done by the script

- GitHub repository rename / transfer
- Creating `dev-prerename` or any other branch
- PyPI registration or uploads
- WandB entity / project account changes
- Migrating on-disk sprite data out of `user_data_dir("jaxatari")`
  (runtime already prefers `jaxtari` and falls back to `jaxatari`; installer
  writes new packs to `~/.local/share/jaxtari`)
- Rewriting other people’s forks
