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
  - `packaging/jaxatari-alias/` (separate PyPI distribution named `jaxatari`;
  must keep that name, directory, and its temporary import shim)
  - Root PyPI display name `JAXtari` (already set; left unchanged — normalizes
  to `jaxtari` on PyPI, distinct from the `jaxatari` alias)
4. **Package move**: `src/jaxatari/` → `src/jaxtari/` (`git mv` when possible)
5. **Removal hint**: recreate `src/jaxatari/__init__.py` as a **raising** stub
  (`ImportError` pointing at `import jaxtari`). This is not a working re-export;
   it only upgrades a bare `ModuleNotFoundError` into a clear message. A
   temporary *working* `import jaxatari` lives solely in
   `packaging/jaxatari-alias/`
6. **Hatch packaging**: root wheel/sdist lists both `src/jaxtari` and
  `src/jaxatari` (the raising hint)
7. **Other path renames** embedding the old name (e.g. `ppo_jaxatari_scan.py`,
  `pqn_jaxatari_*.yaml`, `JAXAtari_Poster.pptx`)
8. **Class aliases** appended on defining modules (`modification.py`,
  `wrappers.py`, `gym_wrapper.py`, `environment.py`) so the old public names
   still resolve for one release cycle

```bash
# Preview (also writes jaxtari_rename.log with every file:line old → new)
python scripts/rename_to_jaxtari.py --dry-run --verbose

# Apply on the rename branch
python scripts/rename_to_jaxtari.py --apply --verbose
python scripts/rename_to_jaxtari.py --check
```

Review `jaxtari_rename.log` (or `--log PATH`) before merging: every unprotected
token rewrite is listed with path, line, column, exact `old → new`, and the
source line. Path renames are listed in a second section. Use `--no-log` to
skip the file.

---



## Checklist — before the rename commit

- [x] All in-flight env PRs that need live fork CI are merged or otherwise settled
- [x] Contributors asked to commit / stash dirty trees
- [x] Create `dev-prerename` from current `dev` tip (**before** the rename merges)
- [x] Tag that tip, e.g. `jaxatari-final`
- [x] Open / update rename branch from the same tip (e.g. `rename/jaxtari`)
- [x] Unit tests green: `pytest tests/test_rename_to_jaxtari.py`
- [ ] Codemod dry-run reviewed against the step list above
- [x] PyPI alias at `packaging/jaxatari-alias/` ships a temporary `import jaxatari`
  ```
  shim with a “removed soon” warning (canonical path is `import jaxtari`)
  ```



## Checklist — land the rename

- [ ] Run `python scripts/rename_to_jaxtari.py --apply` on the rename branch
- [ ] Run `python scripts/rename_to_jaxtari.py --check` (must be clean)
- [ ] Reinstall editable package: `pip install -e ".[dev]"` (or project equivalent)
- [ ] Smoke: `python -c "import jaxtari; print(jaxtari.list_available_games()[:3])"`
- [ ] Smoke removal hint: `python -c "import jaxatari"` fails with `ImportError`
  ```
  mentioning `jaxtari` (not a silent success / bare ModuleNotFoundError)
  ```
- [ ] Spot-check root `pyproject.toml`: `name = "JAXtari"`, hatch lists
  ```
  `src/jaxtari` **and** `src/jaxatari`, entry point uses `jaxtari.install_sprites`
  ```
- [ ] Spot-check `packaging/jaxatari-alias/` untouched (`name = "jaxatari"`,
  ```
  working shim still warns + re-exports `jaxtari`)
  ```
- [ ] CI path greps now use `src/jaxtari/games/` (script should have done this)
- [ ] Merge rename PR into `dev`
- [ ] Merge `dev` → `master`



## Checklist — after `dev` / `master` are renamed

- [ ] Rename the GitHub repository `JAXAtari` → `jaxtari` (redirects are automatic)
- [ ] Do **not** create a new repo named `JAXAtari` under the same owner later
- [ ] Update local remotes when convenient:
  ```
  `git remote set-url origin <new-url>`
  (optional — GitHub redirects old clone/fetch/push URLs automatically;
  updating only avoids relying on that redirect)
  ```
- [ ] Spot-check external links (lab site, paper/camera-ready BibTeX, badges) if they hardcode the old name
- [ ] Confirm GitHub Pages / ReadTheDocs project slug if used (Pages is not redirected like git URLs)



## Checklist — PyPI

Layout already in tree:


| Distribution | Source                      | PyPI normalized name | Role                                                                                               |
| ------------ | --------------------------- | -------------------- | -------------------------------------------------------------------------------------------------- |
| `JAXtari`    | root `pyproject.toml`       | `jaxtari`            | Real package — `import jaxtari`; `import jaxatari` **raises** a helpful `ImportError`              |
| `jaxatari`   | `packaging/jaxatari-alias/` | `jaxatari`           | Temporary working shim: depends on `JAXtari`, re-exports with deprecation warning; **remove soon** |


- [x] Claim / register `jaxtari` on PyPI (covers display name `JAXtari`)
- [x] Claim / register `jaxatari` on PyPI in the same window
- [ ] Confirm root `name = "JAXtari"` is unchanged after the codemod
- [ ] Confirm `packaging/jaxatari-alias/` was **not** rewritten
- [ ] Confirm root hatch config ships `packages = ["src/jaxtari", "src/jaxatari"]`
- [ ] Sync versions: root `version` and alias `version` + `dependencies = ["JAXtari==…"]`
  ```
  must match the release you are publishing
  ```
- [ ] Tag the release from renamed `master` (e.g. `v0.1.0` / `v1.0.0` — match
  ```
  `pyproject.toml`)
  ```
- [ ] Build & publish the main package from the repo root:
  ```
  `python -m build && twine upload dist/*`
  ```
- [ ] Build & publish the alias from `packaging/jaxatari-alias/`:
  ```
  `python -m build && twine upload dist/*`
  ```
- [ ] Verify fresh venvs:
  ```
  - `pip install jaxtari` → `import jaxtari` works; `import jaxatari` raises
    `ImportError` mentioning the rename
  - `pip install jaxatari` → installs `JAXtari`, `import jaxatari` works with
    `DeprecationWarning` (alias overwrites the raising stub)
  ```
- [ ] After the GitHub repo rename, update Homepage / Repository URLs in
  ```
  `packaging/jaxatari-alias/` if they still point at the old slug
  ```
- [ ] Later: yank / stop publishing the `jaxatari` alias once the warning window ends



## Checklist — late / stale contributions

- [ ] Point late old-tree PRs at `dev-prerename` (after widening fork CI to allow that base)
- [ ] Port path: merge into `dev-prerename` → run the same codemod → PR onto `dev`
- [ ] In-repo feature branches: rebase onto renamed `dev`, re-run the codemod if needed



## Explicitly not done by the script

- GitHub repository rename / transfer
- Creating `dev-prerename` or any other branch
- PyPI registration or uploads (including building `packaging/jaxatari-alias`)
- Editing `packaging/jaxatari-alias/` (intentionally left alone)
- WandB entity / project account changes
- Migrating on-disk sprite data out of `user_data_dir("jaxatari")`
(runtime already prefers `jaxtari` and falls back to `jaxatari`; installer
writes new packs to `~/.local/share/jaxtari`)
- Rewriting other people’s forks

