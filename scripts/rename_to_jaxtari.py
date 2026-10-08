#!/usr/bin/env python3
"""Local codemod: jaxatari / JAXAtari / JaxAtari → jaxtari / Jaxtari.

This script only mutates the working tree. It does NOT rename the GitHub
repository, publish PyPI packages, create remote branches, or edit
``packaging/jaxatari-alias/`` (temporary PyPI ``jaxatari`` import shim —
canonical install/import is ``jaxtari``).

Designed to be:
  * idempotent (safe to re-run)
  * usable on older branches / forks before merging into renamed `dev`
  * dry-run capable (source tree untouched; occurrence log still written unless
    ``--no-log``)

Usage:
  python scripts/rename_to_jaxtari.py --dry-run
  python scripts/rename_to_jaxtari.py --apply
  python scripts/rename_to_jaxtari.py --check   # exit 1 if unprotected old names remain

By default ``--dry-run`` / ``--apply`` write a full occurrence log
(``jaxtari_rename.log``) listing every token rewrite with file, line, column,
and ``old → new``. Override with ``--log PATH`` or disable with ``--no-log``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLD_PKG = "jaxatari"
NEW_PKG = "jaxtari"
OLD_DIR_NAME = "jaxatari"
NEW_DIR_NAME = "jaxtari"

# Files / trees the codemod must never rewrite (they intentionally document /
# encode the old→new mapping, or are the separate PyPI alias distribution).
SCRIPT_REL = Path("scripts/rename_to_jaxtari.py")
TEST_REL = Path("tests/test_rename_to_jaxtari.py")
# Empty PyPI redirect package: must keep distribution name ``jaxatari`` and
# directory name ``jaxatari-alias``. Touched only by hand / release process.
PYPI_ALIAS_REL = Path("packaging/jaxatari-alias")
DEFAULT_LOG_NAME = "jaxtari_rename.log"
# Populated for the duration of a run when ``--log`` points inside the tree.
_RUNTIME_SKIP_FILES: set[str] = set()

SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    ".env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".local",
    "node_modules",
    ".eggs",
    "dist",
    "build",
    ".idea",
    ".vscode",
}

# Content rewrite is text-only. Binary / asset extensions are skipped.
BINARY_SUFFIXES = {
    ".npy",
    ".npz",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".ico",
    ".pdf",
    ".pptx",
    ".ppt",
    ".zip",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".whl",
    ".so",
    ".dylib",
    ".dll",
    ".o",
    ".a",
    ".pyc",
    ".pyo",
    ".pkl",
    ".pickle",
    ".safetensors",
    ".pt",
    ".pth",
    ".onnx",
    ".mp4",
    ".webm",
    ".wav",
    ".mp3",
    ".ogg",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".exe",
    ".bin",
}

TEXT_SUFFIXES = {
    ".py",
    ".pyi",
    ".toml",
    ".cfg",
    ".ini",
    ".txt",
    ".md",
    ".rst",
    ".yml",
    ".yaml",
    ".json",
    ".jsonl",
    ".sh",
    ".bash",
    ".zsh",
    ".csv",
    ".tsv",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".svg",
    ".in",
    ".cmake",
    ".mak",
    ".makefile",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".flake8",
    ".coveragerc",
    ".dockerignore",
}

# Identifier renames — longest first so prefixes do not double-substitute.
IDENTIFIER_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("GymnasiumJaxAtariWrapper", "GymnasiumJaxtariWrapper"),
    ("JaxAtariInternalModPlugin", "JaxtariInternalModPlugin"),
    ("JaxAtariPostStepModPlugin", "JaxtariPostStepModPlugin"),
    ("JaxAtariModController", "JaxtariModController"),
    ("JaxAtariModWrapper", "JaxtariModWrapper"),
    ("JaxAtariFuncEnv", "JaxtariFuncEnv"),
    ("JAXAtariAction", "JaxtariAction"),  # real class in environment.py
    ("JaxAtariAction", "JaxtariAction"),  # prose / type-hint spelling
    ("JaxatariWrapper", "JaxtariWrapper"),
    ("JaxAtari", "Jaxtari"),
    ("Jaxatari", "Jaxtari"),
    ("JAXAtari", "Jaxtari"),
    ("JAXATARI", "JAXTARI"),
    ("jaxatari", "jaxtari"),
)
IDENTIFIER_TO_NEW: dict[str, str] = dict(IDENTIFIER_REPLACEMENTS)
# Single left-to-right pass; alternation order is longest-first.
IDENTIFIER_PATTERN = re.compile(
    "|".join(re.escape(old) for old, _ in IDENTIFIER_REPLACEMENTS)
)

# Public symbols that need backward-compatible aliases after the rename.
# Maps defining-module path (relative to package root after move) → aliases.
ALIAS_TARGETS: dict[str, tuple[tuple[str, str], ...]] = {
    "modification.py": (
        ("JaxAtariInternalModPlugin", "JaxtariInternalModPlugin"),
        ("JaxAtariPostStepModPlugin", "JaxtariPostStepModPlugin"),
        ("JaxAtariModController", "JaxtariModController"),
        ("JaxAtariModWrapper", "JaxtariModWrapper"),
    ),
    "wrappers.py": (("JaxatariWrapper", "JaxtariWrapper"),),
    "gym_wrapper.py": (
        ("JaxAtariFuncEnv", "JaxtariFuncEnv"),
        ("GymnasiumJaxAtariWrapper", "GymnasiumJaxtariWrapper"),
    ),
    "environment.py": (
        ("JAXAtariAction", "JaxtariAction"),
        ("JaxAtariAction", "JaxtariAction"),
    ),
}

ALIAS_BANNER = (
    "\n# Backward-compatible aliases from the jaxatari → jaxtari rename.\n"
)

# Match alias assignment lines so a second --apply does not mangle them.
_ALIAS_LINE_RE = re.compile(
    r"^(\s*)((?:Gymnasium)?JaxAtari\w*|JAXAtariAction|JaxatariWrapper)"
    r"(\s*=\s*(?:Gymnasium)?Jaxtari\w*\s*)$",
    re.MULTILINE,
)

# In-tree stub shipped by the main wheel: ``import jaxatari`` fails with a
# pointer at ``jaxtari``. A working re-export lives only in the temporary PyPI
# alias at ``packaging/jaxatari-alias/``.
REMOVAL_HINT_INIT = '''\
"""Retired import name — use ``jaxtari``.

This module exists only so ``import jaxatari`` raises a clear ``ImportError``
instead of a bare ``ModuleNotFoundError``. It does **not** load the library.

Canonical usage::

    pip install jaxtari
    import jaxtari

A temporary PyPI distribution also named ``jaxatari`` still re-exports the
library with a deprecation warning; that alias will be removed soon.
"""

raise ImportError(
    "The 'jaxatari' import was renamed to 'jaxtari'. "
    "Use `import jaxtari` after `pip install jaxtari` "
    "(or `pip install -e .` from this repository). "
    "If you still need the old import temporarily, `pip install jaxatari` "
    "installs a short-lived alias package — it will be removed soon."
)
'''

# Placeholders protect strings that contain "jaxatari" but must NOT change.
# Order matters: apply before content rewrite, restore after.
PROTECT_PATTERNS: tuple[tuple[str, str], ...] = (
    # Legacy on-disk appdir constant (sprite path fallback for older installs).
    ('LEGACY_APP_NAME = "jaxatari"', "__JAXTARI_PROTECT_LEGACY_APP_NAME_DQ__"),
    ("LEGACY_APP_NAME = 'jaxatari'", "__JAXTARI_PROTECT_LEGACY_APP_NAME_SQ__"),
    # Direct legacy platformdirs lookups (docs / fallbacks).
    ('user_data_dir("jaxatari")', '__JAXTARI_PROTECT_USER_DATA_DIR_DQ__'),
    ("user_data_dir('jaxatari')", "__JAXTARI_PROTECT_USER_DATA_DIR_SQ__"),
    # Documented legacy filesystem paths for sprites.
    ("~/.local/share/jaxatari", "__JAXTARI_PROTECT_LOCAL_SHARE__"),
    ("$HOME/.local/share/jaxatari", "__JAXTARI_PROTECT_HOME_LOCAL_SHARE__"),
    # WandB entity is an account id, not the Python package name.
    ('ENTITY: "jaxatari"', '__JAXTARI_PROTECT_WANDB_ENTITY_DQ__'),
    ("ENTITY: 'jaxatari'", "__JAXTARI_PROTECT_WANDB_ENTITY_SQ__"),
    # Alias banner written by this script (contains the old name on purpose).
    (ALIAS_BANNER.strip("\n"), "__JAXTARI_PROTECT_ALIAS_BANNER__"),
    # Hatch keeps shipping the in-tree removal-hint package next to jaxtari.
    (
        'packages = ["src/jaxtari", "src/jaxatari"]',
        "__JAXTARI_PROTECT_HATCH_WHEEL_PACKAGES__",
    ),
    (
        'include = ["/src/jaxtari", "/src/jaxatari", "/README.md", "/LICENSE", "/pyproject.toml"]',
        "__JAXTARI_PROTECT_HATCH_SDIST_INCLUDE__",
    ),
)


@dataclass(frozen=True)
class ContentOccurrence:
    """One unprotected token rewrite inside a text file."""

    path: str
    line: int  # 1-based
    column: int  # 1-based, in the protected scan text
    old: str
    new: str
    line_text: str

    def format_log_line(self) -> str:
        snippet = self.line_text.rstrip("\n")
        return (
            f"{self.path}:{self.line}:{self.column}  "
            f"{self.old} → {self.new}\n"
            f"  | {snippet}"
        )


@dataclass
class RenameReport:
    content_files: list[str] = field(default_factory=list)
    content_occurrences: list[ContentOccurrence] = field(default_factory=list)
    renamed_paths: list[tuple[str, str]] = field(default_factory=list)
    package_moved: bool = False
    aliases_updated: list[str] = field(default_factory=list)
    skipped_protected_hits: int = 0
    removal_hint_written: bool = False

    def summarize(self) -> str:
        lines = [
            f"content files changed: {len(self.content_files)}",
            f"content occurrences:   {len(self.content_occurrences)}",
            f"paths renamed:         {len(self.renamed_paths)}",
            f"package moved:         {self.package_moved}",
            f"alias modules updated: {len(self.aliases_updated)}",
            f"protected hits kept:   {self.skipped_protected_hits}",
            f"removal hint written:  {self.removal_hint_written}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def is_excluded_dir(path: Path) -> bool:
    return path.name in SKIP_DIR_NAMES


def is_tooling_file(rel: Path) -> bool:
    posix = rel.as_posix()
    if posix in {
        SCRIPT_REL.as_posix(),
        TEST_REL.as_posix(),
    }:
        return True
    # Occurrence logs from prior runs (contain old→new on purpose).
    if (
        rel.name == DEFAULT_LOG_NAME
        or posix.endswith("/" + DEFAULT_LOG_NAME)
        or posix in _RUNTIME_SKIP_FILES
    ):
        return True
    # Entire PyPI alias tree (manifest + readme) keeps the old distribution name.
    alias_prefix = PYPI_ALIAS_REL.as_posix()
    return posix == alias_prefix or posix.startswith(alias_prefix + "/")


def should_rewrite_content(path: Path) -> bool:
    if path.suffix.lower() in BINARY_SUFFIXES:
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    # Extensionless text-ish files (Makefile, LICENSE, Dockerfile, …)
    if path.suffix == "" and path.is_file():
        name = path.name.upper()
        if name in {"MAKEFILE", "LICENSE", "DOCKERFILE", "AUTHORS", "NOTICE"}:
            return True
        if name.startswith("DOCKERFILE"):
            return True
    return False


def iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(is_excluded_dir(p) for p in path.relative_to(root).parents):
            continue
        if is_excluded_dir(path):
            continue
        rel = path.relative_to(root)
        if is_tooling_file(rel):
            continue
        yield path


def git_available(root: Path) -> bool:
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def move_path(root: Path, src: Path, dst: Path, *, apply: bool) -> None:
    if not apply:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if git_available(root) and (root / ".git").exists():
        result = subprocess.run(
            ["git", "mv", str(src), str(dst)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        # Fall through for untracked paths.
    src.rename(dst)


# ---------------------------------------------------------------------------
# Text transforms
# ---------------------------------------------------------------------------

def protect(text: str) -> tuple[str, int]:
    hits = 0
    for old, token in PROTECT_PATTERNS:
        count = text.count(old)
        if count:
            hits += count
            text = text.replace(old, token)

    def _mask_alias(match: re.Match[str]) -> str:
        nonlocal hits
        hits += 1
        # Hex-encode the LHS so leftover-name scanners cannot see JaxAtari*.
        encoded = match.group(2).encode("ascii").hex()
        return f"{match.group(1)}__JAXTARI_PROTECT_ALIAS_LHS_{encoded}__{match.group(3)}"

    text = _ALIAS_LINE_RE.sub(_mask_alias, text)
    return text, hits


def unprotect(text: str) -> str:
    for old, token in PROTECT_PATTERNS:
        text = text.replace(token, old)

    def _restore_alias(match: re.Match[str]) -> str:
        lhs = bytes.fromhex(match.group(1)).decode("ascii")
        return lhs

    text = re.sub(
        r"__JAXTARI_PROTECT_ALIAS_LHS_([0-9a-f]+)__",
        _restore_alias,
        text,
    )
    return text


def line_col_at(text: str, index: int) -> tuple[int, int]:
    """Return 1-based (line, column) for ``index`` in ``text``."""
    line = text.count("\n", 0, index) + 1
    last_nl = text.rfind("\n", 0, index)
    col = index + 1 if last_nl < 0 else index - last_nl
    return line, col


def line_text_at(text: str, line: int) -> str:
    """Return the 1-based line from ``text`` (without inventing a trailing newline)."""
    lines = text.splitlines()
    if 1 <= line <= len(lines):
        return lines[line - 1]
    return ""


def rewrite_identifiers(text: str) -> str:
    return IDENTIFIER_PATTERN.sub(
        lambda m: IDENTIFIER_TO_NEW[m.group(0)],
        text,
    )


def collect_identifier_occurrences(
    protected_text: str,
    *,
    original_text: str,
    rel_path: str,
) -> list[ContentOccurrence]:
    """List every identifier match that will be rewritten (protected regions excluded)."""
    occurrences: list[ContentOccurrence] = []
    for match in IDENTIFIER_PATTERN.finditer(protected_text):
        old = match.group(0)
        line, col = line_col_at(protected_text, match.start())
        occurrences.append(
            ContentOccurrence(
                path=rel_path,
                line=line,
                column=col,
                old=old,
                new=IDENTIFIER_TO_NEW[old],
                line_text=line_text_at(original_text, line),
            )
        )
    return occurrences


def transform_text(text: str) -> tuple[str, int]:
    updated, hits, _ = transform_text_detailed(text)
    return updated, hits


def transform_text_detailed(
    text: str,
    *,
    rel_path: str = "",
) -> tuple[str, int, list[ContentOccurrence]]:
    protected, hits = protect(text)
    occurrences = collect_identifier_occurrences(
        protected,
        original_text=text,
        rel_path=rel_path,
    )
    rewritten = rewrite_identifiers(protected)
    return unprotect(rewritten), hits, occurrences


def rewrite_filename(name: str) -> str:
    """Rename path components that encode the old project name.

    Game modules like ``jax_pong.py`` are intentionally untouched: they use the
    ``jax_`` prefix for the JAX library, not the package name.
    """
    new = name
    for old, repl in (
        ("JAXAtari", "Jaxtari"),
        ("JaxAtari", "Jaxtari"),
        ("Jaxatari", "Jaxtari"),
        ("jaxatari", "jaxtari"),
        ("JAXATARI", "JAXTARI"),
    ):
        new = new.replace(old, repl)
    return new


def ensure_aliases(module_text: str, aliases: tuple[tuple[str, str], ...]) -> str:
    """Append ``OldName = NewName`` aliases if missing."""
    missing = [(old, new) for old, new in aliases if f"{old} =" not in module_text]
    if not missing:
        return module_text
    block = ALIAS_BANNER + "".join(f"{old} = {new}\n" for old, new in missing)
    if not module_text.endswith("\n"):
        module_text += "\n"
    return module_text + block


def ensure_hatch_ships_removal_hint(text: str) -> str:
    """Ship the in-tree ``jaxatari`` removal-hint package beside ``jaxtari``."""
    if 'packages = ["src/jaxtari", "src/jaxatari"]' not in text:
        text = text.replace(
            'packages = ["src/jaxtari"]',
            'packages = ["src/jaxtari", "src/jaxatari"]',
        )
    if '"/src/jaxtari", "/src/jaxatari"' not in text:
        text = text.replace(
            'include = ["/src/jaxtari", "/README.md", "/LICENSE", "/pyproject.toml"]',
            'include = ["/src/jaxtari", "/src/jaxatari", "/README.md", "/LICENSE", "/pyproject.toml"]',
        )
    return text


def is_removal_hint_init(path: Path, root: Path) -> bool:
    return path.relative_to(root).as_posix() == f"src/{OLD_DIR_NAME}/__init__.py"


def _is_removal_hint_text(text: str) -> bool:
    return "raise ImportError" in text and "renamed to 'jaxtari'" in text


def _is_legacy_reexport_shim_text(text: str) -> bool:
    lowered = text.lower()
    return "deprecated" in lowered and (
        f"import {NEW_PKG}" in lowered or f"from {NEW_PKG}" in lowered
    )


REMAINING_OLD_NAME = re.compile(
    r"JAXAtari|JaxAtari|Jaxatari|JAXATARI|(?<![A-Za-z0-9_])jaxatari(?![A-Za-z0-9_])"
)


def remaining_unprotected_matches(text: str) -> list[str]:
    """Return unprotected old-name hits (after applying the same protects)."""
    protected, _ = protect(text)
    return REMAINING_OLD_NAME.findall(protected)


def step_rewrite_file_contents(root: Path, *, apply: bool, report: RenameReport) -> None:
    for path in iter_files(root):
        if not should_rewrite_content(path):
            continue
        # Post-rename removal hint must keep saying ``jaxatari``.
        if is_removal_hint_init(path, root):
            try:
                existing = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if _is_removal_hint_text(existing):
                continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(root).as_posix()
        updated, hits, occurrences = transform_text_detailed(
            original, rel_path=rel
        )
        report.skipped_protected_hits += hits
        if rel == "pyproject.toml":
            updated = ensure_hatch_ships_removal_hint(updated)
        if updated == original:
            continue
        report.content_files.append(rel)
        report.content_occurrences.extend(occurrences)
        if apply:
            path.write_text(updated, encoding="utf-8")


def step_rename_paths(root: Path, *, apply: bool, report: RenameReport) -> None:
    """Rename files/dirs whose names contain the old package token.

    Deepest paths first so children move before parents.
    """
    candidates: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path == root:
            continue
        if any(is_excluded_dir(p) for p in path.relative_to(root).parents):
            continue
        if is_excluded_dir(path):
            continue
        rel = path.relative_to(root)
        if is_tooling_file(rel):
            continue
        # Package directory is handled in a dedicated step.
        if rel.as_posix() in {f"src/{OLD_DIR_NAME}", f"src/{NEW_DIR_NAME}"}:
            continue
        if rewrite_filename(path.name) != path.name:
            candidates.append(path)

    for src in candidates:
        if not src.exists():
            continue
        new_name = rewrite_filename(src.name)
        dst = src.with_name(new_name)
        if dst.exists() and dst.resolve() != src.resolve():
            raise FileExistsError(f"Cannot rename {src} → {dst}: destination exists")
        report.renamed_paths.append(
            (src.relative_to(root).as_posix(), dst.relative_to(root).as_posix())
        )
        move_path(root, src, dst, apply=apply)


def _is_hint_only_package(pkg_dir: Path) -> bool:
    """True if ``pkg_dir`` is only a single ``__init__.py`` (hint or old re-export)."""
    if not pkg_dir.is_dir():
        return False
    leftover_modules = [p for p in pkg_dir.glob("*.py") if p.name != "__init__.py"]
    has_subdirs = any(p.is_dir() and p.name != "__pycache__" for p in pkg_dir.iterdir())
    return (pkg_dir / "__init__.py").exists() and not leftover_modules and not has_subdirs


def step_move_package(root: Path, *, apply: bool, report: RenameReport) -> None:
    old_pkg = root / "src" / OLD_DIR_NAME
    new_pkg = root / "src" / NEW_DIR_NAME
    if new_pkg.exists() and not old_pkg.exists():
        return
    if new_pkg.exists() and old_pkg.exists():
        # Expected post-rename layout: real package + removal-hint stub.
        if _is_hint_only_package(old_pkg):
            return
        raise FileExistsError(
            f"Both src/{OLD_DIR_NAME} and src/{NEW_DIR_NAME} exist; resolve manually."
        )
    if not old_pkg.exists():
        return
    report.package_moved = True
    report.renamed_paths.append((f"src/{OLD_DIR_NAME}", f"src/{NEW_DIR_NAME}"))
    move_path(root, old_pkg, new_pkg, apply=apply)


def step_write_removal_hint(root: Path, *, apply: bool, report: RenameReport) -> None:
    """Write ``src/jaxatari/__init__.py`` that raises a helpful ImportError."""
    new_pkg = root / "src" / NEW_DIR_NAME
    if not new_pkg.exists():
        return
    hint_dir = root / "src" / OLD_DIR_NAME
    hint_init = hint_dir / "__init__.py"
    if hint_init.exists():
        current = hint_init.read_text(encoding="utf-8")
        if _is_removal_hint_text(current):
            return
        # Replace a leftover working re-export shim with the raising hint.
        if not _is_hint_only_package(hint_dir) and not _is_legacy_reexport_shim_text(
            current
        ):
            return
    report.removal_hint_written = True
    if apply:
        if hint_dir.exists() and not _is_hint_only_package(hint_dir):
            # Should not happen after a clean move; refuse to clobber a full tree.
            return
        if hint_dir.exists():
            shutil.rmtree(hint_dir)
        hint_dir.mkdir(parents=True, exist_ok=True)
        hint_init.write_text(REMOVAL_HINT_INIT, encoding="utf-8")


def step_add_aliases(root: Path, *, apply: bool, report: RenameReport) -> None:
    pkg = root / "src" / NEW_DIR_NAME
    if not pkg.exists():
        return
    for rel, aliases in ALIAS_TARGETS.items():
        path = pkg / rel
        if not path.exists():
            continue
        original = path.read_text(encoding="utf-8")
        updated = ensure_aliases(original, aliases)
        if updated == original:
            continue
        report.aliases_updated.append(f"src/{NEW_DIR_NAME}/{rel}")
        if apply:
            path.write_text(updated, encoding="utf-8")


def _alias_modules_needing_update(pkg_dir: Path) -> list[str]:
    needed: list[str] = []
    if not pkg_dir.exists():
        return needed
    for rel, aliases in ALIAS_TARGETS.items():
        path = pkg_dir / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        # Dry-run reads the pre-rename tree; simulate the content pass.
        transformed, _ = transform_text(text)
        if ensure_aliases(transformed, aliases) != transformed:
            needed.append(f"src/{NEW_DIR_NAME}/{rel}")
    return needed


def run_rename(root: Path, *, apply: bool) -> RenameReport:
    root = root.resolve()
    report = RenameReport()
    # 1) Rewrite file contents while paths still use the old layout so diffs
    #    stay easy to review on a pre-rename tree.
    step_rewrite_file_contents(root, apply=apply, report=report)
    # 2) Move the package directory.
    step_move_package(root, apply=apply, report=report)
    # 3) Rename other files/dirs that embed the old name.
    step_rename_paths(root, apply=apply, report=report)
    # 4) Removal-hint stub + class aliases for one release cycle.
    if apply:
        step_write_removal_hint(root, apply=True, report=report)
        step_add_aliases(root, apply=True, report=report)
    else:
        old_pkg = root / "src" / OLD_DIR_NAME
        new_pkg = root / "src" / NEW_DIR_NAME
        if new_pkg.exists() or old_pkg.exists():
            report.removal_hint_written = True
            probe_pkg = new_pkg if new_pkg.exists() else old_pkg
            report.aliases_updated.extend(_alias_modules_needing_update(probe_pkg))
    return report


def check_tree(root: Path) -> list[str]:
    """Return relative paths that still contain unprotected old names."""
    root = root.resolve()
    offenders: list[str] = []
    for path in iter_files(root):
        rel = path.relative_to(root)
        # Filenames
        if REMAINING_OLD_NAME.search(path.name):
            # Allow the in-tree removal-hint package directory / file names.
            if rel.parts[:2] == ("src", OLD_DIR_NAME):
                pass
            else:
                offenders.append(f"{rel.as_posix()}  [filename]")
                continue
        if not should_rewrite_content(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits = remaining_unprotected_matches(text)
        if hits:
            # Removal hint intentionally mentions jaxatari.
            if rel.as_posix() == f"src/{OLD_DIR_NAME}/__init__.py":
                continue
            offenders.append(f"{rel.as_posix()}  {sorted(set(hits))}")
    return offenders


_GITHUB_URL_RE = re.compile(r"https?://github\.com/[^\s\"'<>)\]}]+")


def collect_github_url_rewrites(
    report: RenameReport,
    *,
    root: Path,
) -> list[tuple[str, str, str]]:
    """Return ``(path, before_url, after_url)`` for GitHub URLs the codemod rewrites.

    Also lists URLs under excluded trees (e.g. ``packaging/jaxatari-alias/``) that
    still contain the old repo slug and will need a **manual** update after the
    GitHub repository rename.
    """
    found: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(path: str, before: str, after: str) -> None:
        key = (path, before)
        if key in seen or before == after:
            return
        seen.add(key)
        found.append((path, before, after))

    for occ in report.content_occurrences:
        if "github.com" not in occ.line_text:
            continue
        for match in _GITHUB_URL_RE.finditer(occ.line_text):
            before = match.group(0).rstrip(".,;")
            after, _ = transform_text(before)
            if before != after:
                _add(occ.path, before, after)

    # Manual follow-ups in trees the codemod intentionally skips.
    for rel_root in (PYPI_ALIAS_REL,):
        base = root / rel_root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or not should_rewrite_content(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            rel = path.relative_to(root).as_posix()
            for match in _GITHUB_URL_RE.finditer(text):
                before = match.group(0).rstrip(".,;")
                if not REMAINING_OLD_NAME.search(before):
                    continue
                after, _ = transform_text(before)
                # Flag for humans even if after == before (e.g. only case change
                # desired later to ``jaxtari``); still show transformed form.
                _add(f"{rel}  [manual — excluded from codemod]", before, after)

    found.sort(key=lambda row: (row[0], row[1]))
    return found


def format_occurrence_log(
    report: RenameReport,
    *,
    root: Path,
    mode: str,
) -> str:
    """Render the full per-occurrence rename log."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    github_urls = collect_github_url_rewrites(report, root=root)
    lines: list[str] = [
        f"# Jaxtari rename occurrence log",
        f"# mode={mode}",
        f"# root={root}",
        f"# generated={stamp}",
        f"# content_occurrences={len(report.content_occurrences)}",
        f"# path_renames={len(report.renamed_paths)}",
        f"# protected_hits_kept={report.skipped_protected_hits}",
        f"# github_urls={len(github_urls)}",
        "",
        "## Content replacements",
        "",
    ]
    if report.content_occurrences:
        for occ in report.content_occurrences:
            lines.append(occ.format_log_line())
            lines.append("")
    else:
        lines.append("(none)")
        lines.append("")

    lines.append("## Path renames")
    lines.append("")
    if report.renamed_paths:
        for old, new in report.renamed_paths:
            lines.append(f"{old} → {new}")
    else:
        lines.append("(none)")
    lines.append("")

    lines.append("## GitHub URLs needing rename")
    lines.append("")
    lines.append(
        "Codemod rewrites ``JAXAtari`` → ``Jaxtari`` inside these URLs. "
        "After the GitHub repo is renamed to ``jaxtari``, prefer the final slug "
        "``https://github.com/<owner>/jaxtari`` (GitHub redirects are case-"
        "insensitive, but updating avoids relying on redirects)."
    )
    lines.append("")
    if github_urls:
        for path, before, after in github_urls:
            lines.append(f"{path}")
            lines.append(f"  before: {before}")
            lines.append(f"  after:  {after}")
            lines.append("")
    else:
        lines.append("(none)")
        lines.append("")
    return "\n".join(lines)


def write_occurrence_log(path: Path, report: RenameReport, *, root: Path, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_occurrence_log(report, root=root, mode=mode),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (default: parent of scripts/)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned changes without writing",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Apply the rename to the working tree",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if unprotected old names remain",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="List every content file / path rename",
    )
    log_group = parser.add_mutually_exclusive_group()
    log_group.add_argument(
        "--log",
        type=Path,
        default=None,
        help=(
            f"Write full occurrence log to this path "
            f"(default: <root>/{DEFAULT_LOG_NAME})"
        ),
    )
    log_group.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write an occurrence log file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = (args.root or default_repo_root()).resolve()
    if not root.exists():
        print(f"error: root does not exist: {root}", file=sys.stderr)
        return 2

    if args.check:
        offenders = check_tree(root)
        if offenders:
            print(f"Unprotected old-name hits under {root}:")
            for line in offenders:
                print(f"  {line}")
            return 1
        print(f"OK: no unprotected old names under {root}")
        return 0

    log_path: Path | None
    if args.no_log:
        log_path = None
    elif args.log is not None:
        log_path = args.log.expanduser().resolve()
    else:
        log_path = (root / DEFAULT_LOG_NAME).resolve()

    _RUNTIME_SKIP_FILES.clear()
    if log_path is not None:
        try:
            log_rel = log_path.relative_to(root).as_posix()
        except ValueError:
            log_rel = None
        if log_rel is not None:
            _RUNTIME_SKIP_FILES.add(log_rel)

    try:
        report = run_rename(root, apply=args.apply)
    finally:
        _RUNTIME_SKIP_FILES.clear()

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] root={root}")
    print(report.summarize())
    if log_path is not None:
        write_occurrence_log(log_path, report, root=root, mode=mode)
        print(f"occurrence log:      {log_path}")
    if args.verbose:
        if report.content_occurrences:
            print("\nContent occurrences:")
            for occ in report.content_occurrences:
                print(f"  {occ.path}:{occ.line}:{occ.column}  {occ.old} → {occ.new}")
        elif report.content_files:
            print("\nContent files:")
            for rel in report.content_files:
                print(f"  {rel}")
        if report.renamed_paths:
            print("\nRenames:")
            for old, new in report.renamed_paths:
                print(f"  {old} → {new}")
        if report.aliases_updated:
            print("\nAlias modules:")
            for rel in report.aliases_updated:
                print(f"  {rel}")
    if args.apply:
        print(
            "\nNext: run `python scripts/rename_to_jaxtari.py --check`, then "
            "handle remote / PyPI steps (repo rename, alias package) by hand."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
