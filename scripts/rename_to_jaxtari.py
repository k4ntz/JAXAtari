#!/usr/bin/env python3
"""Local codemod: jaxatari / JAXAtari / JaxAtari → jaxtari / Jaxtari.

This script only mutates the working tree. It does NOT rename the GitHub
repository, publish PyPI packages, or create remote branches. Those steps live
in docs/jaxtari_rename_checklist.md.

Designed to be:
  * idempotent (safe to re-run)
  * usable on older branches / forks before merging into renamed `dev`
  * dry-run capable

Usage:
  python scripts/rename_to_jaxtari.py --dry-run
  python scripts/rename_to_jaxtari.py --apply
  python scripts/rename_to_jaxtari.py --check   # exit 1 if unprotected old names remain
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLD_PKG = "jaxatari"
NEW_PKG = "jaxtari"
OLD_DIR_NAME = "jaxatari"
NEW_DIR_NAME = "jaxtari"

# Files the codemod must never rewrite (they intentionally document / encode
# the old→new mapping).
SCRIPT_REL = Path("scripts/rename_to_jaxtari.py")
TEST_REL = Path("tests/test_rename_to_jaxtari.py")
CHECKLIST_REL = Path("docs/jaxtari_rename_checklist.md")

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

SHIM_INIT = '''\
"""Deprecated import path. Prefer ``import jaxtari``."""

from __future__ import annotations

import warnings

warnings.warn(
    "The 'jaxatari' import path is deprecated; use 'jaxtari' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from jaxtari import *  # noqa: F403
from jaxtari import (  # noqa: F401
    ALT_SPRITES_MARKER_FILE,
    DATA_DIR,
    MARKER_FILE,
    check_ownership,
    list_available_games,
    make,
)

__all__ = [
    "ALT_SPRITES_MARKER_FILE",
    "DATA_DIR",
    "MARKER_FILE",
    "check_ownership",
    "list_available_games",
    "make",
]
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
)


@dataclass
class RenameReport:
    content_files: list[str] = field(default_factory=list)
    renamed_paths: list[tuple[str, str]] = field(default_factory=list)
    package_moved: bool = False
    shim_written: bool = False
    aliases_updated: list[str] = field(default_factory=list)
    skipped_protected_hits: int = 0

    def summarize(self) -> str:
        lines = [
            f"content files changed: {len(self.content_files)}",
            f"paths renamed:         {len(self.renamed_paths)}",
            f"package moved:         {self.package_moved}",
            f"shim written:          {self.shim_written}",
            f"alias modules updated: {len(self.aliases_updated)}",
            f"protected hits kept:   {self.skipped_protected_hits}",
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
    return rel.as_posix() in {
        SCRIPT_REL.as_posix(),
        TEST_REL.as_posix(),
        CHECKLIST_REL.as_posix(),
    }


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


def rewrite_identifiers(text: str) -> str:
    for old, new in IDENTIFIER_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def transform_text(text: str) -> tuple[str, int]:
    protected, hits = protect(text)
    rewritten = rewrite_identifiers(protected)
    return unprotect(rewritten), hits


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


REMAINING_OLD_NAME = re.compile(
    r"JAXAtari|JaxAtari|Jaxatari|JAXATARI|(?<![A-Za-z0-9_])jaxatari(?![A-Za-z0-9_])"
)


def remaining_unprotected_matches(text: str) -> list[str]:
    """Return unprotected old-name hits (after applying the same protects)."""
    protected, _ = protect(text)
    return REMAINING_OLD_NAME.findall(protected)


def is_shim_init(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    return rel == f"src/{OLD_DIR_NAME}/__init__.py"


def step_rewrite_file_contents(root: Path, *, apply: bool, report: RenameReport) -> None:
    for path in iter_files(root):
        if not should_rewrite_content(path):
            continue
        # Compatibility shim intentionally mentions the old import name.
        if is_shim_init(path, root):
            try:
                existing = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "deprecated" in existing.lower() and f"import {NEW_PKG}" in existing:
                continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated, hits = transform_text(original)
        report.skipped_protected_hits += hits
        if updated == original:
            continue
        rel = path.relative_to(root).as_posix()
        report.content_files.append(rel)
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


def step_move_package(root: Path, *, apply: bool, report: RenameReport) -> None:
    old_pkg = root / "src" / OLD_DIR_NAME
    new_pkg = root / "src" / NEW_DIR_NAME
    if new_pkg.exists() and not old_pkg.exists():
        return
    if new_pkg.exists() and old_pkg.exists():
        # Expected post-rename layout: real package + compatibility shim.
        leftover_modules = [
            p for p in old_pkg.glob("*.py") if p.name != "__init__.py"
        ]
        has_subdirs = any(p.is_dir() for p in old_pkg.iterdir())
        only_shim = (
            (old_pkg / "__init__.py").exists()
            and not leftover_modules
            and not has_subdirs
        )
        if only_shim:
            return
        raise FileExistsError(
            f"Both src/{OLD_DIR_NAME} and src/{NEW_DIR_NAME} exist; resolve manually."
        )
    if not old_pkg.exists():
        return
    report.package_moved = True
    report.renamed_paths.append((f"src/{OLD_DIR_NAME}", f"src/{NEW_DIR_NAME}"))
    move_path(root, old_pkg, new_pkg, apply=apply)


def step_write_shim(root: Path, *, apply: bool, report: RenameReport) -> None:
    shim_dir = root / "src" / OLD_DIR_NAME
    shim_init = shim_dir / "__init__.py"
    new_pkg = root / "src" / NEW_DIR_NAME
    # Only write the shim once the real package lives at the new path.
    if not new_pkg.exists():
        return
    if shim_init.exists():
        current = shim_init.read_text(encoding="utf-8")
        if "deprecated" in current.lower() and f"import {NEW_PKG}" in current:
            return
        # An old full package still at src/jaxatari means move did not happen.
        if (shim_dir / "core.py").exists():
            return
    report.shim_written = True
    if apply:
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim_init.write_text(SHIM_INIT, encoding="utf-8")


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
    # 4) Compatibility shim + class aliases.
    if apply:
        step_write_shim(root, apply=True, report=report)
        step_add_aliases(root, apply=True, report=report)
    else:
        old_pkg = root / "src" / OLD_DIR_NAME
        new_pkg = root / "src" / NEW_DIR_NAME
        if old_pkg.exists() or new_pkg.exists():
            report.shim_written = True
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
            # Allow the shim package directory / file names.
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
            # Shim may mention jaxatari in the deprecation warning by design.
            if rel.as_posix() == f"src/{OLD_DIR_NAME}/__init__.py":
                continue
            offenders.append(f"{rel.as_posix()}  {sorted(set(hits))}")
    return offenders


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

    report = run_rename(root, apply=args.apply)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] root={root}")
    print(report.summarize())
    if args.verbose:
        if report.content_files:
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
            "follow docs/jaxtari_rename_checklist.md for remote / PyPI steps."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
