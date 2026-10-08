"""Unit tests for scripts/rename_to_jaxtari.py.

These exercises a temporary miniature repo so the real tree is never mutated.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "rename_to_jaxtari.py"


def _load_rename_module():
    spec = importlib.util.spec_from_file_location("rename_to_jaxtari", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rename = _load_rename_module()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")


def _mini_repo(tmp_path: Path) -> Path:
    """Create a small pre-rename tree covering every rename surface."""
    root = tmp_path / "repo"
    _write(
        root / "pyproject.toml",
        """
        [project]
        name = "JAXtari"
        version = "0.1.0"

        [project.scripts]
        install-sprites = "jaxatari.install_sprites:download_and_extract"

        [tool.hatch.build.targets.wheel]
        packages = ["src/jaxatari"]

        [tool.hatch.build.targets.sdist]
        include = ["/src/jaxatari", "/README.md", "/LICENSE", "/pyproject.toml"]
        """,
    )
    _write(
        root / "packaging" / "jaxatari-alias" / "pyproject.toml",
        """
        [project]
        name = "jaxatari"
        version = "0.1.0"
        dependencies = ["JAXtari==0.1.0"]

        [tool.hatch.build.targets.wheel]
        packages = ["src/jaxatari"]
        """,
    )
    _write(
        root / "packaging" / "jaxatari-alias" / "README.md",
        """
        # jaxatari
        Temporary shim — prefer pip install jaxtari / import jaxtari.
        pip install jaxatari
        import jaxatari
        """,
    )
    _write(
        root / "packaging" / "jaxatari-alias" / "src" / "jaxatari" / "__init__.py",
        """
        import warnings
        warnings.warn("temporary", DeprecationWarning, stacklevel=2)
        from jaxtari import make
        """,
    )
    _write(
        root / "src" / "jaxatari" / "__init__.py",
        """
        from pathlib import Path
        from platformdirs import user_data_dir

        APP_NAME = "jaxatari"
        LEGACY_APP_NAME = "jaxatari"
        DATA_DIR = Path(user_data_dir(APP_NAME))
        MARKER_FILE = DATA_DIR / ".ownership_confirmed"
        ALT_SPRITES_MARKER_FILE = DATA_DIR / ".alternative_sprites_installed"

        def check_ownership():
            pass

        def make(name):
            return name

        def list_available_games():
            return []

        # Prose brand names mixed with imports.
        # JAXAtari / JaxAtari / Jaxatari should all become Jaxtari.
        """,
    )
    _write(
        root / "src" / "jaxatari" / "paths.py",
        """
        from platformdirs import user_data_dir

        APP_NAME = "jaxatari"
        LEGACY_APP_NAME = "jaxatari"

        def canonical_storage_dir():
            return user_data_dir(APP_NAME)

        def legacy_storage_dir():
            return user_data_dir(LEGACY_APP_NAME)
        """,
    )
    _write(
        root / "src" / "jaxatari" / "modification.py",
        """
        from jaxatari.wrappers import JaxatariWrapper

        class JaxAtariInternalModPlugin:
            pass

        class JaxAtariPostStepModPlugin:
            pass

        class JaxAtariModController:
            pass

        class JaxAtariModWrapper(JaxatariWrapper):
            pass
        """,
    )
    _write(
        root / "src" / "jaxatari" / "wrappers.py",
        """
        class JaxatariWrapper:
            pass
        """,
    )
    _write(
        root / "src" / "jaxatari" / "gym_wrapper.py",
        """
        class JaxAtariFuncEnv:
            pass

        class GymnasiumJaxAtariWrapper:
            pass
        """,
    )
    _write(
        root / "src" / "jaxatari" / "environment.py",
        """
        class JAXAtariAction:
            NOOP = 0
        """,
    )
    _write(
        root / "src" / "jaxatari" / "install_sprites.py",
        """
        import os
        from pathlib import Path
        from platformdirs import user_data_dir

        SPRITES_URL = os.environ.get("JAXATARI_SPRITES_URL", "https://example.test")
        STORAGE_DIR = Path(user_data_dir("jaxatari"))
        auto_accept = os.environ.get("JAXATARI_CONFIRM_OWNERSHIP", "0") == "1"
        # Docs path: ~/.local/share/jaxatari/sprites
        """,
    )
    _write(
        root / "src" / "jaxatari" / "games" / "jax_pong.py",
        """
        # Game module filename must stay jax_pong.py (JAX library prefix).
        from jaxatari.environment import JaxEnvironment

        class JaxPong:
            pass
        """,
    )
    _write(
        root / "src" / "jaxatari" / "games" / "mods" / "pong_mods.py",
        """
        from jaxatari.modification import JaxAtariModController

        class PongEnvMod(JaxAtariModController):
            '''Inspired by HackAtari and OCAtari — those names must survive.'''
            pass
        """,
    )
    _write(
        root / ".github" / "workflows" / "ci.yml",
        """
        - run: rsync -av pr_head/src/jaxatari/games/ src/jaxatari/games/
        - run: JAXATARI_CONFIRM_OWNERSHIP=1 install-sprites
        """,
    )
    _write(
        root / "scripts" / "benchmarks" / "ppo_jaxatari_scan.py",
        """
        from ppo_jaxatari_vmap_eval import evaluate
        import jaxatari
        """,
    )
    _write(
        root / "scripts" / "benchmarks" / "ppo_jaxatari_vmap_eval.py",
        """
        def evaluate():
            return 0
        """,
    )
    _write(
        root / "scripts" / "benchmarks" / "config" / "config.yaml",
        """
        ENTITY: "jaxatari"
        PROJECT: "JAXAtari-Rebuttal"
        """,
    )
    _write(
        root / "scripts" / "benchmarks" / "config" / "alg" / "pqn_jaxatari_pixel.yaml",
        """
        NAME: pqn_jaxatari_pixel
        """,
    )
    _write(
        root / "README.md",
        """
        # JAXAtari
        Inspired by OCAtari and HackAtari.
        Put sprites in ~/.local/share/jaxatari
        import jaxatari
        """,
    )
    # Binary-ish asset with old name in filename should be renamed, not opened.
    asset = root / "docs" / "JAXAtari_Poster.bin"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"\x00\x01jaxatari\xff")
    return root


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return _mini_repo(tmp_path)


def test_dry_run_does_not_mutate(repo: Path):
    before = {
        p.relative_to(repo).as_posix(): p.read_bytes()
        for p in repo.rglob("*")
        if p.is_file()
    }
    report = rename.run_rename(repo, apply=False)
    assert report.content_files
    assert report.package_moved
    after = {
        p.relative_to(repo).as_posix(): p.read_bytes()
        for p in repo.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_apply_moves_package_and_rewrites_imports(repo: Path):
    rename.run_rename(repo, apply=True)

    assert (repo / "src" / "jaxtari" / "modification.py").exists()
    assert (repo / "src" / "jaxtari" / "__init__.py").exists()
    assert not (repo / "src" / "jaxatari").exists()

    init = (repo / "src" / "jaxtari" / "__init__.py").read_text(encoding="utf-8")
    assert 'APP_NAME = "jaxtari"' in init
    assert 'LEGACY_APP_NAME = "jaxatari"' in init

    paths = (repo / "src" / "jaxtari" / "paths.py").read_text(encoding="utf-8")
    assert 'APP_NAME = "jaxtari"' in paths
    assert 'LEGACY_APP_NAME = "jaxatari"' in paths

    mods = (repo / "src" / "jaxtari" / "games" / "mods" / "pong_mods.py").read_text(
        encoding="utf-8"
    )
    assert "from jaxtari.modification import JaxtariModController" in mods
    assert "class PongEnvMod(JaxtariModController)" in mods
    assert "HackAtari" in mods
    assert "OCAtari" in mods


def test_preserves_wandb_entity_and_rewrites_project(repo: Path):
    rename.run_rename(repo, apply=True)
    cfg = (
        repo / "scripts" / "benchmarks" / "config" / "config.yaml"
    ).read_text(encoding="utf-8")
    assert 'ENTITY: "jaxatari"' in cfg
    assert 'PROJECT: "Jaxtari-Rebuttal"' in cfg


def test_preserves_local_share_path(repo: Path):
    rename.run_rename(repo, apply=True)
    readme = (repo / "README.md").read_text(encoding="utf-8")
    assert "~/.local/share/jaxatari" in readme  # legacy path mention kept
    assert "# Jaxtari" in readme
    assert "OCAtari" in readme
    assert "HackAtari" in readme
    assert "import jaxtari" in readme


def test_transform_text_unit_cases():
    text, hits = rename.transform_text(
        'from jaxatari.modification import JaxAtariModController\n'
        'APP_NAME = "jaxatari"\n'
        'LEGACY_APP_NAME = "jaxatari"\n'
        'DATA = user_data_dir("jaxatari")\n'
        'ENTITY: "jaxatari"\n'
        "HackAtari and OCAtari\n"
        "JAXATARI_CONFIRM_OWNERSHIP=1\n"
    )
    assert hits == 3  # LEGACY_APP_NAME + user_data_dir + ENTITY
    assert "from jaxtari.modification import JaxtariModController" in text
    assert 'APP_NAME = "jaxtari"' in text
    assert 'LEGACY_APP_NAME = "jaxatari"' in text
    assert 'user_data_dir("jaxatari")' in text
    assert 'ENTITY: "jaxatari"' in text
    assert "HackAtari" in text and "OCAtari" in text
    assert "JAXTARI_CONFIRM_OWNERSHIP=1" in text


def test_content_occurrences_include_file_line_and_token(repo: Path):
    report = rename.run_rename(repo, apply=False)
    assert report.content_occurrences
    # Exact token + location for a known import line in the mini-repo.
    match = next(
        o
        for o in report.content_occurrences
        if o.path.endswith("pong_mods.py") and o.old == "jaxatari"
    )
    assert match.new == "jaxtari"
    assert match.line >= 1
    assert match.column >= 1
    assert "jaxatari" in match.line_text
    # Longer identifiers win over embedded shorter ones on the same match.
    controller = next(
        o
        for o in report.content_occurrences
        if o.old == "JaxAtariModController"
    )
    assert controller.new == "JaxtariModController"
    assert not any(
        o.path == controller.path
        and o.line == controller.line
        and o.old == "JaxAtari"
        and o.column == controller.column
        for o in report.content_occurrences
    )
    # Protected WandB entity must not appear as a rewrite.
    assert not any(
        o.path.endswith("config.yaml") and o.old == "jaxatari"
        for o in report.content_occurrences
    )


def test_cli_writes_occurrence_log(repo: Path, tmp_path: Path):
    log_path = tmp_path / "out" / "occurrences.log"
    assert (
        rename.main(
            ["--root", str(repo), "--dry-run", "--log", str(log_path)]
        )
        == 0
    )
    text = log_path.read_text(encoding="utf-8")
    assert "## Content replacements" in text
    assert "jaxatari → jaxtari" in text
    assert "JaxAtariModController → JaxtariModController" in text
    assert "pong_mods.py:" in text
    assert "## Path renames" in text
    assert "ppo_jaxatari_scan.py →" in text
    # Dry-run must not mutate sources; only the log is written.
    assert (repo / "src" / "jaxatari" / "__init__.py").exists()
    assert not (repo / "src" / "jaxtari").exists()


def test_renames_benchmark_scripts_and_configs(repo: Path):
    rename.run_rename(repo, apply=True)
    assert (repo / "scripts" / "benchmarks" / "ppo_jaxtari_scan.py").exists()
    assert not (repo / "scripts" / "benchmarks" / "ppo_jaxatari_scan.py").exists()
    assert (
        repo / "scripts" / "benchmarks" / "config" / "alg" / "pqn_jaxtari_pixel.yaml"
    ).exists()
    scan = (repo / "scripts" / "benchmarks" / "ppo_jaxtari_scan.py").read_text(
        encoding="utf-8"
    )
    assert "from ppo_jaxtari_vmap_eval import evaluate" in scan
    assert "import jaxtari" in scan


def test_does_not_rename_jax_game_modules(repo: Path):
    rename.run_rename(repo, apply=True)
    game = repo / "src" / "jaxtari" / "games" / "jax_pong.py"
    assert game.exists()
    text = game.read_text(encoding="utf-8")
    assert "from jaxtari.environment import JaxEnvironment" in text
    assert "class JaxPong" in text  # JaxPong is a game class, not JaxAtari*


def test_rewrites_ci_paths_and_env_vars(repo: Path):
    rename.run_rename(repo, apply=True)
    ci = (repo / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "src/jaxtari/games/" in ci
    assert "src/jaxatari/games/" not in ci
    assert "JAXTARI_CONFIRM_OWNERSHIP=1" in ci


def test_rewrites_pyproject(repo: Path):
    rename.run_rename(repo, apply=True)
    toml = (repo / "pyproject.toml").read_text(encoding="utf-8")
    # Display name JAXtari is already the post-rename PyPI name; leave it.
    assert 'name = "JAXtari"' in toml
    assert "jaxtari.install_sprites:download_and_extract" in toml
    assert 'packages = ["src/jaxtari"]' in toml
    assert 'include = ["/src/jaxtari", "/README.md", "/LICENSE", "/pyproject.toml"]' in toml


def test_does_not_create_legacy_import_package(repo: Path):
    rename.run_rename(repo, apply=True)
    assert not (repo / "src" / "jaxatari").exists()


def test_leaves_pypi_alias_package_untouched(repo: Path):
    rename.run_rename(repo, apply=True)
    alias_dir = repo / "packaging" / "jaxatari-alias"
    assert alias_dir.is_dir()
    assert not (repo / "packaging" / "jaxtari-alias").exists()
    toml = (alias_dir / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "jaxatari"' in toml
    assert 'dependencies = ["JAXtari==0.1.0"]' in toml
    assert 'packages = ["src/jaxatari"]' in toml
    readme = (alias_dir / "README.md").read_text(encoding="utf-8")
    assert "# jaxatari" in readme
    assert "pip install jaxatari" in readme
    assert "import jaxatari" in readme
    shim = (alias_dir / "src" / "jaxatari" / "__init__.py").read_text(encoding="utf-8")
    assert "from jaxtari import make" in shim
    assert "DeprecationWarning" in shim


def test_class_aliases_are_appended(repo: Path):
    rename.run_rename(repo, apply=True)
    mod = (repo / "src" / "jaxtari" / "modification.py").read_text(encoding="utf-8")
    assert "class JaxtariModController" in mod
    assert "JaxAtariModController = JaxtariModController" in mod
    assert "JaxAtariInternalModPlugin = JaxtariInternalModPlugin" in mod
    wrap = (repo / "src" / "jaxtari" / "wrappers.py").read_text(encoding="utf-8")
    assert "class JaxtariWrapper" in wrap
    assert "JaxatariWrapper = JaxtariWrapper" in wrap
    gym = (repo / "src" / "jaxtari" / "gym_wrapper.py").read_text(encoding="utf-8")
    assert "GymnasiumJaxtariWrapper" in gym
    assert "GymnasiumJaxAtariWrapper = GymnasiumJaxtariWrapper" in gym
    env = (repo / "src" / "jaxtari" / "environment.py").read_text(encoding="utf-8")
    assert "class JaxtariAction" in env
    assert "JAXAtariAction = JaxtariAction" in env
    assert "JaxAtariAction = JaxtariAction" in env


def test_renames_binary_filename_without_touching_bytes(repo: Path):
    original = (repo / "docs" / "JAXAtari_Poster.bin").read_bytes()
    rename.run_rename(repo, apply=True)
    renamed = repo / "docs" / "Jaxtari_Poster.bin"
    assert renamed.exists()
    assert renamed.read_bytes() == original
    assert b"jaxatari" in renamed.read_bytes()  # content untouched


def test_idempotent_second_apply(repo: Path):
    rename.run_rename(repo, apply=True)
    first_files = {
        p.relative_to(repo).as_posix(): p.read_bytes()
        for p in repo.rglob("*")
        if p.is_file()
    }
    report = rename.run_rename(repo, apply=True)
    assert report.content_files == []
    assert report.package_moved is False
    second_files = {
        p.relative_to(repo).as_posix(): p.read_bytes()
        for p in repo.rglob("*")
        if p.is_file()
    }
    assert first_files == second_files


def test_check_clean_after_apply(repo: Path):
    rename.run_rename(repo, apply=True)
    offenders = rename.check_tree(repo)
    assert offenders == []


def test_check_finds_leftover_unprotected_name(repo: Path):
    rename.run_rename(repo, apply=True)
    leftover = repo / "notes.md"
    leftover.write_text("still says JAXAtari here\n", encoding="utf-8")
    offenders = rename.check_tree(repo)
    assert any("notes.md" in line for line in offenders)


def test_rewrite_filename_leaves_jax_game_prefix():
    assert rename.rewrite_filename("jax_pong.py") == "jax_pong.py"
    assert rename.rewrite_filename("ppo_jaxatari_scan.py") == "ppo_jaxtari_scan.py"
    assert rename.rewrite_filename("JAXAtari_Poster.pptx") == "Jaxtari_Poster.pptx"


def test_cli_check_and_dry_run(repo: Path, capsys):
    assert rename.main(["--root", str(repo), "--dry-run", "--no-log"]) == 0
    assert (repo / "src" / "jaxatari" / "__init__.py").exists()
    assert rename.main(["--root", str(repo), "--apply", "--no-log"]) == 0
    assert rename.main(["--root", str(repo), "--check"]) == 0
    (repo / "oops.md").write_text("JAXAtari\n", encoding="utf-8")
    assert rename.main(["--root", str(repo), "--check"]) == 1
