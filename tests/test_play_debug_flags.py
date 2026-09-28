"""The play script's level-skip / unlimited-lives / unlimited-dynamite flags.

    python scripts/play.py -g hero -l 5 -lifes -granades

Each flag is just a shorthand for a mod name, so the mapping is tested here
without opening a window.
"""
import os
import sys

import pytest

pytest.importorskip("pygame")

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import play  # noqa: E402  (scripts/play.py)


def _parse(argv):
    return play.build_parser().parse_args(argv)


def test_user_command_form_parses():
    args = _parse(["-g", "hero", "-l", "5", "-lifes", "-granades"])
    assert args.game == "hero"
    assert args.level == 5
    assert args.unlimited_lives is True
    assert args.unlimited_dynamite is True


def test_flags_default_off():
    args = _parse(["-g", "hero"])
    assert args.level is None
    assert args.unlimited_lives is False
    assert args.unlimited_dynamite is False
    assert play.debug_flag_mods(args) == []


def test_flags_map_to_mod_names():
    args = _parse(["-g", "hero", "-l", "5", "-lifes", "-granades"])
    assert play.debug_flag_mods(args) == ["start_level_5", "unlimited_lives", "unlimited_dynamite"]


def test_level_alone_maps_to_start_level_only():
    args = _parse(["-g", "hero", "-l", "12"])
    assert play.debug_flag_mods(args) == ["start_level_12"]


def test_long_spellings_are_accepted_too():
    args = _parse(["-g", "hero", "--level", "2", "--unlimited-lives", "--unlimited-dynamite"])
    assert play.debug_flag_mods(args) == ["start_level_2", "unlimited_lives", "unlimited_dynamite"]


def test_level_below_one_is_rejected():
    with pytest.raises(SystemExit):
        _parse(["-g", "hero", "-l", "0"])


def test_flags_append_to_explicit_mods():
    args = _parse(["-g", "hero", "-m", "unlimited_lives", "-l", "3"])
    mods = play._normalize_mods(args.mods)
    assert play.merge_debug_mods(mods, args) == ["unlimited_lives", "start_level_3"]
