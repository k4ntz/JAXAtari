from jaxatari.modification import JaxAtariInternalModPlugin


class GodModeMod(JaxAtariInternalModPlugin):
    """
    God Mode: Harry cannot die, pits are solid under him, and logs neither
    block nor drain him. Speed and the jump arc stay at the ROM's normal rate.

    The base environment already implements all of this behind the runtime
    `PitfallState.god_mode` flag, which every reset initializes from
    `consts.debug_god_mode`. The mod only pins that constant at construction
    time; the base constant's default value is not altered. No method is
    patched and no state is introduced, so the modded environment stays
    JIT-compilable and vectorizable exactly like the base environment.
    """

    constants_overrides = {
        "debug_god_mode": True,
    }


class StationaryScorpionMod(JaxAtariInternalModPlugin):
    """
    Stationary scorpion: the scorpion remains at its current position.

    Patch-point note: `jax_pitfall.py` has no scorpion-update function — the
    movement is inline in `JaxPitfall.step()` (the `move_tick`/`scorpion_x`
    update guarded by `consts.scorpion_move_period_steps`: the ROM moves the
    scorpion one pixel every period, default 4 steps). This mod therefore
    uses the existing period constant instead of a method patch: with the
    period set to 2**30, the move tick is unreachable (`frame_cnt` only
    increments while gameplay is active, is bounded by the episode length,
    and resets every episode), so the scorpion never leaves its position.
    Scorpion state, method signatures, return values, shapes and dtypes are
    all untouched.
    """

    constants_overrides = {
        "scorpion_move_period_steps": 2**30,
    }


class FastPlayerMod(JaxAtariInternalModPlugin):
    """
    Fast player: doubles Harry's horizontal speed through the existing
    `consts.player_speed` (ROM rate 1 px/step -> 2 px/step).

    The same constant feeds every horizontal movement in `JaxPitfall`, so all
    of them become faster: ground running (`vx` in `step`), the airborne
    jump carry (`jump_launch_vx`), the ladder walk-off (`walk_dx` in
    `_apply_ladder`), the ladder-exit hop (`exit_vx`) and the liana release
    (`release_vx`). Vertical physics (gravity, fall_speed, jump arc height)
    are unchanged.
    """

    constants_overrides = {
        "player_speed": 2.0,
    }
