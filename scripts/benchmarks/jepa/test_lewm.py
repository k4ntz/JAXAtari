"""
Tests for the LeWM implementation.

These cover the properties that are easy to break silently and expensive to
notice later — the ones where a bug produces plausible-looking numbers rather
than a crash. Several are regression tests for bugs that actually occurred.

Deliberately fast: nothing here constructs a JAXtari environment, so the whole
file runs in seconds.

    pytest scripts/benchmarks/jepa/test_lewm.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lewm_jaxatari import (
    CNNEncoder, LeWM, Predictor, SequenceBuffer, effective_rank, episode_windows,
)


# ---------------------------------------------------------------------------
# Collapse diagnostic
# ---------------------------------------------------------------------------

def test_effective_rank_detects_collapse():
    """A collapsed representation must score near 1, an isotropic one near D."""
    d = 32
    direction = torch.randn(1, d)
    collapsed = direction * torch.randn(512, 1)      # all on one line
    isotropic = torch.randn(512, d)

    assert effective_rank(collapsed) < 1.5, "collapse should read as rank ~1"
    assert effective_rank(isotropic) > d * 0.7, "isotropic should use most dims"
    assert effective_rank(collapsed) < effective_rank(isotropic)


def test_effective_rank_is_scale_invariant():
    """Rank measures how many directions are used, not how large they are."""
    x = torch.randn(256, 16)
    assert effective_rank(x) == pytest.approx(effective_rank(x * 100.0), rel=1e-3)


# ---------------------------------------------------------------------------
# Predictor: AdaLN zero-init and causality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action_cond", ["adaln", "add"])
def test_predictor_shapes(action_cond):
    p = Predictor(emb_dim=32, n_actions=6, n_heads=4, n_layers=2,
                  action_cond=action_cond)
    out = p(torch.randn(3, 5, 32), torch.randint(0, 6, (3, 5)))
    assert out.shape == (3, 5, 32)


def test_adaln_blocks_are_identity_at_init():
    """Zero-initialised modulation means every gate starts at 0, so the stack of
    blocks is exactly the identity. This is the whole point of AdaLN-zero: the
    predictor begins as a no-op rather than injecting random action noise."""
    p = Predictor(emb_dim=32, n_actions=4, n_heads=4, n_layers=2,
                  action_cond="adaln").eval()
    emb = torch.randn(2, 6, 32)
    act = torch.randint(0, 4, (2, 6))

    x = emb + p.pos_emb[:, :6]
    expected = x.clone()
    mask = nn.Transformer.generate_square_subsequent_mask(6)
    for blk in p.blocks:
        x = blk(x, p.act_emb(act), mask)

    assert torch.allclose(x, expected, atol=1e-6)


@pytest.mark.parametrize("action_cond", ["adaln", "add"])
def test_predictor_is_causal(action_cond):
    """Changing frame t must not alter any prediction before t. A broken mask
    leaks the future and quietly inflates every prediction metric."""
    p = Predictor(emb_dim=32, n_actions=4, n_heads=4, n_layers=2,
                  action_cond=action_cond).eval()
    a = torch.randint(0, 4, (1, 6))
    e1 = torch.randn(1, 6, 32)
    e2 = e1.clone()
    e2[:, -1] += 5.0                      # perturb only the last frame

    with torch.no_grad():
        o1, o2 = p(e1, a), p(e2, a)
    assert torch.allclose(o1[:, :-1], o2[:, :-1], atol=1e-4)


def test_adaln_uses_actions():
    """After a gradient step the gates are non-zero, so different actions must
    give different predictions — otherwise conditioning is silently dead."""
    p = Predictor(emb_dim=32, n_actions=4, n_heads=4, n_layers=2,
                  action_cond="adaln")
    emb = torch.randn(4, 5, 32)
    p(emb, torch.randint(0, 4, (4, 5))).sum().backward()
    with torch.no_grad():
        for q in p.parameters():
            if q.grad is not None:
                q += 0.1 * torch.randn_like(q)

    p.eval()
    with torch.no_grad():
        o_a = p(emb, torch.zeros(4, 5, dtype=torch.long))
        o_b = p(emb, torch.full((4, 5), 3, dtype=torch.long))
    assert not torch.allclose(o_a, o_b, atol=1e-5)


# ---------------------------------------------------------------------------
# Sequence collection
# ---------------------------------------------------------------------------

def test_episode_windows_never_straddle_a_reset():
    """Windows are cut inside one episode. A window spanning a reset would train
    the model on a discontinuity that never occurs at run time."""
    n, seq_len = 40, 7
    obs = np.arange(n, dtype=np.uint8).reshape(n, 1, 1, 1)
    acts = np.arange(n, dtype=np.int64)

    for w_obs, w_act in episode_windows(obs, acts, seq_len, max_windows=10):
        assert len(w_obs) == seq_len + 1
        flat = w_obs.reshape(-1)
        assert np.array_equal(flat, np.arange(flat[0], flat[0] + seq_len + 1))
        assert np.array_equal(w_act, np.arange(w_act[0], w_act[0] + seq_len + 1))


def test_episode_windows_respects_cap_and_short_episodes():
    obs = np.zeros((100, 1, 1, 1), dtype=np.uint8)
    acts = np.zeros(100, dtype=np.int64)
    assert len(episode_windows(obs, acts, 7, max_windows=3)) == 3

    short = np.zeros((5, 1, 1, 1), dtype=np.uint8)      # shorter than seq_len+1
    assert episode_windows(short, np.zeros(5, dtype=np.int64), 7, 4) == []


def test_episode_windows_returns_copies():
    """Windows must own their memory; views would pin the whole episode array in
    the replay buffer and make its reported size wrong."""
    obs = np.zeros((20, 1, 1, 1), dtype=np.uint8)
    w_obs, _ = episode_windows(obs, np.zeros(20, dtype=np.int64), 7, 1)[0]
    assert w_obs.base is None


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

def test_buffer_stores_uint8_and_samples_unit_range():
    """Frames are kept as uint8 (4x smaller) and scaled only at sample time."""
    buf = SequenceBuffer(capacity=8, seq_len=3)
    for _ in range(4):
        buf.add(np.full((4, 3, 8, 8), 255, dtype=np.uint8),
                np.zeros(4, dtype=np.int64))

    assert buf.obs_buf[0].dtype == np.uint8
    obs, act = buf.sample(2)
    assert obs.dtype == torch.float32 and act.dtype == torch.int64
    assert obs.max() <= 1.0 and obs.min() >= 0.0
    assert obs.max() == pytest.approx(1.0)


def test_buffer_ring_overwrites_and_stays_capped():
    buf = SequenceBuffer(capacity=3, seq_len=1)
    for i in range(7):
        buf.add(np.full((2, 3, 4, 4), i, dtype=np.uint8), np.zeros(2, dtype=np.int64))
    assert len(buf) == 3
    assert buf.nbytes() == sum(o.nbytes for o in buf.obs_buf)


# ---------------------------------------------------------------------------
# Loss wiring
# ---------------------------------------------------------------------------

def test_stop_grad_flag_controls_target_gradient():
    """Faithful LeWM has NO stop-gradient — the paper's central claim depends on
    it. Guard against the flag silently inverting."""
    obs = torch.rand(2, 4, 3, 84, 84)
    act = torch.randint(0, 6, (2, 4))

    faithful = LeWM(n_actions=6, emb_dim=32, stop_grad=False)
    faithful(obs, act)["loss"].backward()
    grad_faithful = faithful.encoder.net[0].weight.grad.abs().sum().item()

    ablation = LeWM(n_actions=6, emb_dim=32, stop_grad=True)
    ablation(obs, act)["loss"].backward()
    grad_ablation = ablation.encoder.net[0].weight.grad.abs().sum().item()

    assert grad_faithful > 0 and grad_ablation > 0     # SIGReg reaches both
    assert not faithful.stop_grad and ablation.stop_grad


def test_sigreg_weight_zero_removes_the_term():
    obs, act = torch.rand(2, 4, 3, 84, 84), torch.randint(0, 6, (2, 4))
    model = LeWM(n_actions=6, emb_dim=32, sigreg_weight=0.0)
    out = model(obs, act)
    assert out["loss"].item() == pytest.approx(out["pred_loss"].item(), rel=1e-6)


# ---------------------------------------------------------------------------
# Regression: BatchNorm must stay frozen in the policy trunk
# ---------------------------------------------------------------------------

def test_finetuned_trunk_keeps_encoder_batchnorm_in_eval():
    """Regression test.

    PPO stores log-probs during the rollout and recomputes them during the
    update to build an importance ratio. If the encoder's BatchNorm runs in
    training mode, the two passes normalise by different statistics (rollout
    batches are ~32x smaller than update minibatches), the ratio compares two
    different functions, and training diverges — observed as value loss 1e15 and
    entropy 0. The trunk must therefore hold encoder BN in eval mode even when
    the encoder's weights are being fine-tuned.
    """
    from ppo_lewm import LeWMTrunk

    encoder = CNNEncoder(in_channels=3, emb_dim=16)
    trunk = LeWMTrunk(encoder, emb_dim=16, frame_stack=2, hidden=8, frozen=False)
    trunk.train()

    bns = [m for m in trunk.encoder.modules() if isinstance(m, nn.BatchNorm1d)]
    assert bns, "encoder should contain BatchNorm"
    assert all(not m.training for m in bns), "encoder BN must stay in eval mode"
    assert trunk.fc.training, "the learned head should still be in train mode"
    assert any(p.requires_grad for p in trunk.encoder.parameters()), \
        "fine-tuning should leave encoder weights trainable"


def test_frozen_trunk_blocks_encoder_gradients():
    from ppo_lewm import LeWMTrunk

    encoder = CNNEncoder(in_channels=3, emb_dim=16)
    trunk = LeWMTrunk(encoder, emb_dim=16, frame_stack=2, hidden=8, frozen=True)
    trunk.train()
    trunk(torch.rand(2, 2, 3, 84, 84)).sum().backward()

    assert all(p.grad is None for p in trunk.encoder.parameters())
    assert trunk.fc[0].weight.grad is not None


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_gives_same_initial_weights():
    """Every RNG the run touches is seeded; unseeded action sampling silently
    cost us reproducibility once already."""
    torch.manual_seed(7)
    a = LeWM(n_actions=6, emb_dim=32).encoder.net[0].weight.clone()
    torch.manual_seed(7)
    b = LeWM(n_actions=6, emb_dim=32).encoder.net[0].weight.clone()
    assert torch.equal(a, b)
