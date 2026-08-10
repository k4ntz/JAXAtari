# LeWM — a JEPA world model for JAXtari, and an agent on top of it

Implementation of **LeWorldModel** (Maes, Le Lidec, Scieur, LeCun, Balestriero,
2026 — [arXiv:2603.19312](https://arxiv.org/abs/2603.19312), reference code
[lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)) trained on JAXtari, plus
PPO agents that learn on its representations.

LeWM is a joint-embedding predictive architecture: it predicts the *embedding* of
the next frame rather than the next frame's pixels. The usual failure mode of that
objective is representation collapse — the encoder can drive the loss to zero by
mapping every frame to the same vector. LeWM's claim is that **SIGReg** (a sketched
isotropic-Gaussian regularizer) prevents this without the stop-gradient or EMA
target that BYOL-style methods rely on, so the implementation here trains
end-to-end with **no stop-gradient**.

That claim is tested rather than assumed — see *Results* below.

## Files

| file | what it is |
|---|---|
| `lewm_jaxatari.py` | the world model (CNN encoder, SIGReg, AdaLN Transformer predictor), data collection, training, open-loop rollout evaluation, plotting |
| `lewm_features.py` | loads a trained encoder as a frozen feature extractor, recomputing its BatchNorm statistics first |
| `ppo_lewm.py` | PPO with a pluggable trunk: from-scratch CNN, frozen LeWM, or fine-tuned LeWM |
| `run_all_games.py` | trains the world model on all 15 required games |
| `run_ablations.py` | one-variable-at-a-time variants: SIGReg weight, stop-gradient, action conditioning |
| `run_ppo_comparison.py` | runs the agent arms per game and plots them together |
| `test_lewm.py` | unit tests (`pytest scripts/benchmarks/jepa/test_lewm.py`, ~1 s) |

The model itself is one self-contained file; the others are experiment runners and
the downstream agent.

## Architecture

- **Encoder** — 3-layer CNN (the Nature-DQN trunk) → Linear → BatchNorm, giving a
  256-d embedding per frame. The paper uses a ViT; a CNN is substituted because
  Atari frames are small and the paper shows the method is encoder-agnostic
  (App. G).
- **Predictor** — 4-layer causal Transformer over the embedding sequence.
  Actions enter through **adaptive LayerNorm with a zero-initialised modulation
  head** (as in the paper), so at step 0 every gate is 0 and the predictor is
  exactly the identity. Output goes through a projector matching the encoder's
  (Linear → BatchNorm).
- **Loss** — `MSE(pred, target) + λ · SIGReg(embeddings)`, λ = 0.1. Targets come
  from the *same* encoder with gradients flowing through them.

`--action_cond add` swaps AdaLN for a simple additive action embedding, kept as an
ablation rather than deleted.

## Running

```bash
# world model: one game, then all 15
python lewm_jaxatari.py --game pong --total_steps 10000 --outdir results/pong
python run_all_games.py --outdir results/full --total_steps 5000 \
    --init_sequences 600 --collect_every 50 --collect_n 20 --eval_seq 64

# ablations (~1 h)
python run_ablations.py --outdir results/ablations --total_steps 5000

# agents — needs encoders from run_all_games.py first (~3.5 h)
python run_ppo_comparison.py --outdir results/ppo \
    --arms scratch lewm_frozen lewm_finetune --total_timesteps 1000000

# fast sanity check that all 15 games start (~3 min)
python run_all_games.py --outdir results/smoke --total_steps 60 --init_sequences 48
```

Each run writes `<outdir>/<game>/{curve.png, history.json, model.pt}` plus a
top-level `summary.json`, `summary.md` and an overview figure. Curves refresh
during training, so an interrupted run still leaves usable artifacts. `results/`
is gitignored.

## Reading the results

**Falling loss does not mean learning.** Collapse also lowers the loss — in fact
it lowers it *more*. Two diagnostics guard against this:

- **Effective rank** of the embeddings (out of 256). BatchNorm pins per-dimension
  variance to 1, so collapse does not show up as shrinking norms; it shows up as
  the embeddings occupying a low-dimensional subspace. Near 1 means collapse.
- **Open-loop rollout MSE against a frozen baseline.** The predictor is seeded
  with 3 true embeddings then run autoregressively on its own outputs, and scored
  against the trivial predictor that assumes the embedding never changes. MSE in a
  learned latent space has no absolute scale, so only the comparison is meaningful.

## Results

**14 of 15 games** train without collapse and beat the frozen baseline. MsPacman
does not (rollout 0.282 vs baseline 0.272) and is reported as a failure rather
than dropped.

**Ablations** (`results/ablations`, 5000 steps, Pong / Seaquest). Rollout error is
given as a **ratio to each variant's own frozen baseline**, because latent MSE has
no absolute scale and is not comparable across variants that define different
embedding geometries. Lower is better; 1.0 means no better than assuming nothing
changes.

| variant | eff. rank | pred loss | rollout / frozen |
|---|---|---|---|
| **faithful** (paper) | 28.4 / 28.5 | **0.0087 / 0.0113** | **0.286 / 0.235** |
| no SIGReg | **1.7 / 1.9** | 0.0010 / 0.0004 | **1.006** / 0.695 |
| stop-gradient + SIGReg | 174.4 / 68.6 | 0.0730 / 0.0526 | 0.379 / 0.356 |
| stop-gradient only | 149.4 / 42.8 | 0.0960 / 0.0942 | 0.430 / 0.689 |
| additive actions | 21.2 / 21.1 | 0.0092 / 0.0115 | 0.310 / **0.188** |
| λ = 0.5 | 58.9 / 44.6 | 0.0209 / 0.0225 | 0.289 / 0.305 |
| λ = 2.0 | 89.9 / 59.5 | 0.0399 / 0.0393 | 0.330 / 0.409 |

**Removing SIGReg collapses the representation.** Effective rank falls to 1.7 / 1.9
while the prediction loss *improves* 9x / 28x — the collapse is invisible in the
training loss, which is the entire reason the rank diagnostic exists. On Pong the
collapsed model then fails to beat even its own trivial baseline.

**But SIGReg is sufficient, not necessary.** The `stop_grad_only` control (λ = 0,
stop-gradient on) does *not* collapse either: rank 149 / 43. A plain stop-gradient
prevents collapse on its own here. What SIGReg buys is a much better model — 11x /
8x lower prediction error and a better rollout ratio than either stop-gradient
variant. So the claim this work supports is narrower than "SIGReg is what prevents
collapse": **SIGReg prevents collapse without a stop-gradient, and yields a
markedly better predictor than a stop-gradient does.** Adding a stop-gradient on
top of SIGReg only makes things worse.

High rank is not itself good — `stop_grad` has the highest rank of any variant
(174) and a worse rollout ratio than faithful.

**λ = 0.1 is the right setting on Atari.** Raising it monotonically increases rank
and degrades the rollout ratio on both games (0.286→0.289→0.330 and
0.235→0.305→0.409).

**AdaLN vs additive action conditioning is a wash at this scale.** AdaLN gives
slightly lower prediction loss on both games; the additive variant gives a better
rollout ratio on Seaquest and worse on Pong. This is reported rather than omitted:
the paper's conditioning scheme shows no measurable benefit here, and is kept for
fidelity rather than because it helps. (An earlier run had 25% of action labels
corrupted by sticky actions, which would have masked any difference; these numbers
are from the corrected pipeline and the conclusion is unchanged.)

## Known limitations

- **Budget.** Agents are trained for 1M steps against a 200M-step reference. The
  curves show learning trends, not converged performance.
- **Single seed.** The ablation effects are far too large for seed noise to
  explain, but the agent gaps (e.g. Pong +3.7) are within plausible seed variance.
- **Fine-tuning is constrained.** Pretrained encoder weights use
  `--encoder_lr_scale` (default 0.1). At the full rate training diverges; at this
  rate the encoder barely adapts. Fine-tuning here is caught between the two, so
  the fine-tuned arm should be read as "does not close the gap at this setting"
  rather than a general statement.
- **Random data-collection policy.** The world model only sees the state space a
  random agent reaches — a small slice on sparse-reward games such as
  MontezumaRevenge and Gravitar. Episodes run up to `--max_episode_steps` (500)
  with `episodic_life=False`, and windows are sampled across the whole episode, so
  coverage is not confined to the opening seconds; but it is still random-policy
  coverage.
- **MsPacman fails.** Its rollout error does not beat the frozen baseline. The
  ratio table above shows the same variant ordering holds elsewhere, so this looks
  game-specific rather than systematic, but it is not explained.
- **BatchNorm at evaluation.** `evaluate_rollout` keeps BatchNorm in
  batch-statistics mode for consistency with training. Using fixed running
  statistics instead changes rollout MSE by ~7% (measured on Pong); that is the
  size of the methodological wobble under these numbers.
- **Remaining deviations from the paper.** CNN encoder rather than ViT (argued
  above), and online random collection rather than an offline reward-free dataset.
- **The predictor is not used for control.** The paper evaluates LeWM by latent
  MPC planning; the agents here learn a policy on the encoder's representation
  instead. Planning in the latent space is the natural next step.
