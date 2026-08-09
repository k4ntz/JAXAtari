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
python run_all_games.py --outdir results/full --total_steps 5000

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

All 15 games train without collapse, and every one beats the frozen baseline.

**Ablations** (`results/ablations`, 5000 steps, Pong / Seaquest):

| variant | eff. rank | pred loss | beats frozen baseline |
|---|---|---|---|
| faithful | 44.1 / 31.5 | 0.0116 / 0.0149 | yes |
| no SIGReg | **1.5 / 1.6** | 0.0003 / 0.0002 | **no** |
| stop-gradient | 140.8 / 78.5 | 0.0728 / 0.0831 | yes |
| λ = 0.5 | 76.5 / 47.5 | 0.0374 / 0.0290 | yes |
| λ = 2.0 | 107.3 / 68.1 | 0.0827 / 0.0594 | yes |

Removing SIGReg improves the prediction loss ~39× while the effective rank falls
to 1.5 and the model then *loses to the trivial baseline*. This is the collapse the
regularizer exists to prevent, and it is invisible in the training loss.

Stop-gradient does **not** collapse — it gives the highest rank of any variant. It
simply produces a worse model. So the precise claim supported here is: SIGReg alone
prevents collapse, and adding stop-gradient costs predictive quality for nothing.
High rank is not itself good: λ = 2.0 has rank 107 and poor rollout error.

**Agents** (`results/ppo`, 1M steps, single seed):

| game | PPO from scratch | on frozen LeWM | on fine-tuned LeWM |
|---|---|---|---|
| pong | 15.3 | **19.0** | −14.0 |
| seaquest | **949** | 496 | 552 |
| breakout | **40.8** | 14.4 | 15.4 |

LeWM features make PPO markedly more sample-efficient on Pong (+19 by 0.4M steps
while the scratch CNN is still at −20) but plateau below it on Seaquest and
Breakout. Fine-tuning the encoder does not close that gap, which rules out the
obvious explanation that the frozen encoder is merely stale with respect to the
improving policy.

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
  MontezumaRevenge and Gravitar.
- **BatchNorm at evaluation.** `evaluate_rollout` keeps BatchNorm in
  batch-statistics mode for consistency with training. Using fixed running
  statistics instead changes rollout MSE by ~7% (measured on Pong); that is the
  size of the methodological wobble under these numbers.
- **Remaining deviations from the paper.** CNN encoder rather than ViT (argued
  above), and online random collection rather than an offline reward-free dataset.
- **The predictor is not used for control.** The paper evaluates LeWM by latent
  MPC planning; the agents here learn a policy on the encoder's representation
  instead. Planning in the latent space is the natural next step.
