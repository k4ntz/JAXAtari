# LeWM — a JEPA world model for JAXtari

Implementation of **LeWorldModel** (Maes, Le Lidec, Scieur, LeCun, Balestriero,
2026 — [arXiv:2603.19312](https://arxiv.org/abs/2603.19312), reference code
[lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)) trained on JAXtari
environments.

LeWM is a joint-embedding predictive architecture: it learns to predict the
*embedding* of the next frame rather than the next frame's pixels. The usual
failure mode of that objective is representation collapse — the encoder can drive
the loss to zero by mapping every frame to the same vector. LeWM's claim is that
**SIGReg** (a sketched isotropic-Gaussian regularizer) prevents this without the
stop-gradient / EMA-target machinery that BYOL-style methods rely on, which is
why the implementation here trains end-to-end with **no stop-gradient**.

## Files

| file | what it is |
|---|---|
| `lewm_jaxatari.py` | the model (CNN encoder, SIGReg, Transformer predictor), data collection, training loop, evaluation, plotting |
| `run_all_games.py` | trains every one of the 15 required games in-process and collects the results |

## Architecture

- **Encoder** — 3-layer CNN (the standard Nature-DQN trunk) → linear → BatchNorm,
  producing a 256-d embedding per frame. The paper uses a ViT; a CNN is
  substituted here because Atari frames are small and the CNN is much cheaper.
- **Predictor** — 4-layer causal Transformer encoder over the embedding sequence,
  with a learned embedding of the discrete action added at each position. Given
  embeddings `z_1..z_t` and actions `a_1..a_t` it predicts `z_2..z_{t+1}`.
- **Loss** — `MSE(pred, target) + sigreg_weight * SIGReg(embeddings)`, where the
  target embeddings are produced by the *same* encoder with gradients flowing
  through them (no stop-gradient — see above).

## Running

```bash
# a single game
python lewm_jaxatari.py --game pong --total_steps 10000 --outdir results/pong

# all 15 required games
python run_all_games.py --outdir results/full --total_steps 5000

# quick check that nothing is broken (~3 min for all 15 games)
python run_all_games.py --outdir results/smoke --total_steps 60 --init_sequences 48
```

### Output layout

```
results/full/
    summary.json          all games, machine-readable
    summary.md            the same table, paste-ready for the report
    all_games.png         grid of every learning curve
    pong/curve.png        per-game learning curve (refreshed during training)
    pong/history.json     per-game loss history + eval metrics
    pong/model.pt         final weights
```

`results/` is gitignored, so runs never end up in a commit.

## Reading the results

**Learning curves** plot total / prediction / SIGReg loss on the left axis and the
**effective rank** of the embeddings on the right. Falling loss on its own does not
mean the model learned anything — collapse also lowers the loss. Because BatchNorm
pins the per-dimension variance to 1, collapse does not show up as shrinking
embedding norms; it shows up as the embeddings occupying a low-dimensional
subspace. Effective rank (the entropy-based rank of the embedding matrix, out of
256) is therefore the diagnostic to watch: near 1 means collapse.

**Open-loop rollout MSE** is the real evaluation. On held-out trajectories the
predictor is seeded with 3 true embeddings and then run autoregressively on its
own outputs, with no teacher forcing, and compared against the true embeddings at
each horizon. It is reported next to a **frozen baseline** — the trivial predictor
that assumes the embedding never changes. A world model that does not beat the
frozen baseline has learned nothing useful, and the comparison matters because MSE
in a learned latent space has no absolute scale to interpret.

## Known limitations

- **Random data-collection policy.** Sequences come from a uniform-random policy,
  so the model only ever sees the part of the state space a random agent reaches.
  On sparse-reward games (montezumarevenge, gravitar) that is a small and
  unrepresentative slice.
- **BatchNorm during evaluation.** `evaluate_rollout` deliberately keeps BatchNorm
  in batch-statistics mode. Under online training the encoder's features are
  non-stationary, so BN's running averages are stale and inflate the embedding
  scale several-fold, which would make the reported numbers meaningless. A frozen
  target encoder, or recomputing BN statistics before evaluation, is the cleaner
  fix.
- **Sequences within episodes only.** Windows are cut inside a single episode and
  never straddle a reset, but the windows within one episode are consecutive and
  therefore correlated.
