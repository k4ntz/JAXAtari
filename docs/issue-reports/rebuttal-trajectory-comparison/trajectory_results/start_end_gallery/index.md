# Start vs end lockstep gallery

Horizon n=150 (~5s at 30 FPS play / 2.5s at 60 Hz ALE).
Each panel: **OC | JAX | |diff|** at inject (start) and after replay (end).

| game | soft_div best | soft_div worst | note |
|------|---------------|----------------|------|
| freeway | surv | 91 | hard to tell |
| montezumarevenge | surv | 0 | hard to tell |
| pong | surv | 46 | hard to tell |
| bankheist | surv | 18 | hard to tell |
| seaquest | 66 | 22 | partial |
| skiing | 0 | 0 | easy to tell / early diverge |

## How to read
- Left→right in each strip: ALE/OC frame, JAX render, absolute pixel diff.
- Top = inject (t=0); bottom = after n steps (or traj end).
- If the claim is 'hard to tell after 5s', look at **best/** and especially games tagged surv / high soft_div.
- Early bright diff at END ⇒ claim fails for that title without manual latent/RNG alignment.

Per-game: `/home/paul/Documents/JAXAtari/data/oc_parity_corpus/final_report/rebuttal/start_end_gallery/best` and `/home/paul/Documents/JAXAtari/data/oc_parity_corpus/final_report/rebuttal/start_end_gallery/worst`.
