# Rebuttal micro-parity (preliminary)

Protocol: OCAtari trajectory (tag=`play`) → **automated** OC→JAX object transfer (no oracle / no manual latent edits) → lockstep replay of the same actions for **n=150** frames.

Primary metric: unmodified object-position L1 (player/enemies). Pixel MAE is secondary. Values are **mean / median** across sync runs.

| game | L1 @0 mean | L1 @0 med | MAE @0 med | L1 @end mean | L1 @end med | MAE @end med | #syncs |
|------|-----------:|----------:|-----------:|-------------:|------------:|-------------:|-------:|
| freeway | 0.00 | 0.00 | 0.39 | 18.16 | 0.36 | 0.60 | 5 |
| montezumarevenge | 0.07 | 0.00 | 3.95 | 17.20 | 11.33 | 4.52 | 5 |
| pong | 0.00 | 0.00 | 1.10 | 20.19 | 15.27 | 1.45 | 5 |
| bankheist | 0.00 | 0.00 | 0.48 | 0.43 | 0.50 | 0.48 | 5 |
| seaquest | 1.27 | 1.00 | 1.02 | 10.37 | 2.33 | 1.43 | 5 |
| skiing | 6.96 | 0.86 | 24.27 | 97.81 | 111.91 | 24.72 | 5 |

## Per-object L1 (mean / median across syncs)

### freeway

| object | L1 @0 mean | L1 @0 med | L1 @end mean | L1 @end med | n_ok @end |
|--------|-----------:|----------:|-------------:|------------:|----------:|
| car_0 | 0.00 | 0.00 | 0.40 | 0.00 | 5 |
| car_1 | 0.00 | 0.00 | 31.80 | 0.00 | 5 |
| car_2 | 0.00 | 0.00 | 0.40 | 0.00 | 5 |
| car_3 | 0.00 | 0.00 | 31.80 | 0.00 | 5 |
| car_4 | 0.00 | 0.00 | 32.00 | 0.00 | 5 |
| car_5 | 0.00 | 0.00 | 32.00 | 0.00 | 5 |
| car_6 | 0.00 | 0.00 | 32.80 | 1.00 | 5 |
| car_7 | 0.00 | 0.00 | 0.20 | 0.00 | 5 |
| car_8 | 0.00 | 0.00 | 32.20 | 0.00 | 5 |
| car_9 | 0.00 | 0.00 | 0.40 | 0.00 | 5 |
| chicken | 0.00 | 0.00 | 5.80 | 0.00 | 5 |

Worst run: t0=391  L1@0=0.00  L1@end=87.36  gif=`worst_gifs/freeway_worst.gif`
Worst-run per-object L1@end: car_0=0.00, car_1=159.00, car_2=0.00, car_3=159.00, car_4=160.00, car_5=160.00, car_6=160.00, car_7=0.00, car_8=160.00, car_9=1.00, chicken=2.00

### montezumarevenge

| object | L1 @0 mean | L1 @0 med | L1 @end mean | L1 @end med | n_ok @end |
|--------|-----------:|----------:|-------------:|------------:|----------:|
| key | 0.00 | 0.00 | 0.00 | 0.00 | 5 |
| player | 0.20 | 0.00 | 38.20 | 30.00 | 5 |
| skull_0 | 0.00 | 0.00 | 8.33 | 5.00 | 3 |

Worst run: t0=1230  L1@0=0.00  L1@end=42.00  gif=`worst_gifs/montezumarevenge_worst.gif`
Worst-run per-object L1@end: key=0.00, player=84.00, skull_0=∞

### pong

| object | L1 @0 mean | L1 @0 med | L1 @end mean | L1 @end med | n_ok @end |
|--------|-----------:|----------:|-------------:|------------:|----------:|
| ball | 0.00 | 0.00 | 32.40 | 25.00 | 5 |
| enemy | 0.00 | 0.00 | 24.00 | 18.00 | 5 |
| player | 0.00 | 0.00 | 4.17 | 2.77 | 5 |

Worst run: t0=446  L1@0=0.00  L1@end=44.65  gif=`worst_gifs/pong_worst.gif`
Worst-run per-object L1@end: ball=74.00, enemy=58.00, player=1.96

### bankheist

| object | L1 @0 mean | L1 @0 med | L1 @end mean | L1 @end med | n_ok @end |
|--------|-----------:|----------:|-------------:|------------:|----------:|
| bank_0 | 0.00 | 0.00 | 0.00 | 0.00 | 5 |
| bank_1 | 0.00 | 0.00 | 0.00 | 0.00 | 5 |
| bank_2 | 0.00 | 0.00 | 0.00 | 0.00 | 4 |
| player | 0.00 | 0.00 | 1.60 | 2.00 | 5 |

Worst run: t0=1230  L1@0=0.00  L1@end=0.67  gif=`worst_gifs/bankheist_worst.gif`
Worst-run per-object L1@end: bank_0=0.00, bank_1=0.00, player=2.00

### seaquest

| object | L1 @0 mean | L1 @0 med | L1 @end mean | L1 @end med | n_ok @end |
|--------|-----------:|----------:|-------------:|------------:|----------:|
| diver_0 | — | — | — | — | 0 |
| diver_2 | 0.00 | 0.00 | 2.00 | 2.00 | 2 |
| diver_3 | 0.00 | 0.00 | 3.00 | 3.00 | 1 |
| missile | — | — | 5.00 | 5.00 | 1 |
| player | 0.00 | 0.00 | 9.80 | 2.00 | 5 |
| shark_0 | 3.00 | 3.00 | — | — | 0 |
| shark_3 | 3.00 | 3.00 | — | — | 0 |
| shark_6 | — | — | — | — | 0 |
| shark_9 | 1.00 | 1.00 | — | — | 0 |

Worst run: t0=391  L1@0=1.00  L1@end=43.00  gif=`worst_gifs/seaquest_worst.gif`
Worst-run per-object L1@end: diver_2=∞, player=43.00, shark_0=∞, shark_6=∞

### skiing

| object | L1 @0 mean | L1 @0 med | L1 @end mean | L1 @end med | n_ok @end |
|--------|-----------:|----------:|-------------:|------------:|----------:|
| flag_0 | 45.80 | 0.00 | 68.22 | 58.84 | 5 |
| flag_1 | — | — | 80.84 | 80.84 | 2 |
| flag_2 | — | — | — | — | 0 |
| flag_3 | — | — | — | — | 0 |
| mogul_0 | 0.00 | 0.00 | 105.06 | 90.84 | 5 |
| mogul_1 | 0.00 | 0.00 | 220.50 | 220.50 | 1 |
| skier | 6.00 | 6.00 | 23.17 | 26.17 | 5 |
| tree_0 | 0.00 | 0.00 | 126.06 | 145.16 | 5 |
| tree_1 | 0.00 | 0.00 | 98.48 | 83.50 | 5 |
| tree_2 | 0.00 | 0.00 | 116.79 | 121.20 | 5 |
| tree_3 | 0.00 | 0.00 | 149.20 | 157.58 | 4 |

Worst run: t0=922  L1@0=17.86  L1@end=126.20  gif=`worst_gifs/skiing_worst.gif`
Worst-run per-object L1@end: flag_0=119.87, flag_1=20.87, flag_2=∞, flag_3=∞, mogul_0=119.87, mogul_1=∞, skier=14.76, tree_0=153.13, tree_1=186.13, tree_2=201.13, tree_3=193.87

Reports: `/home/paul/Documents/JAXAtari/data/oc_parity_corpus/reports_rebuttal`
Gallery: `/home/paul/Documents/JAXAtari/data/oc_parity_corpus/final_report/rebuttal/start_end_gallery`
Worst GIFs: `/home/paul/Documents/JAXAtari/data/oc_parity_corpus/final_report/rebuttal/worst_gifs`

