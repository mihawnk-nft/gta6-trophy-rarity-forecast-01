# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mihawnk-nft/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/mihawnk-nft/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per trophy, scored by mean pinball loss on the logit scale.

## Method

An Exophase earn rate is a ratio — players who earned the trophy divided by players tracked — and the denominator jumps whenever Exophase refreshes its counts, so the numerator and denominator are forecast separately: `log N` and `log(E + 0.5)` are each fit by least squares on `log(1 + hours since launch)` over the last 24 hours of reads and extrapolated to the target. Each trophy's slope is shrunk toward its class-mean slope, weighted by how many players earned it (launch-night counts are 4–13 players per trophy, so tiny-count trophies lean on their class).

The data-driven rate is blended in logit space with a per-class day-7 prior anchor grounded in long-run GTA V / RDR2 rates: story chapters 60/45/35/25%, strand and mission finals ~8%, the final mission 12%, 100% completion 0.5%, grinds 2–8%, easy misc 8–30%, the platinum 0.1% (200+ hours), the opening mission 90%. The data weight is 0.7 when a trophy's earned count grew ≥3× with a tight fit, 0.4 when it grew 1.5–3×, else 0.2 — so a flat launch-night history leans on the prior, which is correct because it says nothing about the coming refresh.

The spread is deliberately wide: sigma combines a random-walk term growing with the horizon, the fit residual, a 0.7-logit refresh-move scale (the first refresh moved individual trophies ~0.3–0.9 logit), and 0.5-logit prior uncertainty, floored at 0.8 and capped at 2.5. On the practice cases the house baselines scored ~1.51 with ~0% 90% coverage because their bands were too narrow for the refresh; this forecast's bands are built to cover it.

## Run it

```sh
python3 main.py
```

Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (or `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (or `/output`). Python 3.13, standard library only, no network, deterministic (two runs are byte-identical).

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
