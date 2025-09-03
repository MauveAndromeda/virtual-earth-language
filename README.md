# Virtual Earth Language (MWE v0.1.0)

**Goal (v0.1.0):** a minimal, reproducible example that shows a slot‑based, explainable code⇄explain (C↔E) language on a tiny referential game. It prints three metrics and five sample messages in one command.

> Future versions will add RL training (PPO/Gumbel-ST), consistency/align/learnability losses, population dynamics, and the virtual‑earth visualization.

## Quickstart
```bash
make setup
make run-min
make test
```

## What this version includes

* Deterministic C↔E codec with **slots**: `ACT, OBJ, ATTR, LOC`
* Tiny referential game (target vs distractor)
* Minimal metrics: **Success**, **Topo~** (proxy), **AvgLen**
* CI on GitHub Actions + MIT License

## Repo layout

```
src/
  ontology/slots.py         # slots & tiny vocab
  explain/codec.py          # C↔E reversible codec (human-readable DSL)
  envs/referential.py       # minimal referential game + distances
  agents/speaker.py         # maps semantics -> code & explain
  agents/listener.py        # parses code and picks candidate
  objectives/losses.py      # success/length/topo~ proxy
experiments/minimal_run.py  # prints 3 metrics + 5 samples
results/                    # run outputs (add your plots later)
.github/workflows/ci.yml    # CI: run minimal example & tests
Makefile                    # setup/run/test
LICENSE                     # MIT
```

## Sample output (will vary slightly by seed)

```
=== MWE Metrics ===
Success=1.000  Topo~=0.850  AvgLen=24.0

=== Samples (C ↔ E) ===
1. CODE=ACT:GO|OBJ:TRI|ATTR:RED|LOC:L01
2. CODE=ACT:GO|OBJ:SQ|ATTR:BLU|LOC:L02
...
```

## Next (v0.2) — real training & explainability losses

* Add RL (PPO/Gumbel-ST) and losses: Cons / Align(CTC) / Learn(few-shot listener)
* Export corpus, tiny LM training scripts, and visualization page

## License

MIT (see LICENSE)
