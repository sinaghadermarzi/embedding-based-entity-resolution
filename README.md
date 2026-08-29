# er_lab — embedding-based entity resolution, measured honestly

A numbered notebook series (00–18 + appendix) over a thin `src/er_lab` package that studies
**entity resolution of person records under realistic, measured noise**, for designs that must
scale to 1e8–1e9 records. Deduplication is the primary task; one chapter (notebook 18) converts
the pipeline to two-source linkage. Privacy research — privacy-preserving record linkage,
embedding-inversion risk, the legal status of person embeddings — is **intentionally out of
scope** ([PLAN.md §1](PLAN.md)): experiments run on public-record and synthetic data, and the repo
ships download scripts, never data.

**Working hypothesis** (stated in [PLAN.md §1](PLAN.md), pre-registered before any model was
trained): *whole-record learned embeddings are a net-positive component of large-scale person ER —
with deterministic rules kept wherever they demonstrably earn their place.*

The lab's product is not a winning system; it is statistically sound, intuitively demonstrated
comparisons of competing designs — **including where the hypothesis fails**. Every claim-bearing
computation is preceded by an immutable, hash-frozen conjecture card and closed by a verdict box
that must honestly read CONFIRMED, REFUTED, or UNEXPLAINED off the computed numbers.

## The verdict — including the losses

The series verdict (notebook 17, card `VERDICT-1E7`): **CONFIRMED at smoke scale.** On the
16,095-record entity-disjoint eval half, at the locked precision@0.99 operating point, all three
paired B³F1 deltas are sign-stable across 500 shared entity resamples:

- **P1** — embeddings add value over the tuned Fellegi–Sunter stand-in: **+0.0376** [+0.0254, +0.0521]
- **P2** — the one earned rule still earns on top of embeddings: **+0.0338** [+0.0294, +0.0387]
- **P3** — the hybrid composes against FS: **+0.0714** [+0.0589, +0.0864]

All three deltas sit below the MET-04 replicate-noise bars (0.0862 single-seed / 0.0865
residual-inclusive) — the card's pre-registered rule tests sign-stability on shared resamples,
not cross-seed detectability, which is the node run's job.

"At smoke scale" is a real qualifier, not a hedge: notebook 17 ran live on a 4-CPU container with
a 32k-record corpus standing in for 1e7, and says so in its own outputs; the definitive
target-tier run is a `RUN-IN-TARGET node` placard, not a claim.

The lab's credibility is the refutations that stand right next to that verdict — the closing
ledger stands at **16 CONFIRMED / 7 REFUTED / 13 UNEXPLAINED** (notebook 17, recomputed from each
card's pre-registered decision rule):

- **BAS-02 REFUTED (notebook 12):** BM25 blocking meets or beats the trained dense encoder at
  *every* matched candidate budget — gap **+0.279** pair-completeness at k=25. The hypothesis's
  retrieval story loses to a sparse index at this tier, and notebook 17 quotes that refutation
  unrebutted.
- **HYB-01 REFUTED (notebook 14):** rules do not earn their place *as a class*. The plausible
  dob-year guard **hurts** (−0.0190 B³F1, sign-stable — 28.7% of the pairs it fires on are true
  matches), while the exact-key override earns (+0.0247, sign-stable). Each rule pays rent
  individually or not at all.
- **X01 REFUTED (appendix A, exploratory):** the scratch cross-encoder reranker made retrieval
  *worse* (ΔPC@10 −0.0403, sign-stable) at 29× the measured per-pair cost.
- Also refuted along the way: the folklore-grade Fellegi–Sunter number (BAS-01, notebook 06 —
  measured B³F 0.78 against the predicted 0.90, with the miss attributed to blocking, not FS), the
  chain-merge catastrophe (notebook 07 — percolation refused to happen), the NC acquisition plan
  (notebook 01), and the entity-ID-churn scare (X02 — 0.129% of records remap).

## Verified quickstart

```bash
uv sync
uv run python -m er_lab.run nb=00 run.tier=smoke
```

That executes notebook 00 headlessly at smoke tier: it downloads its own corpus
(`historical_50k`, fetched by script on first run — network required), runs a complete dedup
(blocking → scoring → clustering → entity-level metrics with BCa bootstrap CIs), and registers
two artifacts. The committed run of notebook 00 measures the dedup pipeline end-to-end at
**1.93 s wall-clock** on the reference container (4-CPU/15GB, CPU-only) — that number is
`quickstart_runtime_s` in the `smoke_check` run the committed notebook registered (2026-08-26),
printed in the notebook's own output,
and `smoke_check` re-measures **your** machine's throughput coefficients on every run: the only
source any wall-clock claim in this repo is allowed to cite. Python 3.11–3.12
(`requires-python = ">=3.11,<3.13"`).

Every config knob is a CLI dotlist override; `nb=all` runs the whole series in DAG order:

```bash
uv run python -m er_lab.run nb=09 run.tier=smoke train.loss=supcon infra.device=mps infra.precision=bf16
uv run python -m er_lab.run nb=all run.tier=smoke
```

The launcher forwards the dotlist to the executing notebook as JSON in `ER_LAB_DOTLIST`, and
notebooks apply it via `load_config_from_env()` — the cell you read is the cell that ran, under
the config you passed. Typo'd sections get a loud stderr warning. Defaults and the full key list:
[`configs/default.yaml`](configs/default.yaml).

## Tiers and hardware

One code path across all tiers — a tier only ever arrives through `run.tier`, never through
forked notebook logic. Artifacts from different tiers never mix: the registry refuses to serve a
smoke artifact to a target run.

| Tier | Hardware | What runs there |
|---|---|---|
| `smoke` | 4-CPU / 15GB CPU-only container (this repo's build class) | **The whole series, live.** Every number in the committed notebooks was computed here, on 1e5–1e6-scale substrates. |
| `mac` | Apple M2 Max, 96GB unified memory (MPS) | Mac-first policy: every heavy cell that fits, including all pretrained-encoder arms (Hugging Face is blocked from the smoke container — see [`notes/COMPAT.md`](notes/COMPAT.md)). |
| `node` | **One** machine: 4×A100 80GB, 900GB RAM, 90 cores | **The hard ceiling — nothing in the lab may require more.** Definitive multi-seed factorials, the 1e7 target runs, the real Verdict at 1e7. No cluster code paths exist. |
| `analytical` | CPU, minutes | 1e8–1e9 by validated extrapolation and cost modeling only — never a live run, always watermarked EXTRAPOLATED. |

Every heavy cell is inventoried in [`notes/RUN_IN_TARGET.md`](notes/RUN_IN_TARGET.md) (generated
by [`tools/gen_run_in_target.py`](tools/gen_run_in_target.py)): which notebook, which cell, `mac`
or `node`, and what it computes when you run it there.

## The notebooks

Verdicts below are the honest outcomes of each notebook's pre-registered cards, quoted from the
committed executed outputs. UNEXPLAINED is a first-class outcome — an effect can be real while its
claimed mechanism is not, and single-seed smoke results that await their definitive mac/node runs
say so.

| # | Notebook | The question | Honest outcome |
|---|---|---|---|
| 00 | The Hardest Easy Problem | Can one screenful of honest code dedup 50k person records — and what does "good" even mean? | **CONFIRMED**: B³ precision 0.996, recall 0.528 — blocking pair-completeness (0.486) is the measured ceiling. |
| 01 | Corpora and Declared Schemas | Are two county-filtered NC voter snapshots a usable same-person diff substrate? | **REFUTED**: duplicated-NCID share 3.2–3.7% vs the predicted <0.5%; the pre-registered adjustment rule kicked in. |
| 02 | How to Tell Who Won | Do metric choice, operating points, and bootstrap unit change who wins? | MET-01/02/03 all **CONFIRMED**: rankings reverse, per-method best-F1 lies, wrong resampling units under-cover. |
| 03 | Is the Truth True? | Is NCID usable as pseudo-truth, and how much of the corpus is unresolvable-in-principle? | MET-05 & MET-06 **CONFIRMED**: the gate passes with dup-NCID exclusions; the missing full DOB is the binding identifiability constraint. |
| 04 | Auditing Real Noise | What does two years of real drift actually do to person records? | NSE-01 **CONFIRMED**: wholesale address replacement dominates; family-name-change exposure is strongly sex-correlated. |
| 05 | The Calibrated Dirt Machine | Can a generator reproduce the measured noise, and which train/eval splits leak? | NSE-02 & MET-07 **CONFIRMED**: the calibrated generator tracks the measured error mix far better than generic corruption (one pre-stated acceptance threshold — name collision mass — honestly failed); the split machinery is leak-free (0 straddling entities vs 1,911 under a random split) and passes its null-instrument control. |
| 06 | The Baseline That Must Be Beaten | How strong is a genuinely well-tuned Fellegi–Sunter baseline? | **REFUTED**: B³F 0.78, not the predicted 0.90 — and the blocking-oracle ceiling (0.785) pins the miss on candidate generation, not FS. |
| 07 | The Chain-Merge Catastrophe | Does transitive closure detonate while accepted pairs still look excellent? | **REFUTED**: percolation denied by blocking + term frequency; the real fusions are small, dense household clusters. |
| 08 | A Person as a Vector | First embeddings — does serialization matter, and does the invariance battery behave? | TRN-05 **CONFIRMED** (single-seed scope); NB08-BATTERY **UNEXPLAINED**. The `full_name` leakage exclusion is measured and set here. |
| 09 | Pressures I: Losses and Negatives | Which loss/miner wins — and how large is replicate noise? | MET-04 **CONFIRMED** (seed sd 0.0305 → 73 seeds to detect a 0.01 delta: the lab's detectability bars); TRN-01/02 **UNEXPLAINED** pending the node-tier factorials. |
| 10 | Pressures II: Typos — Tokenizer or Augmentation? | Pretrained subword vs scratch char encoder × calibrated augmentation? | TRN-03/04 **UNEXPLAINED**: the pretrained side is RUN-IN-TARGET(mac) — HF is blocked at smoke, and the notebook says so instead of pretending. |
| 11 | Pressures III: Nicknames, Missing Fields, and the Circular Teacher | Does nickname supervision help, and does training on your own matcher's labels bite? | LABEL-PROV **CONFIRMED** (truth-labeled beats teacher-labeled by +0.074, sign-stable); TRN-06 & TRN-05-MISSING-STRESS **UNEXPLAINED**. |
| 12 | Finding Candidates in Ten Million | Dense ANN vs BM25 vs matchkeys at matched candidate budget? | BAS-02 **REFUTED**: BM25 ≥ dense at every matched budget, +0.279 pair-completeness at k=25. |
| 13 | From Pairs to People | Does calibration survive blocking bias, and how much does the clustering scheme matter? | CLU-01 **CONFIRMED** (scheme spread 0.064, below the 0.0865 replicate bar); CAL-01 **UNEXPLAINED**. |
| 14 | Where Rules Earn Their Place | Which deterministic rules pay rent, and does parsing or the noise model flip the ranking? | HYB-01 **REFUTED** as a class (guard −0.0190 hurts, override +0.0247 earns); PRS-01 **CONFIRMED** (parsing flips nothing); NSE-03 **UNEXPLAINED** (no sign-stable reversal across noise models). |
| 15 | The Last Two Orders of Magnitude | Do error-vs-n fits on small rungs predict the held-out largest rung? | SCL-01 **CONFIRMED** on the smoke ladder (with the threshold-transfer policy measured); SCL-02 **UNEXPLAINED**. Every point beyond the ladder renders dashed and watermarked EXTRAPOLATED. |
| 16 | Unequal Noise, Unequal Errors | Under measured group-correlated exposure, who pays? | FAIR-01 **UNEXPLAINED**: embeddings help both groups absolutely; the disparity tilt (+0.0112) sits below both detectability bars. Group membership SIMULATED, exposure multipliers measured — stated on every row. |
| 17 | The Verdict at 1e7 | The working hypothesis on trial under the full locked protocol. | VERDICT-1E7 **CONFIRMED** at smoke scale (deltas above), refutations re-rendered verbatim beside it; the honesty audit runs here (§8). |
| 18 | Your Data, Your Schema | Is adapting the lab to a new corpus a configuration exercise or an engineering project? | ADP-01 **UNEXPLAINED** (20 config lines, 0 package edits — but the battery acceptance test fails, 5/8 slices); ADP-02 **CONFIRMED** (dedup→linkage is one cross-source filter; the override's +0.0045 earns on real NCID truth). |
| A | Appendix: Exploratory Arms | Reranking, ID churn, static distillation — hypothesis-generating only. | X01 **REFUTED** (rerank hurts, −0.0403 at 29× cost); X02 **REFUTED** (full re-cluster remaps 0.129% of records); EFF-02 **UNEXPLAINED**. |

## Data: what downloads, and what doesn't

The repo ships **download scripts and checksums, never data**; `data/` and `artifacts/` are
gitignored. Full terms per source: [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md).

| Acquisition | Sources | How |
|---|---|---|
| Live at smoke (fetched by script, in-container) | `splink_datasets` (`fake_1000`, `historical_50k` — the quickstart corpus), NC voter snapshots (`dl.ncsbe.gov`), ONC patient-matching, Gecko frequency tables, nickname lexicons, pseudopeople's bundled 10k sample | Loaders download on first use, verify checksums where the upstream is versioned, and print a one-line provenance/terms note. |
| User-side (hosts blocked from the build container) | BPID (Zenodo) → `data/raw/bpid/`; Ohio voter file → `data/raw/ohio/` | You download; the loaders are checksum-gated before use. |
| Deferred | pseudopeople's gated 1M/330M populations (per-project IHME data-access request) | Optional-later by sign-off decision — nothing in the lab depends on them. |

NC- and Ohio-derived artifacts are aggregates-only, and stay out of the repo entirely until a
written redistribution check clears — notebooks regenerate them locally instead.

## The honesty rails

Everything above is a claim; this is the machinery that keeps claims honest, visible to any
skeptic in the committed notebooks themselves:

- **Immutable conjecture cards.** Every claim is pre-registered (pressure → measurable property →
  system metric, with a directional prediction), rendered and content-hash-frozen *before* the
  results exist. Re-rendering an edited card raises an error by design — tamper-evidence, not
  convention. Verdicts are recomputed from each card's pre-registered decision rule, never
  hand-written.
- **Figures only from registered artifacts.** No figure renders from an in-memory frame. Data is
  registered first (name, tier, config hash, git revision, seed); `er_lab.reporting.figures` loads
  it back and stamps every plot with its provenance. Anything beyond measured rungs renders
  dashed, shaded, and watermarked **EXTRAPOLATED**.
- **RUN-IN-TARGET placards are real code.** A `# [RUN-IN-TARGET mac|node]` cell is the same cell
  at `tier=target` — one code path, no forked logic. At smoke it prints a placard saying what it
  computes and where, instead of pretending.
- **An append-only registry that never mixes tiers.** The runner hard-fails on missing upstream
  artifacts and refuses smoke/target mixing (DAG `requires` in `er_lab.infra.runner`).
- **Detectability bars on every single-seed delta.** MET-04's two bars (single-seed 0.0862,
  residual-inclusive 0.0865 B³F1) are quoted wherever a single-seed margin is discussed — a paired
  CI can exclude zero while the effect still sits inside replicate noise, and the notebooks say so.
- **Raw-cosine ranking convention.** AUC and ranking metrics score the raw cosine, never
  tie-collapsed calibrated probabilities (a Wave-4 review measured isotonic's grid distorting AUC
  by 0.019); rules enter ranking arms rank-natively, and each artifact's meta declares the scored
  representation.
- **The audit is a measurement too.** [`tools/honesty_audit.py`](tools/honesty_audit.py) (and its
  in-series precursor, notebook 17 §8) scans every executed notebook per cell: PNG outputs vs
  provenance stamps, placards, cards vs verdict boxes, error outputs. Measured series totals
  (the tool, on the committed post-execution notebooks — matching the newest registered
  `honesty_audit` run row-for-row): **73/73 figures provenance-stamped, 31 RUN-IN-TARGET
  placards, 37 card renders, 40 verdict boxes (0 without a registered card; notebook 17
  re-renders the three headline refutations beside its own verdict), 0 error outputs across all
  20 executed notebooks**. The in-series run's first-execution totals (69/69, 36 cards = 36
  boxes) differ only in notebook 17's own flagged pre-execution self-scan row.
- **Wall-clock claims cite `smoke_check` or nothing.** Including the quickstart number in this
  README.

## Your data, your schema

Notebook 18 is the walkthrough: pointing the lab at a new corpus (ONC patient matching — new
schema, Excel-serial DOBs, an SSN-format field, no answer key) took **20 lines of configuration
and 0 package edits** — one `DeclaredSchema` YAML plus two dotlist overrides. It also shows the
honest catch: the invariance battery, used as an acceptance test, **failed** (5 of 8 scoreable
slices passed on the starved pseudo-label pool). That is the chapter's lesson — the port is cheap,
and the battery is what tells you whether the ported pipeline actually earned trust on your data
before you believe a single score. Start from [`configs/schemas/`](configs/schemas/) and
[`notebooks/18_your_data_your_schema.ipynb`](notebooks/18_your_data_your_schema.ipynb).

## Layout

```
notebooks/          00–18 + A, executed at smoke tier (notebooks/src/ holds the py:percent sources)
src/er_lab/         the thin package: config, infra (device/artifacts/runner), data, noise,
                    serialize, models, train, blocking, score, cluster, eval, probes, scale,
                    label, reporting
configs/            default.yaml + declared schemas (column→role YAML per corpus)
tools/              honesty_audit.py, gen_run_in_target.py
tests/              the package test suite (uv run pytest)
notes/              lit_review.md, COMPAT.md, RUN_IN_TARGET.md, BUILD_STATE.md
data/, artifacts/   gitignored — populated by scripts and runs, never committed
```

Where to read next: [`PLAN.md`](PLAN.md) — the pre-registered plan, experiment matrix, and
statistics protocol; [`notes/lit_review.md`](notes/lit_review.md) — the literature grounding and
novelty positioning; [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md) — per-source terms;
[`notes/COMPAT.md`](notes/COMPAT.md) — platform fallbacks and the pretrained-weight go/no-go tree;
[`notes/RUN_IN_TARGET.md`](notes/RUN_IN_TARGET.md) — the manifest of every heavy cell;
[`notes/BUILD_STATE.md`](notes/BUILD_STATE.md) — the build's own audit trail, including the full
verdict ledger.
