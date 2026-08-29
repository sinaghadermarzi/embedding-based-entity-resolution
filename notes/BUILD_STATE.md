# BUILD_STATE — the durable map of where this build stands

*Purpose: survive session compaction/restarts. If you are resuming work on this branch, read this
file, PLAN.md, and the newest wave-review output listed below — that is the whole map.*
*Last updated: 2026-08-29, at Wave 5 completion. B-9 finalization is the only remaining phase.*

## Where we are

| Phase | Status | Evidence |
|---|---|---|
| Stage A (lit review + PLAN) | DONE | `notes/lit_review.md`, `PLAN.md` (§10 gate resolved), commits `3ad8005..c181e18` |
| Package (`src/er_lab`, 8 layers + tests) | DONE — 479 tests, 1 designed skip | commits through `ce8bf4a`; adversarial reviews applied per layer |
| Notebooks Wave 1 (00–03) | DONE, reviewed, re-executed | `d1de833` |
| Notebooks Wave 2 (04–07) | DONE, reviewed, re-executed | `526795a`, `f82aba2` |
| Notebooks Wave 3 (08–11) | DONE, reviewed, re-executed | `37648fc` (final; earlier WIP snapshots superseded) |
| Notebooks Wave 4 (12–15) | DONE, reviewed (6 findings), fixed, re-executed | wave-final commit (supersedes WIP `ae31297..40a3a78`); review output `/tmp/claude-0/.../tasks/w916txy9t.output`, fixer report `.../tasks/ab2d379222ae211c5.output` |
| Notebooks Wave 5 (16, 17, 18, A) | DONE, reviewed (5 findings, all applied), re-executed | wave-final commit (supersedes WIP `f551816`/`8af9456`); reviewer's 13-check pass incl. independent audit recount; NB17 executed twice (card idempotent) |
| B-9 finalization | IN PROGRESS | README (measured quickstart), `tools/honesty_audit.py` (carry NB17's per-cell PNG-vs-stamp counting), `notes/RUN_IN_TARGET.md` manifest, fresh-venv full-series smoke pass |

## Process pattern (every wave)

Builder agents (one notebook each, py:percent source in `notebooks/src/` → jupytext → headless
execution via `uv run python -m er_lab.run nb=NN run.tier=smoke` until green) → adversarial
reviewer → fixer agent applies all findings and re-executes → selective commit. WIP snapshot
commits mid-wave are normal; the wave-final commit supersedes them.

## Conventions that MUST NOT be violated (established, enforced by review)

- **Conjecture cards are immutable** once registered (`artifacts/card_*`): never edit
  `conjecture_card(...)` field text in sources; re-renders must be idempotent. Verdict outcomes are
  recomputed honestly from each card's pre-registered decision rule.
- **TEXT_ROLES excludes `full_name`** (NB08's measured stale-sync leakage decision) — inherited by
  every later training notebook with a one-line rationale.
- **Figures only via `er_lab.reporting.figures`** (artifact-fed, caption-stamped); no raw plt
  figures; EXTRAPOLATED labeling/watermark for anything beyond measured rungs.
- **Masked display** for any real NC-derived person fields (initials, truncated ids, padding-shape
  streets) per `DATA_GOVERNANCE.md`.
- **Entity-disjoint eval** via `met07_splits`; budget-matched arms assert identical optimizer-step
  counts and (since Wave 3) identical-init param hashes.
- **Detectability bars**: quote both `detect_bar_single_seed` and `detect_bar_residual_inclusive`
  from `met04_power_table` meta in any single-seed ranking discussion.
- **Ranking metrics (AUC etc.) score the raw cosine**, never tie-collapsed calibrated
  probabilities (Wave-4 review: isotonic's 83-step grid distorted AUC by 0.019; AUC is invariant
  only under *strictly* monotone maps). Rules enter AUC arms rank-natively
  (`apply_rules_ranknative` pattern in NB14); entity operating points keep calibrated probs.
  Declare the scored representation in the artifact meta.
- `# [RUN-IN-TARGET mac|node]` cells are real tier-gated code printing placards at smoke.
- Registry is append-only; smoke/target artifacts never mix (runner gates on DAG `requires`).

## Verdict ledger (honest outcomes so far — cards' pre-registered rules)

CONFIRMED: NB00-quickstart, MET-01, MET-02, MET-03, NSE-01, NSE-02, MET-07 (null-instrument),
TRN-05 (single-seed scope), MET-04 (post full_name-exclusion: seed sd 0.0305, 73 seeds for a 0.01
delta), LABEL-PROV (+0.074 sign-stable), CLU-01 (scheme spread 0.064 < bar), SCL-01 (≤1e6 fits
predict 1e7 within CI), PRS-01 (parsing flips nothing but like-for-like DnD +0.1606
[+0.1557, +0.1653] on raw cosine; sign survives the leak-free 35,181-pair subset), ADP-02
(dedup→linkage = one cross-source filter; override marginal +0.0045 sign-stable on real NCID
truth), **VERDICT-1E7 — the series verdict: the working hypothesis survives its own protocol at
smoke scale (P1 embeddings add value, P2 the earned override still earns, P3 the hybrid composes;
all paired deltas sign-stable at precision@0.99)**.
REFUTED (the refutation is the finding): NB01-NC-ACQUISITION (dup-NCID 3.2–3.7%), BAS-01 (tuned FS
F 0.78 < predicted 0.90), NB07-CHAIN-MERGE (percolation denied by blocking + TF; fusions are small
dense household clusters), BAS-02 (BM25 ≥ dense at every matched budget; gap +0.279 recall@k=25),
HYB-01 (dob-year guard HURTS −0.0190 B³F1 sign-stable, 28.7% of guard fires are true pairs;
override +0.0247 sign-stable positive — rules earn their place individually, not as a class),
X01 (scratch cross-encoder rerank HURTS: ΔPC@10 −0.0403 sign-stable at 29× cost; oracle ceiling
registered as instrument), X02 (churn-is-imaginary: full re-cluster remaps 0.129% of records —
both exploratory).
UNEXPLAINED: NB08-BATTERY, TRN-01, TRN-02, TRN-03, TRN-04 (partial: pretrained side is
RUN-IN-TARGET mac), TRN-06, TRN-05-MISSING-STRESS, CAL-01, NSE-03 (raw-cosine scored: no
sign-stable ranking reversal across noise models; NC-real flips to hyb>fs>emb but adjacent deltas
not sign-stable), SCL-02, FAIR-01 (embedding helps both groups absolutely but its win tilts to
the low-exposure group +0.0112 — below both bars; membership SIMULATED, multipliers measured),
ADP-01 (20 config lines, 0 package edits, but battery acceptance FAILS 5/8 slices on the starved
pseudo-label pool — the chapter's lesson), EFF-02 (exploratory).
Series totals (NB17 honesty audit, measured): 69/69 figures provenance-stamped, 31 RUN-IN-TARGET
placards, 36 cards = 36 verdict boxes, 0 error outputs across all 20 executed notebooks.

## Key facts a resumed session needs

- Branch `claude/person-entity-resolution-lab-qp2cps`, PR #1. Commit footer convention: see any
  recent commit. Never push to other branches.
- Smoke data cache: `data/raw/nc/` (2 snapshot zips + parsed parquets, idempotent re-runs),
  `data/processed/` canonical + aligned parquets, `data/splink_datasets/`. All gitignored.
- NC = 2 snapshots (20240101, 20260101), counties [32, 68] parsed, canonical smoke substrate is
  county 68 (Orange); aligned pair = 391,257 rows.
- Calibrated corpus = 32,259 records from historical_50k base (18,010 base + dups + 649 household
  confusables); truth = entity_id; ops log in `calibrated_corpus_ops`.
- HF is blocked in this container: scratch char encoder is the smoke default; pretrained arms are
  RUN-IN-TARGET(mac) (see `notes/COMPAT.md` go/no-go tree).
- Wave 5 DAG contracts (from `runner.py`): 16 requires calibrated_corpus + nse01_exposure_model +
  cal01_calibration_map → produces fair01_disparity_panel; 17 requires bas01_fs_baseline +
  hyb01_rule_value_map + clu01_clustering_scores + scl01_scaling_fits → produces verdict_1e7 +
  honesty_audit; 18 requires declared_schemas + calibrated_corpus → produces
  adp01_adaptation_report + adp02_linkage_results; A requires calibrated_corpus → produces
  x01_reranker_value + x02_id_churn + eff02_static_distill. 16/18/A are mutually independent
  (build in parallel); 17 LAST (it consumes Wave-4 artifacts and writes the series verdict).
- Session workflow scripts (for resume-with-cache):
  `~/.claude/projects/-home-user-embedding-based-entity-resolution/<session>/workflows/scripts/`.

## Hardware + tiers (locked at sign-off)

smoke = this 4-CPU/15GB container (whole series live, 1e5–1e6) · mac = M2 Max 96GB MPS (mac-first
policy) · node = ONE 4×A100 80GB / 900GB RAM / 90 cores machine — the hard ceiling; nothing may
exceed it · analytical = 1e8–1e9 extrapolation only. Data-request-gated sources (pseudopeople
1M/330M) deferred; privacy is intentionally out of scope (PLAN §1).
