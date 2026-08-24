# PLAN — The Person-ER Embedding Lab

> **Status: signed off 2026-08-24 — Stage B (build) in progress.** The sign-off gate ([§10](#10-sign-off-gate))
> is resolved: hardware pinned (mac = M2 Max 96GB; node = 4×A100 80GB, the hard ceiling), pseudopeople
> data-access request deferred as optional-later. The literature grounding is in
> [`notes/lit_review.md`](notes/lit_review.md). Changes to locked items now go through the §10 changelog.

## 1. What this lab is

A numbered notebook series over a thin `src/er_lab` package that studies **entity resolution (ER) of
person records under realistic noise**, for designs that must scale to 1e8–1e9 records.

**Working hypothesis:** whole-record learned embeddings are a *net-positive component* of large-scale
person ER — with deterministic rules kept wherever they demonstrably earn their place.

The lab's product is not a winning system; it is **statistically sound, intuitively demonstrated
comparisons of competing designs**, including where the hypothesis fails. Two commitments run through
everything:

1. **"A good embedding for ER" is a set of testable properties.** Every training design decision
   (loss, negative sampling, augmentation, tokenization, serialization, split design) is a
   pre-registered conjecture: *this pressure should move this measurable embedding property, which
   should move this system metric*. The mediation is checked, not asserted.
2. **Statistical soundness is demonstrated, not cited.** The metrology notebooks empirically show why
   each protocol rule exists (metric-choice rank reversals, CI under-coverage from wrong bootstrap
   units, optimism from leaky splits) before any embedding is trained.

**Deliberate scope exclusion — privacy.** This lab studies matching quality, statistics, and scale.
Privacy-preserving record linkage, embedding-inversion/membership-inference risk, and the legal
status of person embeddings are **intentionally out of scope**: experiments run on public-record and
synthetic data, the repo ships download scripts rather than data, and no claim in this lab concerns
privacy protection. The literature review surveys the privacy landscape only to mark this boundary;
`DATA_GOVERNANCE.md` remains, but strictly as dataset licensing/terms hygiene.

**Novelty positioning** (defended in `notes/lit_review.md`): every claim is a protocol/measurement
claim. We explicitly renounce novel losses, miners, index structures, clustering algorithms, parsers,
and any live 1e9 benchmark. What does not exist publicly — and what this lab builds — is the
combination: whole-record person embeddings vs a *well-tuned* Fellegi–Sunter baseline, entity-level
metrics, measured (not assumed) noise prevalence with group-correlated exposure, leakage-controlled
splits, multi-seed comparisons with valid confidence intervals, and out-of-sample-validated scaling
extrapolation.

## 2. Task shape and scale tiers

- **Primary task:** deduplication within one corpus under a *declared schema* (column→role mapping).
  Two-source linkage appears in one adaptation chapter (notebook 18): ADP-01 converts the pipeline
  to linkage mode (ONC; BPID fallback), and ADP-02 exercises that mode once at target tier as an
  NC↔OH cross-state arm.
- **Tiers and hardware (resolved at sign-off):** `smoke` (1e5–1e6 records; runs live on a
  4-CPU/15GB container, CPU-only) → `mid` (~1e6; definitive home: the lab's **mac** — M2 Max, 96GB
  unified memory, MPS) → `target` (~1e7 max live; definitive home: the lab's **node** — one machine,
  4×A100 80GB, 900GB RAM, 90 cores) → `analytical` (1e8–1e9 by validated extrapolation +
  cost/ambiguity modeling only). Policy: **mac-first** — every cell that fits the Mac runs there;
  the RUN-IN-TARGET manifest labels each heavy cell `mac` or `node`; **nothing in the lab may
  require more than the single node** — no cluster paths anywhere. A `# [RUN-IN-TARGET]` cell is
  the *same cell* with `tier=target` — one code path, no forked logic.

## 3. Experiment matrix

Tier = where the definitive run happens. EXPL = exploratory arm (cheap, clearly labeled, cuttable).

| ID | Question settled | Tier |
|---|---|---|
| MET-01 | Do pairwise vs entity-level metrics rank the same outputs differently? (incl. constructed one-bad-merge demo, labeled as constructed) | smoke |
| MET-02 | Does per-method best-F1 comparison reverse rankings vs fixed-precision / cost-explicit operating points? | smoke |
| MET-03 | Which bootstrap resampling unit (pair / record / entity / block) yields nominal-coverage CIs? | smoke |
| MET-04 | Variance decomposition (seed / noise-draw / split / eval-sampling) + power table — **the binding matrix-pruning arbiter** | mid |
| MET-05 | NCID duplicate/overlay rates with uncertainty — is NCID usable as pseudo-truth? **Feasibility gate** for all NC-truth arms | smoke→target |
| MET-06 | Unresolvable-fraction accounting (twins, collisions; NC has no full DOB) and metric conditioning on resolvable subsets | smoke |
| MET-07 | Optimism of random vs entity-disjoint vs household-disjoint vs temporal splits | mid |
| NSE-01 | Measured prevalence of each error channel from same-NCID snapshot diffs, incl. per-group exposure model | smoke→target |
| NSE-02 | Fidelity of existing generators (Febrl-lineage, Gecko, pseudopeople, BPID corruptions) vs measured distributions | smoke |
| NSE-03 | Are system rankings invariant across noise models (generic vs calibrated vs real temporal drift)? | mid |
| BAS-01 | How strong is a genuinely well-tuned Fellegi–Sunter baseline (Splink: TF-adjusted, JW-binned, OR-blocked) — measured ourselves | smoke→target |
| BAS-02 | Dense ANN vs multi-pass matchkeys vs BM25 blocking at **matched candidate budget and matched compute** | target |
| TRN-01 | Which contrastive loss family (InfoNCE / SupCon / triplet / CoSENT) wins at matched budget? | mid |
| TRN-02 | Negative mining in a dedup-dense corpus: FN-contamination measured as duplication rises; mitigations tested | mid |
| TRN-03 | Calibrated-noise vs generic vs no augmentation — and is the induced invariance property the mediator? | mid |
| TRN-04 | Fine-tuned small pretrained subword encoder vs from-scratch char/byte encoder at matched parameter/compute budget | mid |
| TRN-05 | Serialization: [COL]/[VAL] vs template vs JSON vs bare; [MISSING] token vs drop; field-order sensitivity | mid |
| TRN-06 | Nickname/variant supervision: does nickname-slice recall improve *without* raising twin/Jr-Sr false merges? | mid |
| CAL-01 | Cosine → match-probability calibration under blocking selection bias and name-frequency heterogeneity | mid |
| CLU-01 | Clustering-scheme value at each score-quality level; hub resistance; percolation onset vs theory | target |
| HYB-01 | Which deterministic rules earn their place (hard override / guard / feature) — marginal entity-F per rule; the noise-regime map | mid |
| SCL-01 | Do error-vs-n fits on ≤1e6 predict held-out 1e7 within CI? (nested entity-preserving ladder; threshold-transfer policy as explicit arm) | target |
| SCL-02 | Non-match similarity tail shape per name-frequency stratum; GPD-predicted vs observed FP at 2×/10×; hubness | mid→target |
| SCL-03 | End-to-end 1e9 cost model per design per platform (measured coefficients + cloud prices; labeled projections) | analytical |
| SCL-04 | Irreducible ambiguity budget at national scale (collision mass, twins, households, cohort effects) | analytical |
| EFF-01 | MRL truncation × {fp16, int8, binary+rescore, IVF-PQ} on **ER metrics** (not nDCG), with PCA-truncation control | mid→target |
| FAIR-01 | Under measured group-correlated exposure: net-positive on BOTH missed-match AND false-match disparity per group, at matched operating points, with demographic-label-error robustness bands | mid→target |
| ADP-01 | Adapt the full pipeline to a new declared schema (ONC; BPID fallback) + dedup→linkage conversion; lines-of-config-changed metric | mid |
| ADP-02 | The linkage mode at target tier: NC↔OH cross-state linkage with the ADP-01-ported pipeline; full-DOB (OH) vs no-DOB (NC) identifiability contrast in the linked setting | target |
| PRS-01 · EXPL | Does upstream parsing/standardization level change the embeddings-vs-rules ranking? 2×2 factorial | mid |
| EFF-02 · EXPL | Static-distilled encoder (~100× cheaper inference): tolerable person-ER blocking quality? | mid |
| X01 · EXPL | Marginal value per dollar of a small cross-encoder reranker over embedding retrieval | mid |
| X02 · EXPL | Entity-ID churn under incremental snapshot updates | mid |

**Pruning:** MET-04's power table is binding. Protected-arm priority when compute forces cuts:
TRN-02 > TRN-03 chain > TRN-04 > TRN-01 > TRN-05/06. Factor sets are locked in §3.1; any
fractionalization follows the pre-registered rule in §10.1, with the chosen alias table rendered in
the notebook *before* the affected runs execute.

### 3.1 Locked factor sets (full factorial unless the §10.1 rule triggers)

- **TRN-01:** loss {InfoNCE, SupCon, triplet, CoSENT} × encoder regime {pretrained-subword,
  scratch-char/byte} — 8 cells.
- **TRN-02:** miner {in-batch, BM25-mined, ANN-mined, ANN + cluster-aware FN filter} × residual
  duplication rate {low, medium, high} — 12 cells, run at the pretrained-subword regime
  (scratch-char spot-checked at the winning miner).
- **TRN-03:** augmentation {none, generic typo, calibrated channels} × encoder regime — 6 cells,
  plus a dose–response dial on calibrated-augmentation rate within the winning cell (for the
  mediation analysis, §5).
- **TRN-04:** carried as the encoder-regime factor inside TRN-01/03, plus one dedicated
  matched-budget head-to-head at each regime's own winning training recipe — 2 cells.
- **TRN-05:** serialization {[COL]/[VAL], template, JSON, bare} × missing-field treatment
  {[MISSING] token, drop} — 8 cells; field-order sensitivity is an evaluation-time probe, not a
  training factor.
- **TRN-06:** nickname supervision {off, lexicon pairs} — 2 cells, scored on the nickname slice AND
  the twin/Jr-Sr must-not-merge slice.
- **Cross-pressure interaction study** (the only anticipated fractionalization site): each of
  TRN-01/02/03/05 contrasted at two levels — its winning level vs its pre-registered default level —
  in a 2⁴ factorial at mid tier; if MET-04's power table rules the full factorial unaffordable, a
  resolution-IV 2⁴⁻¹ half fraction per §10.1, whose alias table must name which two-factor
  interactions the I=ABCD-style generator sacrifices.

Every cell runs with ≥3–5 seeds × noise draws per MET-04; all cells share replicate sets for paired
comparison.

## 4. Data sources and roles

The repo ships **download scripts + checksums, never data**. Per-source terms live in
`DATA_GOVERNANCE.md` (created in Stage B).

| Source | Role | Weight & caveats |
|---|---|---|
| **NC voter snapshots** (`dl.ncsbe.gov`, 73 snapshots 2005–2026, statewide NCID key) | Real-noise audit substrate (NSE-01); real temporal-drift eval pairs; NCID-pseudo-truth arms *if* MET-05 gate passes | One weighted source, not dominant. No full DOB (identifiability hit, MET-06). NCID circularity audited before use. Public records under NC law; aggregates-only in published artifacts pending governance check |
| **Own corruption generator** (wraps Gecko + household/hub/raw-form channels) | Primary controlled-experiment substrate; prevalence calibrated from NSE-01; group-correlated exposure model | Perfect labels by construction; realism validated by NSE-02 acceptance criteria (collision mass, Zipf tail, block-size distribution, diff-mix divergence) |
| **BPID** (Zenodo 13932202, Apache-2.0; 1M synthetic PII profiles + 10k labeled pairs) | Independent benchmark arm; its corruption model scored in NSE-02; ADP-01 fallback target | User-side download (host blocked from build sandbox) |
| **ONC patient-matching** (1M records, full DOB+SSN) | ADP-01 primary adaptation target (different schema); full-DOB identifiability contrast | No license file → research-use only, no redistribution; answer key absent → within-lab labels only |
| **Ohio voter file** (full DOB, statewide key) | Full-DOB collision calibration (SCL-04); the ADP-02 NC↔OH linkage arm | User-side download. Florida: explicitly out (request lead time, no core dependency) |
| **pseudopeople** (10k sample bundled; 1M/330M IHME-gated) | Noise-channel audit in NSE-02 via the bundled 10k sample; the gated 1M/330M populations are deferred as optional-later (sign-off decision — nothing depends on them) | Column noise independent of attributes — structurally cannot express group-correlated exposure; used accordingly |
| **splink_datasets** (`fake_1000`, `historical_50k`) | Notebook-00 quickstart corpus; CI fixtures | Fetchable live from the build sandbox |
| **Nickname lexicons** (carltonnorthern, diminutives.db) | TRN-06 supervision; battery probes | English-centric with documented provenance bias — limitation carried on every affected claim |

## 5. Metrics and statistics protocol

- **Primary metrics:** entity-level — B-cubed P/R/F, generalized merge distance, variation of
  information, cluster-level F. Pairwise P/R secondary. Blocking: pair-completeness & reduction ratio
  **with incomplete-ground-truth bias bounds**.
- **Operating points:** fixed entity-precision {0.99, 0.995} + cost grid {1:1, 1:10, 1:100 FP:FN} as
  the lab-wide primary; **fixed-FP-budget-per-record** as mandatory secondary at scale tiers
  (SCL-01/02, SCL-03); the mapping between the two derived once in the metrology notebook. Never
  per-method best-F1 (MET-02 demonstrates why). Smoke fallback: when a fixed precision is
  unattainable on small eval sets, report at the highest attainable precision with CI — never
  silently switch protocol.
- **Uncertainty:** bootstrap unit chosen by the MET-03 coverage experiment (entity/block expected),
  BCa intervals; ≥3–5 seeds × noise draws per cell, sized by MET-04's power table; paired
  comparisons on shared replicates.
- **Splits:** entity-disjoint and household-disjoint and temporal, per MET-07; no entity appears on
  both sides of any train/eval boundary.
- **Truth handling:** NCID corrections per MET-05 (with human-adjudicated disagreement samples via
  the labeling widget); identifiability conditioning per MET-06 — every headline figure reports the
  unresolvable fraction.
- **Conjecture cards:** rendered before each training run, never edited after; verdict boxes
  (CONFIRMED / REFUTED / UNEXPLAINED).
- **Adoption gate vs mediation claim (two distinct tests).** The *adoption gate* is deliberately
  conservative: a pressure is adopted only if the targeted embedding property AND the system metric
  both move with CIs excluding zero. This is necessary evidence, and *not* evidence of mediation —
  both can move for independent reasons. The *mediation claim* is tested separately, across
  arms × seeds × noise draws: indirect-effect estimation for pressure → property → metric
  (product-of-paths a·b with entity-bootstrap CIs), a conditioning check (does the pressure→metric
  effect shrink when controlling for the property), and dose–response where the pressure has a dial
  (e.g., augmentation rate). A pressure can be adopted with mediation UNEXPLAINED — the verdict box
  then says exactly that; the two marginal effects are never reported as mediation.

## 6. Notebook series (~18 + appendix)

Every notebook: conjecture cards up front → smoke cells live → `# [RUN-IN-TARGET]` for heavy cells →
verdict boxes at the end. Figures render only from registered artifacts.

| # | Title | Delivers |
|---|---|---|
| 00 | The Hardest Easy Problem | Hook; 5-minute end-to-end on `historical_50k`; `smoke_check` throughput micro-benchmark (all wall-clock claims re-render from it) |
| 01 | Corpora and Declared Schemas | DeclaredSchema (column→role YAML); acquisition scripts; governance notes |
| 02 | How to Tell Who Won | MET-01/02/03 — the metrology the whole lab runs on |
| 03 | Is the Truth True? | MET-05 NCID audit + labeling protocol; MET-06 identifiability; graceful-degradation decision recorded |
| 04 | Auditing Real Noise | NSE-01 same-NCID diff queries; channel taxonomy; per-group exposure fits; audit-applicability contract (stable key + ≥2 temporal versions) |
| 05 | The Calibrated Dirt Machine | Generator + NSE-02 fidelity scoring + acceptance criteria; MET-07 split construction |
| 06 | The Baseline That Must Be Beaten | BAS-01: well-tuned Splink FS, measured with CIs |
| 07 | The Chain-Merge Catastrophe | Experiential case for entity-level metrics and clustering care |
| 08 | A Person as a Vector | First embeddings; TRN-05 serialization; invariance battery + geometry panel debut |
| 09 | Pressures I: Losses and Negatives | TRN-01/02 with MET-04 variance/power machinery; the FN-contamination trap |
| 10 | Pressures II: Typos — Tokenizer or Augmentation? | TRN-03/04: 2×2 {pretrained-subword, scratch-char} × {augmentation, none} |
| 11 | Pressures III: Nicknames, Missing Fields, and the Circular Teacher | TRN-06; TRN-05's missing-field factor; label-provenance circularity (FS-links vs NCID as teacher) |
| 12 | Finding Candidates in Ten Million | BAS-02 sparse-vs-dense head-to-head at matched budgets |
| 13 | From Pairs to People | CAL-01 calibration; CLU-01 clustering; percolation onset |
| 14 | Where Rules Earn Their Place | HYB-01 noise-regime map with significance shading; NSE-03 ranking invariance across noise models; PRS-01 parsing factorial |
| 15 | The Last Two Orders of Magnitude | SCL-01/02 ladder + EVT/hubness; EFF-01; SCL-03 cost model; SCL-04 ambiguity budget; MEASURED-vs-EXTRAPOLATED visual convention |
| 16 | Unequal Noise, Unequal Errors | FAIR-01 both-sides disparity; exposure-vs-mechanism decomposition; label-error robustness bands |
| 17 | The Verdict at 1e7 | Headline hybrid-vs-FS-vs-pure comparison under full protocol; honesty audit |
| 18 | Your Data, Your Schema | ADP-01 adaptation walkthrough (ONC), audit re-calibration, battery as acceptance test; linkage mode incl. the ADP-02 NC↔OH target-tier arm |
| A | Appendix: Exploratory Arms | X01 / X02 / EFF-02; negative results; all labeled exploratory |

## 7. Package and infrastructure

Thin `src/er_lab`: `config` (OmegaConf structured configs with **dotlist overrides** — encoder
model, batch size, and numeric precision are user-facing keys, e.g. `model.name=…`
`train.batch_size=…` `infra.precision=bf16`; config-hash, seed lists) · `infra/device` (MPS/CUDA/CPU
detection; per-backend precision policy supplies defaults the dotlist overrides; opt-in
accelerate/torchrun) · `infra/artifacts` (hashed run registry — figures render only from registered
artifacts) · `infra/runner` (headless notebook DAG, hard-fail on missing upstream artifacts, no
smoke/target mixing) · `data/{schema,nc,loaders}` · `noise/{channels,exposure,audit}` · `serialize` ·
`models/encoders` (sentence-transformers wrapper + matched-budget char/byte transformer) ·
`train/{losses,mining,augment,loop}` · `blocking/{matchkeys,sparse,ann}` · `score/{fs,calibrate}`
(Splink-DuckDB adapter) · `cluster/{schemes,graph_qa}` ·
`eval/{metrics,bootstrap,operating_points,power,truth_model,identifiability}` ·
`probes/{battery,geometry}` · `scale/{curves,evt,costmodel}` · `label/widget` (in-notebook
adjudication UI + inter-rater stats) · `reporting/figures`.

Honesty rails: single code path across tiers; never-fabricate (no figure from placeholder data;
wall-clock claims only from `smoke_check` measurements; the README quickstart time is itself asserted
in CI); cross-backend reproducibility policy with stated per-metric tolerances (CPU↔MPS↔CUDA);
`uv`-resolved latest-stable dependencies with fallbacks recorded in `notes/COMPAT.md`.

## 8. Compute estimates (planning numbers, ±2×; re-measured by `smoke_check`)

| Tier / machine | What runs | Estimate |
|---|---|---|
| `smoke` — 4-CPU/15GB container | Whole series at 1e5–1e6; NC 2–3 snapshots processed serially with intermediate deletion | Full pass ≤ ~6–8 h; char-model runs 30–90 min |
| `mac` — M2 Max, 96GB unified (MPS) | Everything that fits, per the mac-first policy: all single-configuration mid runs, probes/battery, CAL-01, CLU-01, index builds at 1e7 (HNSW 1e7×384d well within 96GB), scaling-ladder inference to 1e7 (~1–1.5 h), EVT fits, exploratory arms, single-seed factorial previews | MiniLM-class fine-tune ~1–3 h/run; MPS ≈ 0.2–0.3× A100, so full multi-seed factorials are possible but slow — previews here, definitive matrices on the node |
| `node` — 4×A100 80GB, 900GB RAM, 90 cores (**the hard ceiling**) | Definitive multi-seed training factorials (TRN-01..06 + cross-pressure), BAS-02 matched-compute frontier at 1e7, FAIR-01 and SCL-01 definitive runs, optional 1e8 embed stretch | Factorials ~1–2 days wall via accelerate sweeps (~130–170 A100-h pre-pruning; MET-04 prunes); 1e7 encode in minutes sharded; nothing in the lab may require more than this one node |
| `analytical` | SCL-03/04, EVT extrapolations, power tables | CPU, minutes — fully live at smoke |

1e9 anchors for SCL-03 (labeled projections): 1e9 × 384-d vectors = 1.5 TB fp32 / 384 GB int8 /
48 GB binary — why EFF-01 is decisive; encode ≈ 14 GPU-h per model at 20k rec/s/GPU.

## 9. Scope decisions (the guards)

**In:** dedup primary + one linkage chapter (ADP-01 conversion, ADP-02 target arm); all four noise axes (typos/OCR; nicknames & cultural name variants; missing/swapped/dirty fields; temporal drift & household confusables) with measured prevalence; both model
regimes head-to-head; sparse-vs-dense adjudication; fairness as measurement-with-robustness-bands
(deliberately not the headline); US/English resources with the limitation stated on every affected
claim.

**Out (explicit):** privacy constraints in their entirety — privacy-preserving record linkage,
embedding leakage/inversion analysis, DP training, and the legal status of embeddings — per the
deliberate exclusion in §1; novel algorithms of any kind; live 1e9 runs; the 1e8 NC assembly as a
*promised* deliverable (feasibility-gated at MET-05; a failed gate is a documented finding and NC
demotes to noise-audit substrate); non-Latin scripts beyond lexicon coverage (stated future work);
LLM pair-matchers on real person data (sending it to external APIs conflicts with dataset terms; the
small local reranker X01 is the only reranking arm); Florida data; production serving concerns
beyond the X02 ID-churn probe.

## 10. Sign-off gate

Approving this plan locks:

1. The experiment matrix (§3) with its protected-arm priority and the §3.1 factor sets.
   **Fractionalization rule (pre-registered):** a factor set may be fractionated only if MET-04's
   power table shows the full factorial cannot reach the required power within the compute budget;
   any fraction must be resolution ≥ IV, must leave all main effects and the two named interactions
   of interest (loss × encoder-regime, augmentation × encoder-regime) unaliased, and its alias table
   is rendered in the notebook before the affected runs execute — never chosen after results are
   seen. The two named interactions are binding where their factors co-occur (TRN-01/TRN-03, which
   carry the encoder-regime factor); for the cross-pressure 2⁴⁻¹ (§3.1), whose factors exclude
   encoder regime, the alias table must instead name which two-factor interactions the chosen
   generator sacrifices.
2. Operating-point defaults: entity-precision {0.99, 0.995} + cost grid {1:1, 1:10, 1:100};
   fixed-FP-budget secondary at scale tiers.
3. Data commitments: NC (audit + gated truth arms), own generator, BPID (user downloads), ONC
   (adaptation), Ohio (user downloads; SCL-04 + ADP-02). Florida out.
4. Adjudication: user labels ~200–500 stratified pairs via the shipped widget at target tier.
5. ~~Open decision~~ **Resolved at sign-off:** the pseudopeople IHME data-access request is
   deferred — optional-later, along with anything else requiring a data-access request; the lab
   depends on none of it (bundled 10k sample only, for NSE-02).
6. ~~Open~~ **Resolved at sign-off:** hardware pinned — `mac` = M2 Max 96GB (primary, mac-first),
   `node` = one 4×A100 80GB machine (900GB RAM, 90 cores), the hard ceiling nothing may exceed;
   1e8–1e9 stay analytical.

**Changing any locked item after Stage B starts** triggers a re-plan of the affected arms only, with
the change and its cost recorded in PLAN.md's changelog.

## 11. Stage B outline (after sign-off)

1. `uv` project scaffold; hour-one egress verification (PyPI, raw.githubusercontent, HF weights);
   COMPAT fallbacks recorded; go/no-go tree for pretrained-weight acquisition.
2. Package modules in dependency order, built by parallel workflow agents with an adversarial review
   pass (statistical soundness, honesty rails, MPS/CUDA/CPU paths) before each commit.
3. Notebooks 00→18 in order, each executed headlessly at smoke tier via the runner DAG before the
   next begins; RUN-IN-TARGET manifest maintained.
4. README with measured quickstart; final full-series headless smoke pass in a fresh environment +
   honesty audit; commit and push.

## Open questions (tracked, non-blocking)

- NC demographic-field consistency across 20 years of snapshot layouts — verified empirically in
  notebooks 01/04; stratification power depends on it.
- Written governance check on redistributing NC-derived aggregate statistics before any such artifact
  ships (default: aggregates only, and only after the check).
