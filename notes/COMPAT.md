# COMPAT — platform and dependency fallbacks

This file records every place the lab's "latest stable, resolved with `uv`" policy meets reality:
per-platform fallbacks (MPS / CUDA / CPU), blocked hosts and their workarounds, and pinned
deviations with reasons. It is populated during Stage B; entries are never deleted, only superseded.

## Target hardware (resolved at sign-off, 2026-08-24)

- **mac** (primary, mac-first policy): Apple M2 Max, 96GB unified memory — MPS backend. Everything
  that fits runs here.
- **node** (the hard ceiling): one machine with 4×A100 80GB, 900GB RAM, 90 cores. Definitive
  multi-seed factorials and 1e7-scale target runs. Nothing in the lab may require more than this
  single node; there are no cluster code paths.
- RUN-IN-TARGET cells are labeled `mac` or `node` in the manifest.

## Known constraints going in (from planning)

- **Build sandbox** (smoke tier): 4 CPUs, 15 GB RAM, no GPU, Python 3.11, `uv` 0.8.17. Egress via
  proxy: `dl.ncsbe.gov` and `raw.githubusercontent.com` verified reachable; `zenodo.org` (BPID),
  `huggingface.co` (model weights), `ohiosos.gov`, `dbs.uni-leipzig.de` blocked — acquisition for
  these is user-side with checksums, or `# [RUN-IN-TARGET]`.
- **Hour-one Stage B checks (to be recorded here with results):** PyPI resolution of core deps
  (torch CPU wheel, sentence-transformers, splink, duckdb, faiss-cpu vs hnswlib, gecko-syndata,
  pseudopeople), HF weight acquisition path, and the go/no-go tree for which pretrained-encoder arms
  run at smoke tier.
- **Cross-backend numerics:** MPS/CUDA/CPU do not produce bit-identical results. The lab's
  reproducibility policy (stated per-metric tolerances; seed-matched CPU-vs-accelerator parity test
  in notebook 00) lives here once measured.

## Fallback log

### 2026-08-24 — Stage B hour-one egress checks (build container)

| Check | Result | Consequence |
|---|---|---|
| PyPI via `uv sync` | ✓ all deps resolved (torch 2.13.0+cu130, sentence-transformers 6.0.0, splink 4.0.16, duckdb 1.5.5, faiss-cpu, hnswlib, bm25s, gecko-syndata, omegaconf 2.3.1) | — |
| `download.pytorch.org` | ✗ blocked (connect tunnel failure) | torch comes from default PyPI on all platforms: macOS wheel has MPS, Linux wheel has CUDA 12 (right for the A100 node); the CPU-only smoke container carries unused CUDA libs (~4GB) — accepted cost |
| `raw.githubusercontent.com` (splink_datasets, ONC patient-matching) | ✓ 206 | notebook-00 quickstart corpus and ADP-01 target downloadable in-container |
| `s3.amazonaws.com/dl.ncsbe.gov` (NC voter) | ✓ 206 | NC snapshot acquisition live at smoke tier |
| `huggingface.co` / HF CDN | ✗ blocked (proxy 403) | **Pretrained-weight go/no-go tree:** (1) smoke tier: the from-scratch char/byte encoder is the smoke-tier default — it trains for real on smoke data; (2) every pretrained-encoder cell is `# [RUN-IN-TARGET mac]` — the mac/node have normal internet and pull weights via sentence-transformers as usual, or from a user-set `paths.hf_local` with `local_files_only=True`; (3) unit tests may use a tiny randomly-initialized transformer for pipeline-shape checks only — never in notebook figures, never labeled "pretrained" (honesty rail). |
| `zenodo.org` (BPID) | ✗ blocked (from Stage A probing) | BPID loader is checksum-gated on a user-side download placed under `data/raw/bpid/` |

Python: container pins 3.11.15 (pseudopeople requires <3.13; `requires-python = ">=3.11,<3.13"`).
