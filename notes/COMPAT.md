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

*(empty — populated in Stage B)*
