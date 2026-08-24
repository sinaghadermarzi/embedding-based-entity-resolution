# DATA_GOVERNANCE — dataset licensing and handling

Scope note: privacy *research* is intentionally out of scope for this lab (PLAN.md §1). This file is
strictly dataset licensing/terms hygiene: what each source permits, how the repo handles it, and
what may be redistributed. The standing policy everywhere: **the repo ships download scripts and
checksums, never data**; `data/` and `artifacts/` are gitignored.

| Source | Terms (as recorded) | Lab handling |
|---|---|---|
| **NC voter registration** (`dl.ncsbe.gov` S3) | Public records under N.C.G.S. §132-1; voter registration is the official record under §163-82.10. The statutes, not the bucket ReadMe, govern permitted uses; most states restrict commercial use of voter lists. | Downloaded by script at run time; records never committed or redistributed. Published figures/artifacts show aggregates only (prevalence tables, frequency statistics). Before any NC-derived aggregate artifact ships in the repo, a written check on aggregate redistribution is required (open item, PLAN.md). |
| **BPID** (Zenodo 13932202) | Apache-2.0; synthetic PII profiles (1M) + labeled pairs (10k). | Zenodo is blocked from the build container: user downloads the archive and places it under `data/raw/bpid/`; loader is checksum-gated. Synthetic → no PII concerns; license permits redistribution but we still don't commit data. |
| **ONC patient-matching challenge** (GitHub `onc-healthit/patient-matching`) | No LICENSE file in the repo; challenge data, synthetic-ish patient records with DOB/SSN-format fields; answer key not public. | Treat as research-use only; download by script; never redistribute; within-lab labels only. |
| **Ohio voter file** (`ohiosos.gov`) | Public records; full DOB present; state-law use restrictions apply (non-commercial norms). | Host blocked from the build container: user-side download into `data/raw/ohio/`; checksum-gated loader; aggregates-only in published artifacts, same policy as NC. |
| **pseudopeople** (IHME) | Bundled 10k sample ships with the PyPI package; 1M/330M populations gated behind a per-project data-access request. | Sample used for the NSE-02 generator audit. Gated populations: deferred as optional-later per sign-off — nothing depends on them; if ever requested, their terms govern. |
| **splink_datasets** (`moj-analytical-services/splink_datasets`) | Openly downloadable (fake_1000 is synthetic; historical_50k is Wikidata-derived historical persons with injected errors). | Fetched by script at run time; used for quickstart and CI fixtures. |
| **Gecko / gecko-data** (`ul-mds`) | MIT; frequency tables from public sources. | Dependency via PyPI; frequency tables fetched by script. |
| **Nickname lexicons** (carltonnorthern `nicknames`, `diminutives.db`) | Open repos; carltonnorthern's own README documents provenance bias (African-American genealogy lists); diminutives.db is Wiktionary-derived. | Fetched by script; provenance-bias limitation carried on every claim that uses them (lit_review §12). |
| **Trained checkpoints** | — | Only checkpoints trained purely on synthetic data are candidates for release. Checkpoints trained on real voter/patient records are not released (privacy analysis of embedding artifacts is out of scope, so we take the conservative default rather than argue safety). |

Operational rules:

1. Loaders emit a one-line provenance/terms note on first load of each source (implemented in
   `er_lab.data.loaders`).
2. Every download script verifies checksums where the upstream is versioned; user-side downloads
   are checksum-gated before use.
3. `data/` and `artifacts/` are gitignored; CI asserts no file under `data/` is ever tracked.
4. Aggregate NC/Ohio-derived artifacts (noise-prevalence tables, name-frequency tables, exposure
   coefficients) stay out of the repo until the written redistribution check clears; notebooks
   regenerate them locally instead.
