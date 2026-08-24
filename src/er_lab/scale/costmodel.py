"""End-to-end 1e9 cost model from measured coefficients (SCL-03, PLAN §3/§8).

The lab never runs 1e9 live; SCL-03 *projects* what each design would cost,
and every projection is honest about its provenance: each stage row carries a
``basis`` of ``'MEASURED'`` (n within the range the coefficients were measured
over, so the row is arithmetic on measurements) or ``'EXTRAPOLATED'`` (n
beyond it, so linear-in-n scaling of the coefficients is itself an assumption
— the label the reporting layer turns into dashed lines and watermarks).

Coefficients come from ``smoke_check``-style micro-benchmarks (never assumed):
throughputs in records/pairs/edges per second and bytes per stored vector.
Prices are cloud list prices per hour, recorded alongside.

Single-node ceiling (PLAN §2, sign-off item 6): nothing in the lab may
require more than the one documented node — 4x A100 80GB, 900 GB RAM. A
design whose peak stage memory exceeds the node raises
:class:`SingleNodeCeilingError` instead of pricing a machine that does not
exist. This is why EFF-01 (fp16/int8/binary vectors) is decisive at 1e9:
1e9 x 384-d fp32 = 1.5 TB > 900 GB — the model *refuses* that design, and
the refusal is the finding. No multi-machine path exists here by design.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["CostModel", "SingleNodeCeilingError"]

# The documented node (PLAN §2, sign-off): the hard ceiling for every design.
NODE_RAM_GB = 900.0
NODE_GPU_COUNT = 4
NODE_GPU_RAM_GB = 80.0

#: rough in-memory footprint of one candidate edge (two int64 ids + one
#: float64 score, dict/index overhead ignored) — labeled, deliberately coarse.
BYTES_PER_EDGE = 24

REQUIRED_COEFFICIENTS = (
    "encode_rps",  # records encoded per second (per GPU)
    "index_build_s_per_M",  # seconds to build the ANN index per 1e6 vectors
    "bytes_per_vector",  # stored bytes per vector (dim x precision, incl. codes)
    "score_pairs_ps",  # candidate pairs scored per second
    "cc_edges_ps",  # clustering edges processed per second
)
REQUIRED_PRICES = ("gpu_hour_usd", "cpu_hour_usd")

MEASURED, EXTRAPOLATED = "MEASURED", "EXTRAPOLATED"


class SingleNodeCeilingError(RuntimeError):
    """The design needs more memory than the documented single node has."""


class CostModel:
    """Stage-by-stage time/memory/cost projection from measured coefficients.

    Parameters: *coefficients* must supply ``REQUIRED_COEFFICIENTS`` (all
    positive); *prices* must supply ``REQUIRED_PRICES`` (USD per hour);
    *measured_max_n* is the largest record count the coefficients were
    actually measured at (default 1e7, the target tier's live maximum) —
    the MEASURED/EXTRAPOLATED boundary.
    """

    def __init__(
        self,
        coefficients: dict[str, float],
        prices: dict[str, float],
        *,
        measured_max_n: int = 10_000_000,
    ):
        missing = [k for k in REQUIRED_COEFFICIENTS if k not in coefficients]
        if missing:
            raise KeyError(f"coefficients missing {missing}; need {list(REQUIRED_COEFFICIENTS)}")
        bad = [k for k in REQUIRED_COEFFICIENTS if not coefficients[k] > 0]
        if bad:
            raise ValueError(f"coefficients must be positive, got {bad}")
        missing_p = [k for k in REQUIRED_PRICES if k not in prices]
        if missing_p:
            raise KeyError(f"prices missing {missing_p}; need {list(REQUIRED_PRICES)}")
        self.coefficients = dict(coefficients)
        self.prices = dict(prices)
        self.measured_max_n = int(measured_max_n)

    def breakdown(self, n_records: float, design: dict) -> pd.DataFrame:
        """Per-stage projection for *n_records* under *design*.

        *design* keys: ``candidates_per_record`` (required — the blocking
        budget k; pairs = n*k), optional ``index_on_gpu`` (bool: the vector
        store must then also fit the node's combined GPU RAM), optional
        ``label`` (carried into ``DataFrame.attrs``).

        Stages and their arithmetic (all linear in their driving count —
        that linearity is exactly what EXTRAPOLATED flags as an assumption):

        - ``encode``  : n / encode_rps            (GPU-priced)
        - ``index_build``: (n/1e6) * index_build_s_per_M; memory = the full
          vector store n * bytes_per_vector       (CPU-priced)
        - ``score_pairs``: n*k / score_pairs_ps; memory = vector store
          (vectors stay resident for scoring)     (CPU-priced)
        - ``cluster`` : n*k / cc_edges_ps; memory = n*k * BYTES_PER_EDGE
                                                   (CPU-priced)

        ``memory_gb`` is each stage's peak working set (not additive across
        stages); the single-node ceiling is asserted on the max — see
        :class:`SingleNodeCeilingError`.
        """
        n = float(n_records)
        if n <= 0:
            raise ValueError(f"n_records must be positive, got {n_records}")
        if "candidates_per_record" not in design:
            raise KeyError("design needs 'candidates_per_record' (the blocking budget k)")
        k = float(design["candidates_per_record"])
        if k <= 0:
            raise ValueError(f"candidates_per_record must be positive, got {k}")
        c, p = self.coefficients, self.prices
        pairs = n * k
        vector_gb = n * c["bytes_per_vector"] / 1e9
        edge_gb = pairs * BYTES_PER_EDGE / 1e9
        basis = MEASURED if n <= self.measured_max_n else EXTRAPOLATED

        rows = [
            ("encode", n / c["encode_rps"] / 3600.0, 0.0, "gpu_hour_usd"),
            ("index_build", (n / 1e6) * c["index_build_s_per_M"] / 3600.0, vector_gb,
             "cpu_hour_usd"),
            ("score_pairs", pairs / c["score_pairs_ps"] / 3600.0, vector_gb, "cpu_hour_usd"),
            ("cluster", pairs / c["cc_edges_ps"] / 3600.0, edge_gb, "cpu_hour_usd"),
        ]
        out = pd.DataFrame(
            [
                {
                    "stage": stage,
                    "time_h": time_h,
                    "memory_gb": mem_gb,
                    "cost_usd": time_h * p[price_key],
                    "basis": basis,
                }
                for stage, time_h, mem_gb, price_key in rows
            ]
        )
        self._assert_single_node(out, vector_gb, design)
        out.attrs["design"] = dict(design)
        out.attrs["n_records"] = n
        return out

    def _assert_single_node(self, out: pd.DataFrame, vector_gb: float, design: dict) -> None:
        """Refuse any stage that outgrows the documented node — the honesty rail."""
        worst = out.loc[out["memory_gb"].idxmax()]
        if worst["memory_gb"] > NODE_RAM_GB:
            raise SingleNodeCeilingError(
                f"stage '{worst['stage']}' needs {worst['memory_gb']:.0f} GB but the node has "
                f"{NODE_RAM_GB:.0f} GB RAM (4xA100 box, PLAN §2) — no multi-machine path exists "
                "in this lab; shrink the design (bytes_per_vector via fp16/int8/binary, "
                "candidates_per_record) instead"
            )
        if design.get("index_on_gpu"):
            gpu_total = NODE_GPU_COUNT * NODE_GPU_RAM_GB
            if vector_gb > gpu_total:
                raise SingleNodeCeilingError(
                    f"index_on_gpu: the vector store needs {vector_gb:.0f} GB but the node's "
                    f"{NODE_GPU_COUNT} GPUs hold {gpu_total:.0f} GB combined — use the CPU "
                    "index path or shrink bytes_per_vector"
                )
