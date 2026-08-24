"""Notebook launcher CLI:  ``uv run python -m er_lab.run nb=05 run.tier=smoke a.b=c``.

Every argument is a ``key=value`` token. The special key ``nb`` selects the
notebook (number or name; ``nb=all`` runs the whole series in DAG order);
every other token is merged into the config as an OmegaConf dotlist override.
"""

from __future__ import annotations

import sys
from pathlib import Path

from er_lab.config import load_config
from er_lab.infra import runner

USAGE = (
    "usage: uv run python -m er_lab.run nb=<number|name|all> [key=value ...]\n"
    "  nb selects the notebook (e.g. nb=05, nb=05_calibrated_dirt_machine, nb=all);\n"
    "  all other key=value tokens are config dotlist overrides (e.g. run.tier=smoke)."
)


def parse_argv(argv: list[str]) -> tuple[str | None, list[str]]:
    """Split ``key=value`` tokens into the ``nb`` selector and the config dotlist."""
    nb: str | None = None
    dotlist: list[str] = []
    for token in argv:
        key, sep, value = token.partition("=")
        if not sep or not key:
            raise SystemExit(f"er_lab.run: expected key=value tokens, got {token!r}\n{USAGE}")
        if key == "nb":
            nb = value
        else:
            dotlist.append(token)
    return nb, dotlist


def main(argv: list[str] | None = None) -> None:
    nb, dotlist = parse_argv(sys.argv[1:] if argv is None else argv)
    if nb is None:
        raise SystemExit(USAGE)
    cfg = load_config(dotlist=dotlist)
    tier = cfg.run.tier
    artifacts_root = cfg.paths.artifacts_root
    if nb == "all":
        runner.run_all(tier, artifacts_root=artifacts_root)
    else:
        name = runner.resolve_notebook(nb)
        runner.run_notebook(Path(runner.NOTEBOOKS_DIR) / name, tier, artifacts_root)


if __name__ == "__main__":
    main()
