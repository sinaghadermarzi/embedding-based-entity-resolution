"""Notebook launcher CLI:  ``uv run python -m er_lab.run nb=05 run.tier=smoke a.b=c``.

Every argument is a ``key=value`` token. The special key ``nb`` selects the
notebook (number or name; ``nb=all`` runs the whole series in DAG order);
every other token is merged into the config as an OmegaConf dotlist override.
"""

from __future__ import annotations

import sys

from omegaconf import OmegaConf

from er_lab.config import DEFAULT_YAML, LabConfig, load_config
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


def _warn_unknown_sections(dotlist: list[str]) -> None:
    """Loud stderr warning for dotlist tokens whose top-level section nothing declares.

    The token still merges — extension keys are part of the contract — but a
    typo'd section name ('trian.batch_size=128') must not pass silently.
    """
    known = set(LabConfig.__dataclass_fields__) | set(OmegaConf.load(DEFAULT_YAML).keys())
    for token in dotlist:
        top = token.partition("=")[0].partition(".")[0]
        if top not in known:
            print(
                f"er_lab.run: WARNING: {token!r} does not touch any known config section "
                f"({', '.join(sorted(known))}); merging it as an extension key — check "
                "for a typo if you meant a built-in one.",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> None:
    nb, dotlist = parse_argv(sys.argv[1:] if argv is None else argv)
    if nb is None:
        raise SystemExit(USAGE)
    _warn_unknown_sections(dotlist)
    cfg = load_config(dotlist=dotlist)
    tier = cfg.run.tier
    artifacts_root = cfg.paths.artifacts_root
    if nb == "all":
        runner.run_all(tier, artifacts_root=artifacts_root, dotlist=dotlist)
    else:
        name = runner.resolve_notebook(nb)
        runner.run_notebook(runner.NOTEBOOKS_DIR / name, tier, artifacts_root, dotlist=dotlist)


if __name__ == "__main__":
    main()
