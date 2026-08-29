"""Conjecture cards and verdict boxes — pre-registration, mechanized (PLAN §5/§6).

Every training design decision in this lab is a pre-registered conjecture:
*this pressure should move this measurable embedding property, which should
move this system metric*. A :func:`conjecture_card` states that chain BEFORE
the affected runs execute, and registering it makes the pre-registration
tamper-evident: the card's content is hashed and stored as an artifact
(``card_<id>``), and re-rendering the same card with ANY field changed raises
:class:`CardImmutableError` — cards are never edited after first render.
Re-rendering with identical content is idempotent (the notebook can be
re-executed freely; nothing is re-registered). Every card LOAD re-verifies the
record: kind must be ``card``, the stored sha256 must match the recomputed
content hash, and — because the registry is append-only, so 'editing' can only
mean shadowing with a newer run — ALL completed runs of ``card_<id>`` must
agree on content, or :class:`CardImmutableError` names the tampering.

After the runs, :func:`verdict_box` closes the loop with one of the three
allowed outcomes — ``CONFIRMED`` / ``REFUTED`` / ``UNEXPLAINED`` — and refuses
to render a verdict for a card that was never registered (no post-hoc
'conjectures'). ``UNEXPLAINED`` is a first-class outcome: PLAN §5 allows
adopting a pressure whose mediation is unexplained, but the box must say so.

Cards are protocol objects, not run outputs, so they are registered once under
the fixed ``analytical`` tier and looked up tier-agnostically — a card is the
same card at smoke and at target.

Both functions RETURN the markdown (so callers/tests can inspect it) and also
display it when running under IPython.
"""

from __future__ import annotations

import hashlib
import json

from er_lab.infra.artifacts import (
    ArtifactMissing,
    ArtifactRegistry,
    _read_meta,
    _read_payload,
)

__all__ = ["VERDICTS", "CardImmutableError", "conjecture_card", "verdict_box"]

VERDICTS = ("CONFIRMED", "REFUTED", "UNEXPLAINED")

#: the pre-registered chain, in render order — exactly the PLAN §1 vocabulary
CARD_FIELDS = ("card_id", "conjecture", "pressure", "property", "metric", "prediction")

CARD_TIER = "analytical"  # cards are tier-independent protocol artifacts


class CardImmutableError(RuntimeError):
    """A registered conjecture card was re-rendered with changed content — or the
    registry's record of it shows tampering (wrong kind, hash mismatch, or two
    completed runs of the same card disagreeing on content)."""


def _load_card_verified(registry: ArtifactRegistry, card_id: str) -> dict | None:
    """Load card ``card_<card_id>`` with tamper-evidence checks; None if unregistered.

    The registry is append-only, so 'editing' a card can only mean registering
    a newer run that shadows the original. Every card load therefore
    (a) verifies each completed run has ``kind == 'card'`` and that recomputing
    :func:`_content_hash` over its :data:`CARD_FIELDS` equals its stored
    ``sha256``, and (b) enumerates ALL completed runs of the name and raises
    :class:`CardImmutableError` if any two disagree on content — a second run
    with different fields IS the tamper event. Returns the verified content.
    """
    name = f"card_{card_id}"
    run_dirs = registry._runs(name)
    if not run_dirs:
        return None
    contents: list[dict] = []
    for run_dir in run_dirs:
        meta = _read_meta(run_dir)
        if meta.get("kind") != "card":
            raise CardImmutableError(
                f"conjecture card '{card_id}' has a registered run ({run_dir.name}) with "
                f"kind={meta.get('kind')!r}, not 'card' — the card record has been tampered "
                "with; cards are only ever written by conjecture_card (PLAN §5)"
            )
        payload = _read_payload(run_dir / meta["payload"])
        fields = {f: payload.get(f) for f in CARD_FIELDS}
        if _content_hash(fields) != payload.get("sha256"):
            raise CardImmutableError(
                f"conjecture card '{card_id}' fails its integrity check: run {run_dir.name} "
                "stores a sha256 that does not match its own content — the card record has "
                "been tampered with (PLAN §5)"
            )
        contents.append(fields)
    first = contents[0]
    for fields in contents[1:]:
        changed = [f for f in CARD_FIELDS if fields[f] != first[f]]
        if changed:
            raise CardImmutableError(
                f"conjecture card '{card_id}' has {len(contents)} registered runs that "
                f"disagree on field(s) {changed} — cards are never edited after first "
                "render, so a divergent later run is itself the tamper event (PLAN §5)"
            )
    return contents[-1]


def conjecture_card(
    card_id: str,
    conjecture: str,
    pressure: str,
    property: str,  # the PLAN's own vocabulary for the mediator (shadows the builtin, on purpose)
    metric: str,
    prediction: str,
    registry: ArtifactRegistry,
) -> str:
    """Render a conjecture card and register it immutably as ``card_<card_id>``.

    Fields: *conjecture* (the claim in one sentence), *pressure* (the training
    dial being turned), *property* (the measurable embedding property it
    should move), *metric* (the system metric that should follow), and
    *prediction* (the pre-registered directional/size prediction the verdict
    will be scored against).

    First call registers ``card_<card_id>`` (payload: the fields + their
    sha256). Later calls: identical content -> idempotent no-op re-render;
    any changed field -> :class:`CardImmutableError` naming the fields that
    differ. Returns the markdown.
    """
    content = {
        "card_id": card_id,
        "conjecture": conjecture,
        "pressure": pressure,
        "property": property,
        "metric": metric,
        "prediction": prediction,
    }
    for key, val in content.items():
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"card field {key!r} must be a non-empty string")
    digest = _content_hash(content)
    name = f"card_{card_id}"

    prior = _load_card_verified(registry, card_id)
    if prior is not None:
        changed = [f for f in CARD_FIELDS if prior.get(f) != content[f]]
        if changed:
            raise CardImmutableError(
                f"conjecture card '{card_id}' is already registered and cards are never "
                f"edited after first render (PLAN §5); differing field(s): {changed}. "
                "State a genuinely new conjecture under a new card_id instead."
            )
        # identical content: idempotent — re-render without re-registering
    else:
        registry.register(
            name,
            {**content, "sha256": digest},
            cfg={"card": {"id": card_id, "sha256": digest}},
            tier=CARD_TIER,
            kind="card",
            meta={"sha256": digest},
        )

    md = _card_markdown(content, digest)
    _display(md)
    return md


def verdict_box(
    card_id: str,
    outcome: str,
    evidence: str,
    registry: ArtifactRegistry,
) -> str:
    """Render the verdict for a registered conjecture card.

    *outcome* must be one of :data:`VERDICTS` (anything else is a protocol
    violation, not a typo to be forgiven); the card must already be registered
    (:class:`~er_lab.infra.artifacts.ArtifactMissing` otherwise — a verdict
    without a pre-registered conjecture is exactly the post-hoc storytelling
    the cards exist to prevent). Returns the markdown.
    """
    if outcome not in VERDICTS:
        raise ValueError(f"unknown outcome {outcome!r}; the vocabulary is {VERDICTS} (PLAN §5)")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("evidence must be a non-empty string (name the artifacts/CIs)")
    card = _load_card_verified(registry, card_id)
    if card is None:
        raise ArtifactMissing(
            f"no conjecture card '{card_id}' is registered — render conjecture_card "
            "BEFORE the runs; a verdict cannot precede its conjecture"
        )
    md = (
        f"> ### VERDICT: {outcome} — card `{card_id}`\n"
        f">\n"
        f"> **Conjecture** {card['conjecture']}\n"
        f">\n"
        f"> **Prediction** {card['prediction']}\n"
        f">\n"
        f"> **Evidence** {evidence}\n"
    )
    _display(md)
    return md


# -- internals ---------------------------------------------------------------


def _content_hash(content: dict[str, str]) -> str:
    canonical = json.dumps(content, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _card_markdown(content: dict[str, str], digest: str) -> str:
    return (
        f"### Conjecture card `{content['card_id']}`\n\n"
        f"| | |\n|---|---|\n"
        f"| **Conjecture** | {content['conjecture']} |\n"
        f"| **Pressure (dial)** | {content['pressure']} |\n"
        f"| **Embedding property** | {content['property']} |\n"
        f"| **System metric** | {content['metric']} |\n"
        f"| **Pre-registered prediction** | {content['prediction']} |\n\n"
        f"*Immutable — sha256 `{digest[:12]}`; registered before the affected runs, "
        f"never edited (PLAN §5).*\n"
    )


def _display(md: str) -> None:
    """Show the markdown when running under IPython; silently return otherwise."""
    try:
        from IPython.core.getipython import get_ipython
        from IPython.display import Markdown, display
    except ImportError:  # headless/test environment
        return
    if get_ipython() is not None:
        display(Markdown(md))
