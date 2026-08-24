"""Conjecture cards and verdict boxes — pre-registration, mechanized (PLAN §5/§6).

Every training design decision in this lab is a pre-registered conjecture:
*this pressure should move this measurable embedding property, which should
move this system metric*. A :func:`conjecture_card` states that chain BEFORE
the affected runs execute, and registering it makes the pre-registration
tamper-evident: the card's content is hashed and stored as an artifact
(``card_<id>``), and re-rendering the same card with ANY field changed raises
:class:`CardImmutableError` — cards are never edited after first render.
Re-rendering with identical content is idempotent (the notebook can be
re-executed freely; nothing is re-registered).

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

from er_lab.infra.artifacts import ArtifactMissing, ArtifactRegistry

__all__ = ["VERDICTS", "CardImmutableError", "conjecture_card", "verdict_box"]

VERDICTS = ("CONFIRMED", "REFUTED", "UNEXPLAINED")

#: the pre-registered chain, in render order — exactly the PLAN §1 vocabulary
CARD_FIELDS = ("card_id", "conjecture", "pressure", "property", "metric", "prediction")

CARD_TIER = "analytical"  # cards are tier-independent protocol artifacts


class CardImmutableError(RuntimeError):
    """A registered conjecture card was re-rendered with changed content."""


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

    existing_tier = registry.newest_tier(name)
    if existing_tier is not None:
        prior, _ = registry.load(name, tier=existing_tier)
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
    name = f"card_{card_id}"
    tier = registry.newest_tier(name)
    if tier is None:
        raise ArtifactMissing(
            f"no conjecture card '{card_id}' is registered — render conjecture_card "
            "BEFORE the runs; a verdict cannot precede its conjecture"
        )
    card, _ = registry.load(name, tier=tier)
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
