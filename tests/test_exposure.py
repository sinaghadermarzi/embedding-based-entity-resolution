"""er_lab.noise.exposure — group-correlated exposure model contract tests."""

from __future__ import annotations

import pandas as pd
import pytest

from er_lab.noise.exposure import ExposureModel


def race_table() -> pd.DataFrame:
    return pd.DataFrame({"race": ["A", "B"], "multiplier": [2.0, 0.5]})


def records() -> pd.DataFrame:
    # non-default index: alignment must survive it
    return pd.DataFrame(
        {"race": pd.Series(["A", "B", "C", pd.NA], index=[10, 20, 30, 40], dtype="string")}
    )


def test_uniform_returns_base_rate_everywhere() -> None:
    df = records()
    rates = ExposureModel.uniform().per_record_rates(df, 0.07)
    assert rates.index.equals(df.index)
    assert rates.tolist() == [0.07, 0.07, 0.07, 0.07]


def test_from_table_multiplier_math_and_alignment() -> None:
    model = ExposureModel.from_table(race_table())
    assert model.group_cols == ["race"]
    df = records()
    rates = model.per_record_rates(df, 0.1)
    assert rates.index.equals(df.index)
    assert rates.dtype == float
    # A doubled, B halved, unseen group C and NA group -> default multiplier 1.0
    assert rates.tolist() == pytest.approx([0.2, 0.05, 0.1, 0.1])


def test_from_table_matches_int_coded_groups_as_strings() -> None:
    model = ExposureModel.from_table(pd.DataFrame({"county": [7], "multiplier": [3.0]}))
    df = pd.DataFrame({"county": pd.Series(["7", "8"], dtype="string")})
    assert model.per_record_rates(df, 0.01).tolist() == pytest.approx([0.03, 0.01])


def test_two_group_columns() -> None:
    table = pd.DataFrame(
        {"race": ["A", "A"], "sex": ["F", "M"], "multiplier": [1.5, 0.5]}
    )
    model = ExposureModel.from_table(table)
    df = pd.DataFrame(
        {
            "race": pd.Series(["A", "A", "B", "A"], dtype="string"),
            "sex": pd.Series(["F", "M", "F", pd.NA], dtype="string"),
        }
    )
    rates = model.per_record_rates(df, 0.2)
    # (A,F)->1.5, (A,M)->0.5, (B,F) unknown->1.0, (A,NA) partial-NA key->1.0
    assert rates.tolist() == pytest.approx([0.3, 0.1, 0.2, 0.2])


def test_rates_clip_at_one() -> None:
    model = ExposureModel.from_table(race_table())
    df = pd.DataFrame({"race": pd.Series(["A", "B"], dtype="string")})
    assert model.per_record_rates(df, 0.9).tolist() == pytest.approx([1.0, 0.45])


def test_fit_from_prevalence_unweighted() -> None:
    prev = pd.DataFrame({"race": ["A", "B"], "rate": [0.2, 0.1]})
    model = ExposureModel.fit_from_prevalence(prev)
    # overall = 0.15 -> multipliers 4/3 and 2/3
    df = pd.DataFrame({"race": pd.Series(["A", "B"], dtype="string")})
    rates = model.per_record_rates(df, 0.15)
    assert rates.tolist() == pytest.approx([0.2, 0.1])


def test_fit_from_prevalence_weighted_by_n() -> None:
    prev = pd.DataFrame({"race": ["A", "B"], "rate": [0.2, 0.1], "n": [300, 100]})
    model = ExposureModel.fit_from_prevalence(prev)
    # overall = (0.2*300 + 0.1*100) / 400 = 0.175
    mult = model.multipliers.set_index("race")["multiplier"]
    assert mult["A"] == pytest.approx(0.2 / 0.175)
    assert mult["B"] == pytest.approx(0.1 / 0.175)


def test_fit_from_prevalence_drops_na_group_rows_but_counts_them_in_overall() -> None:
    prev = pd.DataFrame({"race": ["A", None], "rate": [0.2, 0.1]})
    model = ExposureModel.fit_from_prevalence(prev)
    assert len(model.multipliers) == 1  # NA group row dropped from the table
    assert model.multipliers["multiplier"].iloc[0] == pytest.approx(0.2 / 0.15)


def test_from_table_validation_errors() -> None:
    with pytest.raises(ValueError, match="multiplier"):
        ExposureModel.from_table(pd.DataFrame({"race": ["A"]}))
    with pytest.raises(ValueError, match="group column"):
        ExposureModel.from_table(pd.DataFrame({"multiplier": [1.0]}))
    with pytest.raises(ValueError, match="duplicate"):
        ExposureModel.from_table(
            pd.DataFrame({"race": ["A", "A"], "multiplier": [1.0, 2.0]})
        )
    with pytest.raises(ValueError, match="non-missing"):
        ExposureModel.from_table(pd.DataFrame({"race": [None], "multiplier": [1.0]}))
    with pytest.raises(ValueError, match=">= 0"):
        ExposureModel.from_table(pd.DataFrame({"race": ["A"], "multiplier": [-0.5]}))


def test_fit_zero_overall_raises() -> None:
    prev = pd.DataFrame({"race": ["A", "B"], "rate": [0.0, 0.0]})
    with pytest.raises(ValueError, match="zero"):
        ExposureModel.fit_from_prevalence(prev)


def test_base_rate_must_be_probability() -> None:
    model = ExposureModel.uniform()
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="probability"):
            model.per_record_rates(records(), bad)


def test_missing_group_column_raises() -> None:
    model = ExposureModel.from_table(race_table())
    with pytest.raises(KeyError, match="race"):
        model.per_record_rates(pd.DataFrame({"sex": ["F"]}), 0.1)


def test_per_record_rates_deterministic() -> None:
    model = ExposureModel.from_table(race_table())
    df = records()
    pd.testing.assert_series_equal(
        model.per_record_rates(df, 0.1), model.per_record_rates(df, 0.1)
    )
