import polars as pl
from polars.testing import assert_frame_equal


def filter_before_expensive_agg(lf: pl.LazyFrame, min_year: int) -> pl.DataFrame:
    """Считает среднюю сумму по пользователям начиная с min_year"""
    return (
        lf.filter(pl.col("year") >= min_year)
        .group_by("user_id")
        .agg(pl.col("amount").mean().alias("avg_amount"))
    ).collect()


def test_filter_before_agg_result_and_plan() -> None:
    lf = pl.DataFrame(
        {
            "year": [2022, 2024, 2025, 2025],
            "user_id": [1, 1, 1, 2],
            "amount": [1000.0, 200.0, 400.0, 500.0],
        }
    ).lazy()

    plan = (
        lf.filter(pl.col("year") >= 2024)
        .group_by("user_id")
        .agg(pl.col("amount").mean())
        .explain(optimized=True)
    )
    assert "FILTER" in plan.upper()

    actual = filter_before_expensive_agg(lf, min_year=2024)
    expected = pl.DataFrame(
        {
            "user_id": [1, 2],
            "avg_amount": [300.0, 500.0],
        }
    ).sort("user_id")

    assert_frame_equal(actual.sort("user_id"), expected)
