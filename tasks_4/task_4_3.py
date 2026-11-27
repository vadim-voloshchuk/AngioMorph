from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal


def read_eager_top_scores(csv_path: str, limit: int = 3) -> pl.DataFrame:
    """Читает CSV через read_csv и возвращает топ с score >= 80"""
    return (
        pl.read_csv(csv_path)
        .filter(pl.col("score") >= 80)
        .select("name", "score")
        .sort("score", descending=True)
        .head(limit)
    )


def read_lazy_top_scores(csv_path: str, limit: int = 3) -> pl.DataFrame:
    """Читает CSV через scan_csv и возвращает топ с score >= 80"""
    return (
        pl.scan_csv(csv_path)
        .filter(pl.col("score") >= 80)
        .select("name", "score")
        .sort("score", descending=True)
        .head(limit)
        .collect()
    )


def test_eager_and_lazy_return_identical_results(tmp_path: Path) -> None:
    csv_file = tmp_path / "scores.csv"
    df = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
            "score": [95, 72, 88, 91, 84],
            "extra_col": ["a", "b", "c", "d", "e"],
        }
    )
    df.write_csv(csv_file)

    eager_res = read_eager_top_scores(str(csv_file), limit=2)
    lazy_res = read_lazy_top_scores(str(csv_file), limit=2)

    expected = pl.DataFrame(
        {
            "name": ["Alice", "Diana"],
            "score": [95, 91],
        }
    )

    assert_frame_equal(eager_res, expected)
    assert_frame_equal(lazy_res, expected)
