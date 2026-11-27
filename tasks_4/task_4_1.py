import polars as pl
from polars.testing import assert_frame_equal


def build_sales_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
    frame = lf
    frame = frame.filter(pl.col('is_returned') == False)
    frame = frame.with_columns((pl.col('price') * pl.col('quantity')).alias('revenue'))
    frame = frame.group_by('category').agg(pl.col('revenue').sum().alias('total_revenue'))
    frame = frame.sort('category')
    return frame

def test_build_sales_pipeline_is_lazy() -> None:
    df = pl.DataFrame(
        {
            "category": ["books", "books", "games", "games"],
            "price": [100, 200, 500, 300],
            "quantity": [2, 1, 1, 2],
            "is_returned": [False, True, False, False],
        }
    )

    result_lf = build_sales_pipeline(df.lazy())

    # Проверяем, что вернулся именно LazyFrame, а не готовый DataFrame
    assert isinstance(result_lf, pl.LazyFrame)

    actual = result_lf.collect()
    expected = pl.DataFrame(
        {
            "category": ["books", "games"],
            "total_revenue": [200, 1100],
        }
    )
    assert_frame_equal(actual, expected)