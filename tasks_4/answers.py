import polars as pl


# Задание 1
def build_sales_pipeline(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.filter(~pl.col("is_returned"))  # оставляем только невозвращённые заказы
        .with_columns((pl.col("price") * pl.col("quantity")).alias("revenue"))
        .group_by("category")
        .agg(pl.col("revenue").sum().alias("total_revenue"))
        .sort("category")
    )


# Задание 2
def filter_before_expensive_agg(lf: pl.LazyFrame, min_year: int) -> pl.DataFrame:
    return (
        lf.filter(pl.col("year") >= min_year)
        .group_by("user_id")
        .agg(pl.col("amount").mean().alias("avg_amount"))
        .collect()
    )


# Задание 3
def read_eager_top_scores(csv_path: str, limit: int = 3) -> pl.DataFrame:
    # Eager: сразу загружаем весь CSV в память
    df = pl.read_csv(csv_path)
    return (
        df.filter(pl.col("score") >= 80)
        .select(["name", "score"])
        .sort("score", descending=True)
        .head(limit)
    )


def read_lazy_top_scores(csv_path: str, limit: int = 3) -> pl.DataFrame:
    # Lazy: строим план сканирования и вычисляем только в конце через .collect()
    return (
        pl.scan_csv(csv_path)
        .filter(pl.col("score") >= 80)
        .select(["name", "score"])
        .sort("score", descending=True)
        .head(limit)
        .collect()
    )