from pathlib import Path

import polars as pl


# TODO: Реализуйте функцию загрузки данных и обрезки массивов
def process_truncation(src: Path) -> pl.DataFrame:
    """Загружает JSON и оставляет максимум 2 элемента в массивах столбца skills."""
    df = pl.read_json(src)
    df = df.with_columns(pl.col("skills").list.head(2))
    return df