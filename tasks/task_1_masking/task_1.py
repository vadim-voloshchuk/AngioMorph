from pathlib import Path

import polars as pl

# TODO: Реализуйте функцию загрузки и маскировки данных
def process_masking(src: Path) -> pl.DataFrame:
    """Загружает JSON и маскирует столбец email."""
    df = pl.read_json(src)
    df = df.with_columns(pl.lit("!!!Masked!!!").alias("email"))
    return df
