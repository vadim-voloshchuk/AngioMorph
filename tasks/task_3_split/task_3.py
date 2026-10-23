from pathlib import Path

import pandas as pd
import polars as pl

def drop_empty_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.select([col for col in df.columns if df[col].null_count() < df.height])
    
def process_split(src: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Разбивает смешанный JSON на два DataFrame (users и devices) и удаляет пустые столбцы."""
    df = pd.read_json(src)
    df_users = df.filter(pl.col("type")=="user").drop("type")
    df_devices = df.filter(pl.col("type")=="device").drop("type")

    return drop_empty_columns(df_users), drop_empty_columns(df_devices)