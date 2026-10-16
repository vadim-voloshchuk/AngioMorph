from pathlib import Path
import polars as pl

src = Path(__file__).parent / "task_2.json"

def process_truncation(src: Path) -> pl.DataFrame:
    """Загружает JSON и оставляет максимум 2 элемента в массивах столбца skills."""
    df = pl.read_json(src)

    df = df.with_columns(pl.col("skills").list.slice(0,2))

    return df
