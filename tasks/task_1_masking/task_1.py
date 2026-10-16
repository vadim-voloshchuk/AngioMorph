from pathlib import Path
import polars as pl

src = Path(__file__).parent / "task_1.json"

def process_masking(src: Path) -> pl.DataFrame:
    """Загружает JSON и маскирует столбец email."""
    df = pl.read_json(src)
    
    df = df.with_columns(
        pl.lit("!!!Masked!!!").alias("email")
    )

    return df
