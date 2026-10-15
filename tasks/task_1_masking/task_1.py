from pathlib import Path
import polars as pl

# TODO: Реализуйте функцию загрузки и маскировки данных
def process_masking(src: Path) -> pl.DataFrame:
    """Загружает JSON и маскирует столбец email."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError("Файл не найден!")

    if not src.suffix == ".json":
        raise ValueError(f"Файл в формате {src.suffix} не поддерживается")

    data = pl.read_json(src)
    df = pl.DataFrame(data)
    df = df.with_columns((pl.lit("!!!Masked!!!").alias('email')))
    return df

