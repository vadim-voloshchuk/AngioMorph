import logging
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def process_massive_logs(input_csv: Path | str, output_parquet: Path | str) -> None:
    """
    Лениво читает огромный CSV, фильтрует ошибки, размечает критичность
    и потоково сбрасывает результат в Parquet.
    """
    input_csv = Path(input_csv)
    output_parquet = Path(output_parquet)

    if not input_csv.is_file():
        raise FileNotFoundError(f"Файл {input_csv} не найден!")

    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    (
        pl.scan_csv(input_csv)
        .filter(pl.col("status_code") >= 400)
        .drop_nulls("user_id")
        .with_columns(((pl.col("status_code") == 500) & (pl.col("response_time_ms") > 1000)).alias("is_critical"))
        .drop("endpoint")
        .sink_parquet(output_parquet)
    )    