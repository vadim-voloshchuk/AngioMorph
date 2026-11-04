import logging
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def process_nested_orders(file_path: Path | str) -> pl.DataFrame:
    """
    Читает вложенный JSON заказов, разворачивает массивы и структуры,
    добавляет расчетную колонку total_amount.
    """
    src = Path(file_path)
    if not src.is_file():
        raise FileNotFoundError(f"Файл {src} не найден!")

    df = pl.read_json(src)
    df = df.explode("items")
    df = df.unnest("customer", "items")
    df = df.with_columns((pl.col("price") * pl.col("qty")).alias("total_amount"))

    return df

# pytest "tasks_2/task_1/test_task_1.py" -v