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

    if not src.suffix == ".json":
        raise ValueError(f"Файл в формате {src.suffix} не поддерживается")

    data = pl.read_json(src)
    df = pl.DataFrame(data)
    df = df.explode("items", empty_as_null=False)
    df = df.unnest("customer")
    df = df.unnest("items")
    df = df.with_columns((pl.col('price')*pl.col('qty')).alias("total_amount"))
    return df