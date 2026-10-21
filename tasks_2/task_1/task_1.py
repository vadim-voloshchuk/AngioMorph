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

    # ВАШ КОД ЗДЕСЬ:
    # 1. Прочитать JSON
    # 2. Использовать explode для списка товаров
    # 3. Использовать unnest для структур customer и items
    # 4. Добавить колонку total_amount (price * qty)
    # return ...
    return pl.DataFrame()