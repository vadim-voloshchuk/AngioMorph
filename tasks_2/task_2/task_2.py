import logging
from pathlib import Path

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

    # ВАШ КОД ЗДЕСЬ:
    # 1. pl.scan_csv(...)
    # 2. .filter(...)
    # 3. .drop_nulls(...)
    # 4. .with_columns(...) для is_critical
    # 5. .drop(...)
    # 6. .sink_parquet(...)