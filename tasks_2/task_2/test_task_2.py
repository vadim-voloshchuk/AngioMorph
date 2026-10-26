from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from .task_2 import process_massive_logs


@pytest.fixture
def test_csv_file(tmp_path: Path) -> Path:
    """Генерирует тестовый CSV-файл."""
    csv_content = """log_id,timestamp,user_id,endpoint,status_code,response_time_ms
1,2026-10-21T10:00,101,/home,200,45
2,2026-10-21T10:01,,/login,401,120
3,2026-10-21T10:02,102,/api/data,500,1500
4,2026-10-21T10:03,103,/api/data,500,800
5,2026-10-21T10:04,101,/dashboard,404,50
"""
    file_path = tmp_path / "server_logs.csv"
    file_path.write_text(csv_content, encoding="utf-8")
    return file_path


@pytest.fixture
def expected_dataframe() -> pl.DataFrame:
    """Эталонный результат, который должен оказаться в Parquet-файле."""
    return pl.DataFrame({
        "log_id": [3, 4, 5],
        "timestamp": ["2026-10-21T10:02", "2026-10-21T10:03", "2026-10-21T10:04"],
        "user_id": [102, 103, 101],
        "status_code": [500, 500, 404],
        "response_time_ms": [1500, 800, 50],
        "is_critical": [True, False, False]
    })


def test_process_massive_logs(test_csv_file: Path, expected_dataframe: pl.DataFrame, tmp_path: Path):
    output_file = tmp_path / "output_logs.parquet"
    
    # 1. Запускаем ваш ETL-пайплайн
    process_massive_logs(test_csv_file, output_file)
    
    # 2. Проверяем, что файл действительно создался
    assert output_file.exists(), "Parquet-файл не был создан!"
    
    # 3. Читаем результат и сверяем с эталоном
    result_df = pl.read_parquet(output_file)
    
    # Приводим типы user_id к одному стандарту, чтобы избежать конфликтов Int32/Int64 при парсинге
    result_df = result_df.cast({"user_id": pl.Int64})
    expected_dataframe = expected_dataframe.cast({"user_id": pl.Int64})

    assert_frame_equal(result_df, expected_dataframe)