from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from .task_1 import process_nested_orders


@pytest.fixture
def test_json_file(tmp_path: Path) -> Path:
    """Фикстура для создания временного JSON файла перед тестом."""
    json_data = """
    [
      {
        "order_id": "ORD-001",
        "customer": {"user_id": 101, "region": "EU"},
        "items": [
          {"product": "Laptop", "price": 1200.0, "qty": 1},
          {"product": "Mouse", "price": 25.5, "qty": 2}
        ]
      },
      {
        "order_id": "ORD-002",
        "customer": {"user_id": 102, "region": "US"},
        "items": [
          {"product": "Keyboard", "price": 45.0, "qty": 1}
        ]
      }
    ]
    """
    file_path = tmp_path / "orders.json"
    file_path.write_text(json_data, encoding="utf-8")
    return file_path


@pytest.fixture
def expected_dataframe() -> pl.DataFrame:
    """Эталонный датафрейм, который должен получиться в итоге."""
    return pl.DataFrame({
        "order_id": ["ORD-001", "ORD-001", "ORD-002"],
        "user_id": [101, 101, 102],
        "region": ["EU", "EU", "US"],
        "product": ["Laptop", "Mouse", "Keyboard"],
        "price": [1200.0, 25.5, 45.0],
        "qty": [1, 2, 1],
        "total_amount": [1200.0, 51.0, 45.0]
    })


def test_process_nested_orders(test_json_file: Path, expected_dataframe: pl.DataFrame):
    """Сам тест, сравнивающий результат вашей функции с эталоном."""
    result_df = process_nested_orders(test_json_file)
    
    # Проверка на совпадение схем и данных
    assert_frame_equal(result_df, expected_dataframe, check_column_order=False)