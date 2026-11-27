import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from .task_1 import process_masking


@pytest.fixture
def expected_masking_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": ["Alex", "Inna", "Anna"],
            "age": [25, 30, 22],
            "email": ["!!!Masked!!!", "!!!Masked!!!", "!!!Masked!!!"],
        }
    )


def test_process_masking(tmp_path: Path, expected_masking_dataframe: pl.DataFrame):
    test_file = tmp_path / "task_1.json"
    json_data = [
        {"name": "Alex", "age": 25, "email": "alex.smith@example.com"},
        {"name": "Inna", "age": 30, "email": "inna.ivanova@example.com"},
        {"name": "Anna", "age": 22, "email": "anna.k@example.com"},
    ]
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    result_df = process_masking(test_file)

    assert_frame_equal(expected_masking_dataframe, result_df)
