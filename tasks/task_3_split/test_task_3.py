import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from .task_3 import process_split


@pytest.fixture
def expected_users_dataframe() -> pl.DataFrame:
    return pl.DataFrame({"user_id": [1, 2], "name": ["Alex", "Inna"]})


@pytest.fixture
def expected_devices_dataframe() -> pl.DataFrame:
    return pl.DataFrame({"device_id": ["A-100", "I-200"], "os": ["Android", "iOS"]})


def test_process_split(
    tmp_path: Path, expected_users_dataframe: pl.DataFrame, expected_devices_dataframe: pl.DataFrame
):
    test_file = tmp_path / "task_3.json"
    json_data = [
        {"type": "user", "user_id": 1, "name": "Alex"},
        {"type": "device", "device_id": "A-100", "os": "Android"},
        {"type": "user", "user_id": 2, "name": "Inna"},
        {"type": "device", "device_id": "I-200", "os": "iOS"},
    ]
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    df_users, df_devices = process_split(test_file)

    assert_frame_equal(expected_users_dataframe, df_users)
    assert_frame_equal(expected_devices_dataframe, df_devices)
