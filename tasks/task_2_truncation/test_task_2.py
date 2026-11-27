import json
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from .task_2 import process_truncation


@pytest.fixture
def expected_truncation_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {"name": ["Alex", "Inna", "Anna"], "skills": [["Python", "SQL"], ["Rust", "C++"], ["Java"]]}
    )


def test_process_truncation(tmp_path: Path, expected_truncation_dataframe: pl.DataFrame):
    test_file = tmp_path / "task_2.json"
    json_data = [
        {"name": "Alex", "skills": ["Python", "SQL"]},
        {"name": "Inna", "skills": ["Rust", "C++", "Go", "Docker", "Kubernetes"]},
        {"name": "Anna", "skills": ["Java"]},
    ]
    with open(test_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    result_df = process_truncation(test_file)

    assert_frame_equal(expected_truncation_dataframe, result_df)
