# библиотека для работы с json
import json

# библиотека для работы с путями
from pathlib import Path

# библиотека для работы с табличными типами данных
import polars as pl

# импортируем тестер
import pytest

# функция для сравенения датафрема
from polars.testing import assert_frame_equal

# создаем объект Path с расположение json файла
src = Path("/home/jaroslav/work/practice/tasc_1.json")


def read_json(src: Path | str) -> list[dict]:
    src = Path(src)

    # проверяем, если файла нет, то вызаваем ошибку
    if not src.is_file():
        raise FileNotFoundError("Файл не найден")

    # проверяем, если расширение файла не json, то вызаваем ошибку
    if src.suffix != ".json":
        raise ValueError(f"Расжирение файла {src.suffix}, а долдно быть .json")

    # Рекомендуется всегда указывать кодировку utf-8 при работе с JSON
    with open(src, "r", encoding="utf-8") as fh:
        # json.load читает файл целиком и преобразует его в Python-словарь или список
        data = json.load(fh)

    # возвращаем значение типа list[dict]
    return data


# функция для развертки массива в словарь
def restruct_list_to_dict(data: list[dict], keys: set) -> dict:
    # инициализируем начальный словарь
    start_dict = {}
    # проходимся по ключам
    for key in keys:
        # ишим значение в каждом словаре по ключам
        start_dict.update({key: [val.get(key) for val in data]})
    # возвращаем развернутый массив
    return start_dict


# функция для создания датасета
def make_df(data: dict) -> pl.DataFrame:
    return pl.DataFrame(data)


@pytest.fixture
def expected_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": ["Alex", "Inna", "Anna", "Dmitry"],
            "age": [25, 30, 22, 20],
            "city": ["London", "Moskow", "Berlin", "Liverpool"],
            "tags": [
                ["HR", "Python"],
                ["Ml", "Python", "Rust"],
                ["Ml", "DataScience", "Kafka"],
                ["Apache spark", "Apache pulsar", "Kafka"],
            ],
        }
    )


@pytest.fixture
def test_link() -> Path:
    return Path("/home/jaroslav/work/practice/tasc_1.json")


def test_df_load(expected_dataframe: pl.DataFrame, test_link: Path):
    json_list = read_json(test_link)

    data = restruct_list_to_dict(json_list, set(json_list[0].keys()))
    df = make_df(data)
    schema = df.schema.names()
    expected_dataframe = expected_dataframe.select(pl.col(schema))
    assert_frame_equal(expected_dataframe, df)
