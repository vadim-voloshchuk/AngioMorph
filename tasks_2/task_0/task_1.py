import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

EXTENDED_NAMES = [
    "Anna", "Alex", "Dima", "Ivan", "Ioan", "John", "Kate", "Philipp", 
    "Olga", "Yuri", "Isaac", "Kira", "Anastasia", "Vladislav", "Nikita",
    "Maria", "Sergey", "Elena", "Michael", "Sophia", "Daniel", "Emma",
    "David", "Olivia", "James", "William", "Mia", "Alexander", "Victor"
]


def create_lazy_df(size: int) -> pl.LazyFrame:
    df = pl.DataFrame({
        "name": np.random.choice(EXTENDED_NAMES, size=size),
        "age": np.random.randint(18, 100, size=size),
        "score": np.random.randint(0, 1500, size=size),
    })

    null_injection_exprs = [
        pl.when(pl.Series(np.random.rand(size) < 0.05)).then(None).otherwise(pl.col(col)).alias(col)
        for col in df.columns]

    return df.with_columns(null_injection_exprs).lazy()


def lazy_frame_cleaner(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.drop_nulls()


def pipeline_lazy_frame(
    lf: pl.LazyFrame, 
    transform_func: Callable[[pl.LazyFrame], pl.LazyFrame], 
    output_path: Path | str,
    row_group_size: int = 50_000
) -> None:
    """
    Применяет функцию трансформации к LazyFrame и потоково (батчами) выгружает на диск.
    
    Args:
        lf: Исходный ленивый датафрейм.
        transform_func: Функция трансформации (например, lazy_frame_cleaner).
        output_path: Путь к итоговому Parquet-файлу.
        row_group_size: Размер батча при записи файла (аналог chunk_size).
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Запуск потокового пайплайна (Streaming Out-of-Core)...")
    
    # Идиоматичный пайплайн в Polars: 
    # Чтение (если из файла) -> .pipe(функция) -> .sink_parquet(сохранение)
    (
        lf
        .pipe(transform_func)
        .sink_parquet(
            out_path, 
            row_group_size=row_group_size
        )
    )
    
    logger.info(f"Данные успешно очищены и батчами сохранены в: {out_path}")

if __name__ == "__main__":
    # Генерируем 1 миллион строк
    massive_lazy_df = create_lazy_df(size=1_000_000)
    
    # Запускаем конвейер
    pipeline_lazy_frame(lf=massive_lazy_df, transform_func=lazy_frame_cleaner, output_path="data_lake/cleaned_users.parquet",
        row_group_size=100_000)