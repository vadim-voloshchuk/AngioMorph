from pathlib import Path
import polars as pl

src = Path(__file__).parent / "task_3.json"

def process_split(src: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Разбивает смешанный JSON на два DataFrame (users и devices) и удаляет пустые столбцы."""
    df = pl.read_json(src)

    df_users = (
        df.filter(pl.col("type") == "user").select("user_id", "name")
    )
    
    df_devices = (
        df.filter(pl.col("type") == "device").select("device_id", "os")
    )

    return df_users, df_devices