from pathlib import Path
import polars as pl


def process_split(src: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Разбивает JSON на DataFrame пользователей и устройств."""
    src = Path(src)

    if not src.exists():
        raise FileNotFoundError("Файл не найден!")

    if src.suffix != ".json":
        raise ValueError(f"Файл в формате {src.suffix} не поддерживается")

    data = pl.read_json(src)
    df = pl.DataFrame(data)

    df_users = (df.filter(pl.col("type") == "user").select("user_id", "name"))
    df_devices = (df.filter(pl.col("type") == "device").select("device_id", "os"))

    return df_users, df_devices