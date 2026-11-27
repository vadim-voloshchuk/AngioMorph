Задание 2: Проверка плана оптимизации 
Суть задания:
Напишите функцию filter_before_expensive_agg(lf: pl.LazyFrame, min_year: int) -> pl.DataFrame, которая принимает LazyFrame с колонками year (Int32 / Int64), user_id (Int64) и amount (Float64). Функция должна Отфильтровать записи с year >= min_year, сгруппировать по user_id и посчитать среднюю сумму avg_amount (pl.col("amount").mean()). Выполнить вычисление (.collect()) и вернуть pl.DataFrame.

Подсказка: В Polars оптимизатор запросов сам двигает фильтры ближе к источнику данных (predicate pushdown), если писать их на LazyFrame до .collect().