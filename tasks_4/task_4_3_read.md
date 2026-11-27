Задание 3: Сравнение Eager и Lazy при чтении файла и выборке нужных колонок
Суть задания:
Напишите две функции, работающие с CSV-файлом по пути csv_path read_eager_top_scores(csv_path: str, limit: int = 3) -> pl.DataFrame - читает весь файл через pl.read_csv(csv_path), фильтрует score >= 80, выбирает колонки name и score, сортирует по убыванию score и берёт первые limit строк (.head(limit)). read_lazy_top_scores(csv_path: str, limit: int = 3) -> pl.DataFrame — делает то же самое через pl.scan_csv(csv_path) и .collect() в конце.

Подсказка: pl.scan_csv() не читает весь файл сразу в память — оптимизатор (projection pushdown) оставляет только нужные колонки и применяет фильтры на этапе чтения/сканирования.