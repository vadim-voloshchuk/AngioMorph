import time
import functools

def timeout_warning(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        work_time = end_time - start_time
        if work_time > 0.5:
            print(f"Предупреждение: Функция '{func.__name__}' выполнялась {work_time:.2f} сек..")
        
        return result
    
    return wrapper 

def test_timeout_warning_triggers(capsys) -> None:
    @timeout_warning
    def slow_function() -> str:
        time.sleep(0.6)
        return "готово"

    res = slow_function()
    assert res == "готово"

    captured = capsys.readouterr()
    assert "Предупреждение: Функция 'slow_function'" in captured.out

def test_timeout_warning_does_not_trigger(capsys) -> None:
    @timeout_warning
    def fast_function() -> int:
        return 42

    res = fast_function()
    assert res == 42

    captured = capsys.readouterr()
    # Предупреждения быть не должно
    assert captured.out == ""