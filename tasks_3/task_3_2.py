import time

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