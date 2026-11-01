import functools
from typing import Callable, Any

# --- Здесь предполагается ваше решение ---
def call_counter(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        wrapper.calls += 1
        return func(*args, **kwargs)

    wrapper.calls = 0
    return wrapper

def test_call_counter() -> None:
    @call_counter
    def say_hello() -> str:
        return "Привет"

    assert hasattr(say_hello, "calls")
    assert say_hello.calls == 0

    assert say_hello() == "Привет"
    assert say_hello.calls == 1

    say_hello()
    say_hello()
    assert say_hello.calls == 3