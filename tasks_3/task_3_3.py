import functools

def to_upper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        if isinstance(result, str):
            return result.upper()
        return result

    return wrapper

def test_to_upper_with_string() -> None:
    @to_upper
    def get_greeting() -> str:
        return "hello world"

    assert get_greeting() == "HELLO WORLD"

def test_to_upper_with_non_string() -> None:
    @to_upper
    def get_number() -> int:
        return 12345

    assert get_number() == 12345