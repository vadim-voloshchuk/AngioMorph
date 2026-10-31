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