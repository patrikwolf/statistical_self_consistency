
def add_to_results(results: dict, key: str, value):
    if key in results:
        assert results[key] == value, f"Mismatch for key '{key}' and value '{value}'"
    else:
        results[key] = value
