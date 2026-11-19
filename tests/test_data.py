from churn_platform.data import generate_customers, validate_data


def test_generation_is_reproducible_and_valid():
    a, b = generate_customers(100, seed=8), generate_customers(100, seed=8)
    assert a.equals(b)
    assert not validate_data(a)


def test_validator_finds_missing_columns():
    assert validate_data(generate_customers(10).drop(columns="churn"))
