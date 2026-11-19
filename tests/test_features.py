from churn_platform.data import generate_customers
from churn_platform.features import engineer_features, make_preprocessor


def test_features_and_preprocessor_handle_missing_values():
    x = engineer_features(generate_customers(30))
    assert "usage_per_dollar" in x
    assert make_preprocessor().fit_transform(x).shape[0] == 30
