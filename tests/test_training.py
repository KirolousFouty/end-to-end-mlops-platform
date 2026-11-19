from churn_platform.data import generate_customers
from churn_platform.training import chronological_split, fit_candidates


def test_temporal_split_and_reproducible_model():
    df = generate_customers(700, seed=4)
    train, val, test = chronological_split(df)
    assert train.event_date.max() < val.event_date.min() < test.event_date.min()
    _, name, results = fit_candidates(train, val)
    assert name in results and results[name].roc_auc >= 0.5
