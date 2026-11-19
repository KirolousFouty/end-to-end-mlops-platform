from churn_platform.data import generate_customers
from churn_platform.monitoring import drift_report


def test_drift_is_detected_for_shifted_population():
    report = drift_report(
        generate_customers(500, seed=2), generate_customers(500, seed=3, drift=True)
    )
    assert report["drift_detected"]
