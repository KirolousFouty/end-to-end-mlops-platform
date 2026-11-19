import pandas as pd
from churn_platform.config import ARTIFACTS, REFERENCE_DATA
from churn_platform.data import generate_customers
from churn_platform.monitoring import drift_report, save_evidently_report


def main() -> None:
    reference = pd.read_csv(REFERENCE_DATA) if REFERENCE_DATA.exists() else generate_customers()
    current = generate_customers(5000, seed=777, drift=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    current.to_csv(ARTIFACTS / "drifted_customers.csv", index=False)
    result = drift_report(reference, current, ARTIFACTS / "drift_report.json")
    save_evidently_report(reference, current, ARTIFACTS / "evidently_drift_report.html")
    print(result)


if __name__ == "__main__":
    main()
