import pandas as pd
from churn_platform.config import RAW_DATA, REFERENCE_DATA
from churn_platform.data import validate_data
from churn_platform.training import train_and_register


def main() -> None:
    df = pd.read_csv(RAW_DATA, parse_dates=["event_date"])
    errors = validate_data(df)
    if errors:
        raise ValueError(errors)
    _, metadata = train_and_register(df)
    REFERENCE_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REFERENCE_DATA, index=False)
    print("Approved model trained", metadata["metrics"])


if __name__ == "__main__":
    main()
