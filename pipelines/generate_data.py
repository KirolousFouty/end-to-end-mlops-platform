from churn_platform.config import RAW_DATA
from churn_platform.data import generate_customers, validate_data


def main() -> None:
    df = generate_customers()
    errors = validate_data(df)
    if errors:
        raise ValueError(errors)
    RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA, index=False)
    print(f"Wrote {len(df)} synthetic customer records to {RAW_DATA}")


if __name__ == "__main__":
    main()
