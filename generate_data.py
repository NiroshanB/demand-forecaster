"""Generate a realistic synthetic daily retail sales series.

The series is SYNTHETIC: a base level plus a linear trend, weekly
seasonality (weekends sell more), annual seasonality, holiday spikes
(Black Friday, Christmas) and random noise. The fixed seed makes the
series reproducible.
"""
import numpy as np
import pandas as pd

RANDOM_SEED = 42
START_DATE = "2023-01-01"
N_DAYS = 730  # two years of daily sales


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    t = np.arange(N_DAYS)

    base = 200.0
    trend = 0.15 * t
    weekly = np.where(dates.day_of_week >= 5, 45.0, 0.0)  # weekend lift
    annual = 25.0 * np.sin(2 * np.pi * dates.day_of_year / 365.25)
    holiday = np.zeros(N_DAYS)
    for i, d in enumerate(dates):
        if d.month == 12 and d.day in range(20, 26):
            holiday[i] = 120.0
        if d.month == 11 and d.day in range(24, 28):  # Black Friday window
            holiday[i] = 150.0
        if d.month == 7 and d.day in range(1, 5):
            holiday[i] = 40.0
    noise = rng.normal(0, 18, N_DAYS)

    sales = np.maximum(20, base + trend + weekly + annual + holiday + noise)
    df = pd.DataFrame({"date": dates, "sales": np.round(sales, 1)})
    df.to_csv("data/daily_sales.csv", index=False)
    print(f"Wrote {len(df)} days to data/daily_sales.csv")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
