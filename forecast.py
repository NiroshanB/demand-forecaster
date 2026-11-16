"""Daily demand forecasting with lag features and gradient boosting.

Builds lag and rolling-mean features from the daily sales series, trains
a HistGradientBoostingRegressor on a time-based split (last 60 days are
the test set), and compares it against a naive seasonal baseline
(sales from 7 days ago). Saves the forecast, metrics and a chart.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

TEST_DAYS = 60


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["date"].dt.day_of_week
    df["month"] = df["date"].dt.month
    df["dayofyear"] = df["date"].dt.day_of_year
    df["lag_1"] = df["sales"].shift(1)
    df["lag_7"] = df["sales"].shift(7)
    df["lag_14"] = df["sales"].shift(14)
    df["lag_28"] = df["sales"].shift(28)
    df["roll_7"] = df["sales"].shift(1).rolling(7).mean()
    df["roll_28"] = df["sales"].shift(1).rolling(28).mean()
    return df


FEATURES = ["dow", "month", "dayofyear", "lag_1", "lag_7", "lag_14",
            "lag_28", "roll_7", "roll_28"]


def main() -> None:
    df = pd.read_csv("data/daily_sales.csv", parse_dates=["date"])
    df = make_features(df).dropna().reset_index(drop=True)

    split = len(df) - TEST_DAYS
    train, test = df.iloc[:split], df.iloc[split:]

    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                          max_depth=None, random_state=42)
    model.fit(train[FEATURES], train["sales"])
    test_pred = model.predict(test[FEATURES])

    # Naive seasonal baseline: forecast = sales 7 days ago.
    baseline_pred = test["lag_7"].values

    mae = mean_absolute_error(test["sales"], test_pred)
    rmse = root_mean_squared_error(test["sales"], test_pred)
    base_mae = mean_absolute_error(test["sales"], baseline_pred)
    base_rmse = root_mean_squared_error(test["sales"], baseline_pred)

    print(f"Test period: {test['date'].min().date()} to {test['date'].max().date()} ({len(test)} days)")
    print(f"Gradient boosting  MAE: {mae:.2f}  RMSE: {rmse:.2f}")
    print(f"Naive (lag 7)      MAE: {base_mae:.2f}  RMSE: {base_rmse:.2f}")
    print(f"MAE improvement over baseline: {(base_mae - mae) / base_mae * 100:.1f}%")

    with open("outputs/metrics.txt", "w") as f:
        f.write(f"Test period: {test['date'].min().date()} to {test['date'].max().date()} ({len(test)} days)\n")
        f.write(f"Gradient boosting  MAE: {mae:.2f}  RMSE: {rmse:.2f}\n")
        f.write(f"Naive (lag 7)      MAE: {base_mae:.2f}  RMSE: {base_rmse:.2f}\n")
        f.write(f"MAE improvement over baseline: {(base_mae - mae) / base_mae * 100:.1f}%\n")

    out = test[["date", "sales"]].copy()
    out["forecast"] = np.round(test_pred, 1)
    out["baseline"] = np.round(baseline_pred, 1)
    out.to_csv("outputs/forecast.csv", index=False)
    print("Saved outputs/forecast.csv")

    plot_df = df.iloc[split - 30:].copy()
    plot_pred = np.concatenate([np.full(30, np.nan), test_pred])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(plot_df["date"], plot_df["sales"], label="Actual sales", linewidth=1.2)
    ax.plot(plot_df["date"], plot_pred, label="Forecast (gradient boosting)",
            linewidth=1.5, linestyle="--")
    ax.axvline(test["date"].min(), color="grey", linestyle=":", label="Test period starts")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily sales (units)")
    ax.set_title("Daily Retail Demand — Actual vs Forecast")
    ax.legend()
    fig.tight_layout()
    fig.savefig("outputs/forecast_plot.png", dpi=150)
    print("Saved outputs/forecast_plot.png")


if __name__ == "__main__":
    main()
