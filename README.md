# Daily Retail Demand Forecaster

Forecasts daily retail sales with a gradient boosting model trained on
lag and rolling-mean features, evaluated on a time-based holdout against
a naive seasonal baseline.

## Skills demonstrated

- Python, Pandas, NumPy, Matplotlib
- Time-series feature engineering: lags (1, 7, 14, 28 days), rolling means,
  day-of-week / month / day-of-year
- Gradient boosting regression (scikit-learn `HistGradientBoostingRegressor`)
- Time-based train/test split (no shuffling, to avoid leaking the future)
- Evaluation with MAE and RMSE, benchmarked against a naive baseline

## Data

The series is **synthetic**: `generate_data.py` builds two years of daily
sales (2023-01-01 to 2024-12-31) from a base level plus a trend, weekly
seasonality (weekend lift), annual seasonality, holiday spikes (Black
Friday, Christmas) and random noise. The generation seed is fixed, so
the series is reproducible.

## Setup

```bash
pip install -r requirements.txt
```

## How to run

```bash
# 1. Generate the series (creates data/daily_sales.csv)
python generate_data.py

# 2. Train and forecast (creates outputs/forecast.csv,
#    outputs/metrics.txt, outputs/forecast_plot.png)
python forecast.py
```

## Results

Test period: last 60 days of the series (2024-11-01 to 2024-12-30,
which includes the Black Friday and Christmas spikes, so errors are
higher than in a quiet period).

| Model | MAE | RMSE |
|---|---|---|
| Gradient boosting | 29.71 | 39.60 |
| Naive baseline (sales 7 days ago) | 48.14 | 68.75 |

The model cuts MAE by **38.3%** versus the naive seasonal baseline.

(The numbers above are from the run in this repo; re-running
`forecast.py` reproduces them.)

## Project structure

```
demand-forecaster/
├── generate_data.py  # builds the synthetic daily sales series
├── forecast.py       # feature engineering, training, evaluation, chart
├── data/
│   └── daily_sales.csv   # generated series (730 days)
├── outputs/
│   ├── forecast.csv       # actual vs forecast for the test period
│   ├── metrics.txt        # MAE / RMSE vs baseline
│   └── forecast_plot.png  # actual vs forecast chart
└── requirements.txt
```
