# src/arima_forecast.py
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import (
    load_and_prepare, make_total_series, train_test_split_time,
    ensure_dir, rmse, mae
)

warnings.filterwarnings("ignore")


def grid_search_sarimax(train_y, train_exog=None, seasonal_period=52):
    """
    Small grid search for SARIMAX params.
    Keep it small so it runs on normal laptops.
    """
    best = {"aic": np.inf, "order": None, "seasonal_order": None, "model": None}

    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]

    # Seasonal components (keep limited)
    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, seasonal_period)

                            try:
                                model = SARIMAX(
                                    train_y,
                                    exog=train_exog,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                ).fit(disp=False)

                                if model.aic < best["aic"]:
                                    best.update({
                                        "aic": model.aic,
                                        "order": order,
                                        "seasonal_order": seasonal_order,
                                        "model": model
                                    })
                            except Exception:
                                continue

    return best


def main():
    # Update path if your CSV is not inside data/
    DATA_PATH = os.path.join("data", "Walmart.csv")
    OUT_DIR = os.path.join("outputs", "arima")
    ensure_dir(OUT_DIR)

    df = load_and_prepare(DATA_PATH)
    ts = make_total_series(df)

    # Target + exogenous feature(s)
    y = ts["Weekly_Sales"]
    exog = ts[["Holiday_Flag"]]  # simple & effective

    train, test = train_test_split_time(ts, test_weeks=12)
    y_train = train["Weekly_Sales"]
    y_test = test["Weekly_Sales"]
    exog_train = train[["Holiday_Flag"]]
    exog_test = test[["Holiday_Flag"]]

    # Naive baseline (previous week)
    baseline_pred = y_test.shift(1).dropna()
    baseline_true = y_test.loc[baseline_pred.index]
    baseline_rmse = rmse(baseline_true, baseline_pred)
    baseline_mae = mae(baseline_true, baseline_pred)

    print("Running SARIMAX grid search (this may take a few minutes)...")
    best = grid_search_sarimax(y_train, train_exog=exog_train, seasonal_period=52)
    model = best["model"]

    if model is None:
        raise RuntimeError("Grid search failed. Try reducing grid or check data.")

    print("\nBest params:")
    print("  order:", best["order"])
    print("  seasonal_order:", best["seasonal_order"])
    print("  AIC:", round(best["aic"], 2))

    # Forecast for test period
    forecast = model.get_forecast(steps=len(test), exog=exog_test)
    pred_mean = forecast.predicted_mean
    conf_int = forecast.conf_int(alpha=0.05)

    test_rmse = rmse(y_test, pred_mean)
    test_mae = mae(y_test, pred_mean)

    improvement = (baseline_rmse - test_rmse) / baseline_rmse * 100

    # Save forecast CSV
    forecast_df = test[["Weekly_Sales", "Holiday_Flag"]].copy()
    forecast_df["ARIMA_Forecast"] = pred_mean.values
    forecast_df["CI_Lower"] = conf_int.iloc[:, 0].values
    forecast_df["CI_Upper"] = conf_int.iloc[:, 1].values
    forecast_csv_path = os.path.join(OUT_DIR, "forecast.csv")
    forecast_df.to_csv(forecast_csv_path, index=True)

    # Save metrics
    metrics_path = os.path.join(OUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Best SARIMAX order: {best['order']}\n")
        f.write(f"Best seasonal_order: {best['seasonal_order']}\n")
        f.write(f"AIC: {best['aic']}\n")
        f.write(f"ARIMA Test RMSE: {test_rmse}\n")
        f.write(f"ARIMA Test MAE: {test_mae}\n")
        f.write(f"Baseline RMSE (prev week): {baseline_rmse}\n")
        f.write(f"Baseline MAE (prev week): {baseline_mae}\n")
        f.write(f"ARIMA RMSE improvement vs baseline (%): {improvement}\n")

    print("\nEvaluation on last 12 weeks (ARIMA/SARIMAX):")
    print("  RMSE:", round(test_rmse, 2))
    print("  MAE :", round(test_mae, 2))

    print("\nNaive baseline (previous week) on test:")
    print("  RMSE:", round(baseline_rmse, 2))
    print("  MAE :", round(baseline_mae, 2))

    print(f"\nARIMA RMSE improvement vs baseline: {improvement:.2f}%")
    print("Saved metrics to:", metrics_path)
    print("Saved forecast CSV to:", forecast_csv_path)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, y_train, label="Train")
    plt.plot(test.index, y_test, label="Test (Actual)")
    plt.plot(test.index, pred_mean, label="ARIMA Forecast")

    # Confidence interval
    plt.fill_between(
        test.index,
        conf_int.iloc[:, 0].values,
        conf_int.iloc[:, 1].values,
        alpha=0.2,
        label="95% CI"
    )

    plt.title("Walmart Weekly Sales Forecast (SARIMAX)")
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales (Total)")
    plt.legend()
    plot_path = os.path.join(OUT_DIR, "forecast_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("Saved plot to:", plot_path)


if __name__ == "__main__":
    main()

