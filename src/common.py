# src/common.py
import os
import pandas as pd
import numpy as np

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """
    Loads Walmart.csv and returns cleaned dataframe with:
    - Date parsed to datetime
    - sorted by Store, Date
    """
    df = pd.read_csv("E:\\walmart-forecasting\\Walmart.csv")

    # Parse date: dataset is like 05-02-2010 (dd-mm-yyyy)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # Ensure numeric columns are numeric
    num_cols = ["Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Weekly_Sales"]).copy()
    df["Holiday_Flag"] = df["Holiday_Flag"].fillna(0).astype(int)

    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df

def make_total_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates all stores into ONE weekly time series:
    Date -> total Weekly_Sales, plus averaged exogenous features.
    """
    agg = (
        df.groupby("Date", as_index=False)
          .agg({
              "Weekly_Sales": "sum",
              "Holiday_Flag": "max",        # if any store holiday flagged, keep 1
              "Temperature": "mean",
              "Fuel_Price": "mean",
              "CPI": "mean",
              "Unemployment": "mean"
          })
          .sort_values("Date")
          .reset_index(drop=True)
    )

    # Ensure weekly frequency index (some datasets can have missing weeks)
    agg = agg.set_index("Date").asfreq("W-FRI")  # Walmart weeks usually end Friday
    # Fill small gaps if any:
    agg["Weekly_Sales"] = agg["Weekly_Sales"].interpolate(method="time")
    for c in ["Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]:
        agg[c] = agg[c].ffill().bfill()
    agg["Holiday_Flag"] = (agg["Holiday_Flag"] > 0).astype(int)

    return agg

def train_test_split_time(ts_df: pd.DataFrame, test_weeks: int = 12):
    """
    Splits by time: last `test_weeks` observations are test.
    """
    if len(ts_df) <= test_weeks + 10:
        raise ValueError("Not enough data. Reduce test_weeks or check dataset length.")
    train = ts_df.iloc[:-test_weeks].copy()
    test = ts_df.iloc[-test_weeks:].copy()
    return train, test

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))
