# src/lstm_forecast.py
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from common import (
    load_and_prepare, make_total_series, train_test_split_time,
    ensure_dir, rmse, mae
)


def make_sequences(X, y, window_size=12):
    """
    X: [n_samples, n_features]
    y: [n_samples,]
    Returns:
      X_seq: [n_seq, window_size, n_features]
      y_seq: [n_seq,]
    """
    X_seq, y_seq = [], []
    for i in range(window_size, len(X)):
        X_seq.append(X[i - window_size:i, :])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def build_lstm(input_shape):
    """
    input_shape: (window_size, n_features)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    # Update path if your CSV is not inside data/
    DATA_PATH = os.path.join("data", "Walmart.csv")
    OUT_DIR = os.path.join("outputs", "lstm")
    ensure_dir(OUT_DIR)

    # Reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load + aggregate to total weekly sales series
    df = load_and_prepare(DATA_PATH)
    ts = make_total_series(df)

    # Features (multivariate)
    feature_cols = ["Weekly_Sales", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
    data = ts[feature_cols].copy()

    # Time split
    train_df, test_df = train_test_split_time(data, test_weeks=12)

    # Scale features using train only (IMPORTANT!)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    test_scaled = scaler.transform(test_df.values)

    # Combine back for sequence making with continuity
    full_scaled = np.vstack([train_scaled, test_scaled])

    window_size = 12  # 12 weeks ~ about 3 months

    # Target is Weekly_Sales (column index 0)
    y_full = full_scaled[:, 0]
    X_full = full_scaled  # use all features to predict sales

    X_seq, y_seq = make_sequences(X_full, y_full, window_size=window_size)

    # Determine where test starts in sequence space
    train_len = len(train_df)
    test_start_seq_index = train_len - window_size

    X_train = X_seq[:test_start_seq_index]
    y_train = y_seq[:test_start_seq_index]

    X_test = X_seq[test_start_seq_index:]
    y_test = y_seq[test_start_seq_index:]

    # Build model
    model = build_lstm(input_shape=(window_size, X_train.shape[2]))

    es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[es],
        verbose=1
    )

    # Predict (scaled)
    y_pred_scaled = model.predict(X_test).reshape(-1)

    # Inverse transform helper (only for Weekly_Sales)
    n_features = len(feature_cols)

    def inverse_sales(sales_scaled):
        dummy = np.zeros((len(sales_scaled), n_features))
        dummy[:, 0] = sales_scaled
        inv = scaler.inverse_transform(dummy)
        return inv[:, 0]

    y_test_actual = inverse_sales(y_test)
    y_pred_actual = inverse_sales(y_pred_scaled)

    # Metrics (LSTM)
    test_rmse = rmse(y_test_actual, y_pred_actual)
    test_mae = mae(y_test_actual, y_pred_actual)

    # Naive baseline: predict previous week sales (on actual scale)
    baseline_pred = y_test_actual[:-1]
    baseline_true = y_test_actual[1:]
    baseline_rmse = rmse(baseline_true, baseline_pred)
    baseline_mae = mae(baseline_true, baseline_pred)

    improvement = (baseline_rmse - test_rmse) / baseline_rmse * 100

    # Save metrics
    metrics_path = os.path.join(OUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Window size: {window_size}\n")
        f.write(f"Features: {feature_cols}\n")
        f.write(f"LSTM Test RMSE: {test_rmse}\n")
        f.write(f"LSTM Test MAE: {test_mae}\n")
        f.write(f"Baseline RMSE (prev week): {baseline_rmse}\n")
        f.write(f"Baseline MAE (prev week): {baseline_mae}\n")
        f.write(f"LSTM RMSE improvement vs baseline (%): {improvement}\n")

    print("\nEvaluation on last 12 weeks (LSTM):")
    print("  RMSE:", round(test_rmse, 2))
    print("  MAE :", round(test_mae, 2))

    print("\nNaive baseline (previous week) on test:")
    print("  RMSE:", round(baseline_rmse, 2))
    print("  MAE :", round(baseline_mae, 2))

    print(f"\nLSTM RMSE improvement vs baseline: {improvement:.2f}%")
    print("Saved metrics to:", metrics_path)

    # Align prediction dates
    dates = data.index
    seq_dates = dates[window_size:]
    test_dates = seq_dates[test_start_seq_index:]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test_actual, label="Actual")
    plt.plot(test_dates, y_pred_actual, label="LSTM Forecast")
    plt.title("Walmart Weekly Sales Forecast (LSTM)")
    plt.xlabel("Date")
    plt.ylabel("Weekly Sales (Total)")
    plt.legend()
    plot_path = os.path.join(OUT_DIR, "forecast_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("Saved plot to:", plot_path)

    # Save model
    model_path = os.path.join(OUT_DIR, "lstm_model.keras")
    model.save(model_path)
    print("Saved model to:", model_path)


if __name__ == "__main__":
    main()
