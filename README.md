# Walmart Weekly Sales Forecasting

## 📌 Project Overview

This project focuses on forecasting **weekly Walmart sales** using historical data from 2010–2012. The goal is to compare a **statistical time-series model (SARIMAX/ARIMA)** with a **deep learning model (LSTM)** and evaluate their performance.

---

## 🧠 Models Used

### 1️⃣ SARIMAX (ARIMA)

* Captures trend and seasonality
* Uses **Holiday_Flag** as an exogenous variable
* Includes grid search for best parameters

### 2️⃣ LSTM (Deep Learning)

* Multivariate LSTM model
* Uses past sales + holiday and economic indicators
* Compared against a **naive baseline (previous week sales)**

---

## 📈 Evaluation Metrics

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)

Both models are evaluated on the **last 12 weeks** of data.

---

## 📂 Project Structure

```
walmart-forecasting/
├── src/
│   ├── arima_forecast.py
│   ├── lstm_forecast.py
│   └── common.py
├── outputs/
│   ├── arima/
│   └── lstm/
├── Walmart.csv
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run ARIMA model:

```bash
python src/arima_forecast.py
```

3. Run LSTM model:

```bash
python src/lstm_forecast.py
```

---

## ✅ Results

* Generated forecasts and evaluation metrics for both ARIMA and LSTM
* Saved plots, metrics, and trained models in the `outputs/` folder
* Compared models against a naive baseline

## 📊 Results

Models were evaluated on the **last 12 weeks** of data using RMSE and MAE (lower is better).  
A naive baseline (predicting the previous week’s sales) was used for comparison.

| Model | RMSE ↓ | MAE ↓ | vs Baseline (RMSE) |
|------|--------:|------:|-------------------:|
| Naive Baseline (Previous Week) | 1,825,268.32 | 1,195,438.71 | – |
| SARIMAX (order=(0,1,2), seasonal=(0,1,1,52)) | **892,896.12** | **763,460.76** | **+51.08%** |
| LSTM (Multivariate) | 1,538,251.49 | 1,399,755.69 | +15.72% |

### Key Observations
- **SARIMAX performed best**, reducing RMSE by ~51% compared to the baseline, capturing **seasonality (52-week cycle)** and **holiday effects**.
- **LSTM improved over baseline** but underperformed SARIMAX on this dataset, likely due to limited data length and the strong seasonal structure handled well by SARIMAX.
- Both models produced forecast plots and saved outputs for reproducibility.

---

## 🏁 Conclusion

This project demonstrates practical **time-series forecasting**, **model comparison**, and **clean ML project structuring**, making it suitable for **data science / machine learning internships**.

---

