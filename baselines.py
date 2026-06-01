from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA


class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.head(out)


def fit_arima_forecast(
    train_series: pd.Series,
    test_steps: int,
    p_range: tuple[int, ...] = (0, 1, 2),
    d_range: tuple[int, ...] = (0, 1, 2),
    q_range: tuple[int, ...] = (0, 1, 2),
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    values = pd.to_numeric(train_series, errors="coerce").dropna().astype(float)
    if len(values) < 50:
        if len(values) == 0:
            return np.zeros(test_steps, dtype=float), (0, 0, 0), float("nan")
        return np.full(test_steps, values.iloc[-1], dtype=float), (0, 0, 0), float("nan")

    best_order = None
    best_aic = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    try:
                        result = ARIMA(
                            values,
                            order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit()
                        if np.isfinite(result.aic) and result.aic < best_aic:
                            best_aic = float(result.aic)
                            best_order = (p, d, q)
                    except Exception:
                        continue

    if best_order is None:
        return np.full(test_steps, values.iloc[-1], dtype=float), (0, 0, 0), float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(
            values,
            order=best_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        forecast = fitted.forecast(steps=test_steps)

    return np.asarray(forecast, dtype=float), best_order, best_aic


def naive_persistence_forecast(
    full_target: pd.Series,
    forecast_index: pd.DatetimeIndex,
    freq: str = "15min",
) -> np.ndarray:
    shift_delta = pd.to_timedelta(freq)
    predictions = []
    for ts in forecast_index:
        previous_ts = ts - shift_delta
        value = full_target.get(previous_ts, np.nan)
        predictions.append(value)

    arr = np.asarray(predictions, dtype=float)
    if np.isnan(arr).all():
        arr[:] = 0.0
    else:
        valid = ~np.isnan(arr)
        arr[~valid] = np.nanmean(arr[valid])
    return arr
