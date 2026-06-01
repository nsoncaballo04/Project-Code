from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100.0
    return float(coverage)


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean(upper - lower))


def diebold_mariano_test(
    errors_1: np.ndarray,
    errors_2: np.ndarray,
    power: int = 2,
) -> tuple[float, float]:
    e1 = np.asarray(errors_1, dtype=float)
    e2 = np.asarray(errors_2, dtype=float)
    if len(e1) != len(e2):
        raise ValueError("Error vectors must have the same length for DM test.")
    if len(e1) < 3:
        return float("nan"), float("nan")

    d = np.abs(e1) ** power - np.abs(e2) ** power
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    if var_d <= 0:
        return float("nan"), float("nan")

    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def compile_metrics_table(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    preferred_order = ["Model", "RMSE_MW", "MAE_MW", "PICP_pct", "MPIW_MW", "DM_p_value_vs_TCN"]
    existing_order = [c for c in preferred_order if c in frame.columns]
    remaining = [c for c in frame.columns if c not in existing_order]
    return frame[existing_order + remaining]
