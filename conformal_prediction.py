from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class InductiveConformalPredictor:
    def __init__(self, model: nn.Module, confidence: float = 0.95, device: torch.device | None = None):
        self.model = model
        self.confidence = confidence
        self.device = device or torch.device("cpu")
        self.q: float | None = None

    def calibrate(self, calib_loader: DataLoader) -> float:
        self.model.eval()
        residuals = []

        with torch.no_grad():
            for x_batch, y_batch in calib_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred = self.model(x_batch).squeeze(-1)
                residuals.append(torch.abs(y_batch - pred).cpu().numpy())

        if not residuals:
            raise ValueError("Calibration loader is empty; cannot compute conformal quantile.")

        residuals_arr = np.concatenate(residuals)
        self.q = self._conformal_quantile(residuals_arr, self.confidence)
        return float(self.q)

    def calibrate_from_arrays(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        residuals = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
        self.q = self._conformal_quantile(residuals, self.confidence)
        return float(self.q)

    @staticmethod
    def _conformal_quantile(residuals: np.ndarray, confidence: float) -> float:
        residuals = np.asarray(residuals, dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        if len(residuals) == 0:
            return 0.0
        n = len(residuals)
        rank = int(np.ceil((n + 1) * confidence))
        rank = min(max(rank, 1), n)
        return float(np.sort(residuals)[rank - 1])

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.q is None:
            raise RuntimeError("Conformal predictor is not calibrated. Call calibrate() first.")
        self.model.eval()
        with torch.no_grad():
            point = self.model(x.to(self.device)).squeeze(-1)
        lower = point - self.q
        upper = point + self.q
        return point, lower, upper

    def predict_loader(self, test_loader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.q is None:
            raise RuntimeError("Conformal predictor is not calibrated. Call calibrate() first.")

        points = []
        lowers = []
        uppers = []
        self.model.eval()
        with torch.no_grad():
            for x_batch, _ in test_loader:
                x_batch = x_batch.to(self.device)
                point = self.model(x_batch).squeeze(-1).cpu().numpy()
                points.append(point)
                lowers.append(point - self.q)
                uppers.append(point + self.q)

        if not points:
            empty = np.array([], dtype=float)
            return empty, empty, empty
        return (
            np.concatenate(points),
            np.concatenate(lowers),
            np.concatenate(uppers),
        )

    def interval_from_point_predictions(self, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q is None:
            raise RuntimeError("Conformal predictor is not calibrated. Call calibrate() first.")
        arr = np.asarray(y_pred, dtype=float)
        return arr - self.q, arr + self.q
