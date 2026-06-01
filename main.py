from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import config
from baselines import LSTMBaseline, fit_arima_forecast, naive_persistence_forecast
from conformal_prediction import InductiveConformalPredictor
from data_harmonization import build_feature_target_frame, fit_feature_scaler, transform_features
from dataset import build_split_window_datasets, create_data_loaders, split_time_series
from evaluate import compile_metrics_table, diebold_mariano_test, mae, mpiw, picp, rmse
from rolling_origin import run_rolling_origin_validation
from tcn_model import TCN
from train import predict_loader, select_device, set_seed, split_train_validation_dataset, train_model
from visualize import (
    plot_calibration_curve,
    plot_forecast_with_intervals,
    plot_model_comparison,
    plot_rolling_origin_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TCN-CP pipeline for AG4 + AG5 nomination support.")
    parser.add_argument("--start-date", default=config.DATE_START)
    parser.add_argument("--end-date", default=config.DATE_END)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lstm-epochs", type=int, default=config.LSTM_EPOCHS)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--confidence-level", type=float, default=config.CONFIDENCE_LEVEL)
    parser.add_argument("--run-rolling-origin", action="store_true")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return parser.parse_args()


def _save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)


def _split_info_to_dict(split_infos) -> list[dict]:
    out = []
    for item in split_infos:
        out.append(
            {
                "name": item.name,
                "start": item.start,
                "end": item.end,
                "rows": item.rows,
                "window_samples": item.window_samples,
            }
        )
    return out


def _calibration_curve_from_calib_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence_levels: list[float],
) -> list[float]:
    residuals = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    coverages = []
    for level in confidence_levels:
        n = len(residuals)
        if n == 0:
            coverages.append(np.nan)
            continue
        rank = int(np.ceil((n + 1) * level))
        rank = min(max(rank, 1), n)
        q = np.sort(residuals)[rank - 1]
        lower = y_pred - q
        upper = y_pred + q
        coverages.append(picp(y_true, lower, upper))
    return coverages


def main() -> None:
    args = parse_args()
    config.ensure_output_dirs()
    set_seed(args.seed)
    device = select_device(args.device)

    print("[1/8] Loading and harmonizing data...")
    features, target, harmonization_metadata = build_feature_target_frame(
        date_start=args.start_date,
        date_end=args.end_date,
    )

    _save_json(
        {
            **asdict(harmonization_metadata),
            "feature_columns": list(features.columns),
            "target_name": target.name,
            "rows": len(features),
        },
        config.RESULTS_DIR / "harmonization_metadata.json",
    )
    _save_json(harmonization_metadata.rtd_gap_report, config.RESULTS_DIR / "rtd_gap_report.json")

    print("[2/8] Splitting time series and scaling features...")
    split_series = split_time_series(features, target, config.SPLIT_RATIOS)
    scaler = fit_feature_scaler(split_series["train"]["features"])
    with (config.MODELS_DIR / "feature_scaler.pkl").open("wb") as file:
        pickle.dump(scaler, file)
    for split_name in split_series:
        split_series[split_name]["features"] = transform_features(
            split_series[split_name]["features"], scaler
        )

    datasets, split_infos = build_split_window_datasets(
        split_series=split_series,
        window=config.WINDOW_SIZE,
        horizon=config.FORECAST_HORIZON,
    )
    _save_json({"splits": _split_info_to_dict(split_infos)}, config.RESULTS_DIR / "split_manifest.json")

    if len(datasets["train"]) == 0 or len(datasets["calib"]) == 0 or len(datasets["test"]) == 0:
        raise RuntimeError("At least one split has zero window samples. Check date range and preprocessing.")

    loaders = create_data_loaders(datasets, batch_size=args.batch_size)

    print("[3/8] Training TCN...")
    train_subset, val_subset = split_train_validation_dataset(
        datasets["train"], val_ratio=config.VAL_RATIO_WITHIN_TRAIN
    )
    tcn_train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    tcn_val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    tcn = TCN(
        input_channels=config.INPUT_CHANNELS,
        hidden_filters=config.HIDDEN_FILTERS,
        kernel_size=config.KERNEL_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )
    tcn, tcn_history = train_model(
        model=tcn,
        train_loader=tcn_train_loader,
        val_loader=tcn_val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=config.MODELS_DIR / "tcn_best.pt",
        log_prefix="[TCN]",
    )
    pd.DataFrame(tcn_history).to_csv(config.RESULTS_DIR / "tcn_training_history.csv", index=False)

    y_calib = datasets["calib"].y.numpy()
    y_test = datasets["test"].y.numpy()
    test_index = datasets["test"].timestamps

    calib_pred = predict_loader(tcn, loaders["calib"], device=device)
    test_pred = predict_loader(tcn, loaders["test"], device=device)

    print("[4/8] Calibrating conformal prediction...")
    icp = InductiveConformalPredictor(tcn, confidence=args.confidence_level, device=device)
    q_value = icp.calibrate_from_arrays(y_calib, calib_pred)
    tcn_lower, tcn_upper = icp.interval_from_point_predictions(test_pred)

    print("[5/8] Training baseline models...")
    lstm = LSTMBaseline(
        input_size=config.INPUT_CHANNELS,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
        dropout=config.LSTM_DROPOUT,
    )
    lstm, lstm_history = train_model(
        model=lstm,
        train_loader=tcn_train_loader,
        val_loader=tcn_val_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.lstm_epochs,
        patience=args.patience,
        checkpoint_path=config.MODELS_DIR / "lstm_best.pt",
        log_prefix="[LSTM]",
    )
    pd.DataFrame(lstm_history).to_csv(config.RESULTS_DIR / "lstm_training_history.csv", index=False)
    lstm_pred = predict_loader(lstm, loaders["test"], device=device)

    history_end = pd.Timestamp(test_index.min()) - pd.Timedelta(config.MODEL_FREQ)
    history_series = target[target.index <= history_end]
    arima_pred, arima_order, arima_aic = fit_arima_forecast(
        train_series=history_series,
        test_steps=len(test_index),
        p_range=config.ARIMA_P_RANGE,
        d_range=config.ARIMA_D_RANGE,
        q_range=config.ARIMA_Q_RANGE,
    )
    persist_pred = naive_persistence_forecast(
        full_target=target,
        forecast_index=pd.DatetimeIndex(test_index),
        freq=config.MODEL_FREQ,
    )

    print("[6/8] Evaluating models...")
    tcn_errors = y_test - test_pred
    lstm_errors = y_test - lstm_pred
    arima_errors = y_test - arima_pred
    persist_errors = y_test - persist_pred

    _, dm_p_lstm = diebold_mariano_test(tcn_errors, lstm_errors)
    _, dm_p_arima = diebold_mariano_test(tcn_errors, arima_errors)
    _, dm_p_persist = diebold_mariano_test(tcn_errors, persist_errors)

    metrics_rows = [
        {
            "Model": "TCN-CP",
            "RMSE_MW": rmse(y_test, test_pred),
            "MAE_MW": mae(y_test, test_pred),
            "PICP_pct": picp(y_test, tcn_lower, tcn_upper),
            "MPIW_MW": mpiw(tcn_lower, tcn_upper),
            "DM_p_value_vs_TCN": np.nan,
        },
        {
            "Model": "LSTM",
            "RMSE_MW": rmse(y_test, lstm_pred),
            "MAE_MW": mae(y_test, lstm_pred),
            "PICP_pct": np.nan,
            "MPIW_MW": np.nan,
            "DM_p_value_vs_TCN": dm_p_lstm,
        },
        {
            "Model": "ARIMA",
            "RMSE_MW": rmse(y_test, arima_pred),
            "MAE_MW": mae(y_test, arima_pred),
            "PICP_pct": np.nan,
            "MPIW_MW": np.nan,
            "DM_p_value_vs_TCN": dm_p_arima,
        },
        {
            "Model": "Persistence",
            "RMSE_MW": rmse(y_test, persist_pred),
            "MAE_MW": mae(y_test, persist_pred),
            "PICP_pct": np.nan,
            "MPIW_MW": np.nan,
            "DM_p_value_vs_TCN": dm_p_persist,
        },
    ]
    metrics_df = compile_metrics_table(metrics_rows)
    metrics_df.to_csv(config.RESULTS_DIR / "metrics.csv", index=False)

    predictions_df = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(test_index),
            "y_true": y_test,
            "tcn_pred": test_pred,
            "tcn_lower": tcn_lower,
            "tcn_upper": tcn_upper,
            "lstm_pred": lstm_pred,
            "arima_pred": arima_pred,
            "persist_pred": persist_pred,
        }
    )
    predictions_df.to_csv(config.RESULTS_DIR / "predictions_test.csv", index=False)

    _save_json(
        {
            "q_value_mw": q_value,
            "confidence_level": args.confidence_level,
            "arima_order": list(arima_order),
            "arima_aic": arima_aic,
            "device": str(device),
        },
        config.RESULTS_DIR / "model_details.json",
    )

    print("[7/8] Generating plots...")
    plot_forecast_with_intervals(
        y_true=y_test,
        y_pred=test_pred,
        lower=tcn_lower,
        upper=tcn_upper,
        output_path=config.PLOTS_DIR / "forecast_with_intervals.png",
        title="TCN-CP Forecast on Test Set",
    )
    plot_model_comparison(
        metrics_df=metrics_df,
        output_path=config.PLOTS_DIR / "model_comparison.png",
    )
    confidence_levels = [0.80, 0.85, 0.90, 0.95]
    observed_coverages = _calibration_curve_from_calib_residuals(
        y_true=y_calib,
        y_pred=calib_pred,
        confidence_levels=confidence_levels,
    )
    plot_calibration_curve(
        confidence_levels=confidence_levels,
        observed_coverages=observed_coverages,
        output_path=config.PLOTS_DIR / "calibration_curve.png",
    )

    print("[8/8] Optional rolling-origin validation...")
    if args.run_rolling_origin:
        fold_df = run_rolling_origin_validation(
            features=features,
            target=target,
            model_builder=lambda: TCN(
                input_channels=config.INPUT_CHANNELS,
                hidden_filters=config.HIDDEN_FILTERS,
                kernel_size=config.KERNEL_SIZE,
                num_layers=config.NUM_LAYERS,
                dropout=config.DROPOUT,
            ),
            device=device,
            train_months=config.ROLLING_TRAIN_MONTHS,
            test_months=config.ROLLING_TEST_MONTHS,
            window=config.WINDOW_SIZE,
            horizon=config.FORECAST_HORIZON,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=config.ROLLING_EPOCHS,
            patience=max(3, args.patience // 2),
            max_folds=config.ROLLING_MAX_FOLDS,
        )
        fold_df.to_csv(config.RESULTS_DIR / "rolling_origin.csv", index=False)
        plot_rolling_origin_results(
            fold_results=fold_df,
            output_path=config.PLOTS_DIR / "rolling_origin_rmse.png",
        )

    print("Pipeline completed.")
    print(f"Results: {config.RESULTS_DIR}")
    print(f"Plots:   {config.PLOTS_DIR}")


if __name__ == "__main__":
    main()
