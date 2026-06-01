from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_arg: str) -> torch.device:
    if device_arg.lower() == "cpu":
        return torch.device("cpu")
    if device_arg.lower() == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_train_validation_dataset(dataset, val_ratio: float = 0.1) -> tuple[Subset, Subset]:
    n = len(dataset)
    if n < 10:
        return Subset(dataset, list(range(n))), Subset(dataset, [])

    val_size = max(1, int(n * val_ratio))
    train_size = n - val_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, n))
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    if len(data_loader.dataset) == 0:
        return float("nan")

    model.eval()
    losses = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch).squeeze(-1)
            loss = criterion(pred, y_batch)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    checkpoint_path: Path | None = None,
    log_prefix: str | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: list[dict[str, float]] = []
    best_state = None
    best_val_loss = np.inf
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if log_prefix:
            print(
                f"{log_prefix} epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )

        selection_loss = val_loss if np.isfinite(val_loss) else train_loss
        improved = np.isfinite(selection_loss) and selection_loss < best_val_loss
        if improved:
            best_val_loss = selection_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            if log_prefix:
                print(f"{log_prefix} new best checkpoint at epoch {epoch}")
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if log_prefix:
                    print(f"{log_prefix} early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif checkpoint_path is not None and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    return model, history


def predict_loader(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch).squeeze(-1).cpu().numpy()
            preds.append(pred)
    if not preds:
        return np.array([], dtype=float)
    return np.concatenate(preds)
