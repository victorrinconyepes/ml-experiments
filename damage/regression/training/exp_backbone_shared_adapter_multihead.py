#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Head Multi-Output Regression (log1p) con:
- Backbone compartido (timm) + bloque adaptador + una cabeza por target.
- Pérdida L1 enmascarada ponderada (ignora ceros en warmup, luego peso pequeño).
- Regularización suave sobre predicciones en ceros.
- Sampler opcional balanceado de positivos por target (--pos-per-target).
- Métricas ampliadas: mae_pos, rel_mae, medae_pos, q90_abs_err_pos, coverage_±5_pos, mae_top10pct_pos,
  precision/recall/f1 (umbral), corr, r2 y métricas macro correspondientes.
- Análisis por percentiles (bins) con métricas extendidas.
- Logging MLflow similar al baseline original.

Uso mínimo:
python baseline_multioutput.py --csv data.csv --targets t1 t2 t3 t4 t5 --mlflow

Recomendado (si dataset muy desbalanceado):
python baseline_multioutput.py --csv data.csv --targets ... --pos-per-target 20 --zero-warmup-epochs 3 --zero-weight 0.05 --mlflow
"""

import os
import math
import argparse
from typing import List, Dict
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

try:
    import timm
except ImportError:
    raise ImportError("Necesitas 'timm': pip install timm")

try:
    import geopandas as gpd
except ImportError:
    gpd = None
import fiona

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    median_absolute_error,
)

# MLflow helpers (asumimos que existen en tu entorno como en el baseline original)
from pyvexcelutils.aws.secrets import Secrets
from pyvexcelutils.mlflow.mlflow_helper import setup_mlflow, MLFlowNamer


# -------------------------------
# Thresholds (para métricas tipo clasificación)
# -------------------------------
THRESHOLDS = {
    "roof_condition_rust_percent": 0.9,
    "roof_discoloration_algae_staining_percen": 0.9,
    "roof_discoloration_vent_staining_percent": 0.9,
    "roof_discoloration_water_pooling_percent": 0.9,
    "roof_discoloration_debris_percent": 0.9,
}


# -------------------------------
# Utilidades
# -------------------------------
def validate_thresholds(target_cols: List[str], thresholds: Dict[str, float]) -> Dict[str, float]:
    import difflib
    final = {}
    for col in target_cols:
        if col in thresholds:
            final[col] = thresholds[col]
        else:
            close = difflib.get_close_matches(col, thresholds.keys(), n=1, cutoff=0.8)
            if close:
                print(f"[WARN] Target '{col}' no en THRESHOLDS, usando '{close[0]}'={thresholds[close[0]]:.4f} (fuzzy).")
                final[col] = thresholds[close[0]]
            else:
                print(f"[WARN] Target '{col}' sin threshold definido. Usando default 0.1")
                final[col] = 0.1
    return final


def inverse_transform(y_tr: torch.Tensor) -> torch.Tensor:
    y_norm = torch.expm1(y_tr).clamp(min=0)
    return (y_norm * 100.0).clamp(0, 100)


class MultiTargetImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_cols: List[str], transform=None):
        self.df = df.reset_index(drop=True)
        self.target_cols = target_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        y = row[self.target_cols].values.astype("float32")
        y_tr = np.log1p(y / 100.0)
        return image, torch.from_numpy(y_tr), img_path

def set_seeds(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinismo (opcional, puede ralentizar)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------
# Sampler opcional balanceado
# -------------------------------
# REVISAR
# Eso genera un iterador infinito. DataLoader, cuando le pasas batch_sampler=..., itera “for batch in batch_sampler”, esperando que el iterador termine (StopIteration). Aquí nunca termina → el epoch nunca acaba → run_epoch se queda colgado.
#
# Solución: limitar explícitamente el número de batches por época. Puedes:
#
#     Definir batches_per_epoch = ceil(N / batch_size) (o un parámetro).
#     Usar for _ in range(self.batches_per_epoch): en vez de while True.

# class BalancedPositiveBatchSampler(Sampler):
#     """
#     Garantiza una cuota de ejemplos positivos para cada target en cada batch.
#     Asume que un target es 'positivo' si su valor > 0 (en escala original).
#     - df: DataFrame completo (split train)
#     - target_cols: lista de targets
#     - batch_size
#     - pos_per_target: nº de positivos por target a intentar muestrear
#     """
#     def __init__(self, df: pd.DataFrame, target_cols: List[str], batch_size: int, pos_per_target: int, seed: int = 42):
#         self.df = df.reset_index(drop=True)
#         self.target_cols = target_cols
#         self.batch_size = batch_size
#         self.pos_per_target = pos_per_target
#         self.rng = np.random.default_rng(seed)
#         self.pos_indices = {col: np.where(df[col].values > 0)[0] for col in target_cols}
#         self.all_indices = np.arange(len(df))
#
#     def __iter__(self):
#         while True:
#             chosen_parts = []
#             for col in self.target_cols:
#                 idxs = self.pos_indices[col]
#                 if len(idxs) == 0:
#                     continue
#                 take = min(self.pos_per_target, len(idxs))
#                 replace_flag = len(idxs) < take
#                 chosen = self.rng.choice(idxs, size=take, replace=replace_flag)
#                 chosen_parts.append(chosen)
#             if chosen_parts:
#                 chosen_all = np.concatenate(chosen_parts)
#             else:
#                 chosen_all = np.array([], dtype=int)
#
#             remaining = self.batch_size - len(chosen_all)
#             if remaining > 0:
#                 neg_part = self.rng.choice(self.all_indices, size=remaining, replace=True)
#                 batch = np.concatenate([chosen_all, neg_part])
#             else:
#                 batch = chosen_all[: self.batch_size]
#
#             self.rng.shuffle(batch)
#             yield batch.tolist()
#
#     def __len__(self):
#         # Tamaño "virtual" de época (no crítico). Escogemos ~N/batch_size
#         return max(1, len(self.all_indices) // self.batch_size)


# -------------------------------
# Modelo Multi-Head Híbrido
# -------------------------------
class MultiOutputRegressorHybrid(nn.Module):
    def __init__(
        self,
        model_name: str,
        n_targets: int,
        hidden_shared: int = 384,
        hidden_head: int = 128,
        dropout: float = 0.1,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_last_n_stages: int = 1,
        disable_shared: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        in_feats = self.backbone.num_features

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for name, p in self.backbone.named_parameters():
                lname = name.lower()
                if any(k in lname for k in ["norm", "bn", "layernorm"]):
                    p.requires_grad = True
            for name, p in self.backbone.named_parameters():
                for k in range(4 - unfreeze_last_n_stages, 4):
                    if f"stages.{k}." in name:
                        p.requires_grad = True

        # Shared (opcional)
        if disable_shared or hidden_shared is None or hidden_shared <= 0:
            self.shared = nn.Identity()
            shared_out_dim = in_feats
        else:
            self.shared = nn.Sequential(
                nn.Linear(in_feats, hidden_shared),
                nn.LayerNorm(hidden_shared),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            shared_out_dim = hidden_shared

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_out_dim, hidden_head),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_head, 1)
            )
            for _ in range(n_targets)
        ])

    def forward(self, x):
        feats = self.backbone(x)
        h = self.shared(feats)
        outs = [head(h) for head in self.heads]
        return torch.cat(outs, dim=1)


# -------------------------------
# Pérdida dinámica (enmascarado de ceros)
# -------------------------------
def build_dynamic_loss_fn(zero_weight: float, lambda_zero: float):
    """
    zero_weight: peso para ceros (0 => ignorados completamente).
    lambda_zero: regularización L2 suave sobre predicciones en ceros (en espacio log1p).
    """
    def loss_fn(preds_tr: torch.Tensor, y_tr: torch.Tensor) -> torch.Tensor:
        mask_pos = (y_tr > 0).float()
        mask_zero = 1.0 - mask_pos
        weights = mask_pos + zero_weight * mask_zero  # shape (B, T)

        abs_err = (preds_tr - y_tr).abs()
        l1 = (abs_err * weights).sum() / weights.sum().clamp(min=1.0)

        if lambda_zero > 0 and mask_zero.any():
            reg_zero = (preds_tr[mask_zero.bool()] ** 2).mean()
            return l1 + lambda_zero * reg_zero
        return l1
    return loss_fn


# -------------------------------
# Métricas por target
# -------------------------------
def coverage_within(y_true: np.ndarray, y_pred: np.ndarray, tol: float = 5.0):
    return float(np.mean(np.abs(y_true - y_pred) <= tol)) if len(y_true) else float("nan")


def mae_top_k(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.10):
    if len(y_true) == 0:
        return float("nan")
    n = len(y_true)
    cut = max(1, int(math.ceil(k * n)))
    idx = np.argsort(y_true)[::-1][:cut]
    return float(np.mean(np.abs(y_true[idx] - y_pred[idx])))


def compute_per_target_metrics(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    target_cols: List[str],
    thresholds_map: Dict[str, float],
) -> Dict:
    metrics = {}
    mae_list, mae_pos_list, rel_mae_list, q90_list, medae_list = [], [], [], [], []
    coverage_pos_list, mae_top10_list = [], []
    r2_list, corr_list = [], []
    eps = 1e-6

    for i, col in enumerate(target_cols):
        yt = y_true_raw[:, i]
        yp = y_pred_raw[:, i]

        mae = mean_absolute_error(yt, yp)
        mae_list.append(mae)

        mask_pos = yt > eps
        if mask_pos.any():
            yt_pos = yt[mask_pos]
            yp_pos = yp[mask_pos]
            mae_pos = mean_absolute_error(yt_pos, yp_pos)
            rel_mae = mae_pos / (yt_pos.mean() + eps)
            medae_pos = median_absolute_error(yt_pos, yp_pos)
            q90 = float(np.percentile(np.abs(yt_pos - yp_pos), 90))
            coverage_pos = coverage_within(yt_pos, yp_pos, tol=5.0)
            mae_top10 = mae_top_k(yt_pos, yp_pos, k=0.10)
        else:
            mae_pos = rel_mae = medae_pos = q90 = coverage_pos = mae_top10 = float("nan")

        mae_pos_list.append(mae_pos)
        rel_mae_list.append(rel_mae)
        medae_list.append(medae_pos)
        q90_list.append(q90)
        coverage_pos_list.append(coverage_pos)
        mae_top10_list.append(mae_top10)

        # r2 global (puede ser muy negativo si varianza baja)
        try:
            r2 = r2_score(yt, yp)
        except Exception:
            r2 = float("nan")
        r2_list.append(r2)

        if np.std(yt) > 1e-6 and np.std(yp) > 1e-6:
            corr = float(np.corrcoef(yt, yp)[0, 1])
        else:
            corr = float("nan")
        corr_list.append(corr)

        thr = thresholds_map.get(col, 0.1)
        pred_pos = yp > thr
        true_pos = yt > thr
        tp = np.logical_and(pred_pos, true_pos).sum()
        fn = np.logical_and(~pred_pos, true_pos).sum()
        fp = np.logical_and(pred_pos, ~true_pos).sum()
        tn = np.logical_and(~pred_pos, ~true_pos).sum()
        recall_pos = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        recall_zero = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        precision_pos = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        f1_pos = (2 * precision_pos * recall_pos / (precision_pos + recall_pos)
                  if precision_pos and recall_pos and (precision_pos + recall_pos) > 0 else np.nan)

        metrics.update({
            f"{col}/mae": mae,
            f"{col}/mae_pos": mae_pos,
            f"{col}/rel_mae_pos": rel_mae,
            f"{col}/medae_pos": medae_pos,
            f"{col}/q90_abs_err_pos": q90,
            f"{col}/coverage_pm5_pos": coverage_pos,
            f"{col}/mae_top10pct_pos": mae_top10,
            f"{col}/r2": r2,
            f"{col}/corr": corr,
            f"{col}/recall_pos_thr{thr}": recall_pos,
            f"{col}/recall_zero_thr{thr}": recall_zero,
            f"{col}/precision_pos_thr{thr}": precision_pos,
            f"{col}/f1_pos_thr{thr}": f1_pos,
        })

    def _nanmean(arr):
        arr2 = [a for a in arr if not (isinstance(a, float) and (math.isnan(a) or math.isinf(a)))]
        return float(np.mean(arr2)) if arr2 else float("nan")

    metrics.update({
        "macro/mae": _nanmean(mae_list),
        "macro/mae_pos": _nanmean(mae_pos_list),
        "macro/rel_mae_pos": _nanmean(rel_mae_list),
        "macro/medae_pos": _nanmean(medae_list),
        "macro/q90_abs_err_pos": _nanmean(q90_list),
        "macro/coverage_pm5_pos": _nanmean(coverage_pos_list),
        "macro/mae_top10pct_pos": _nanmean(mae_top10_list),
        "macro/r2": _nanmean(r2_list),
        "macro/corr": _nanmean(corr_list),
    })
    return metrics


def print_metrics(metrics: Dict):
    macro_keys = [k for k in metrics if k.startswith("macro/")]
    target_keys = [k for k in metrics if not k.startswith("macro/")]
    print("\n=== Métricas por target ===")
    for k in sorted(target_keys):
        v = metrics[k]
        if isinstance(v, float):
            print(f"{k}: {v:.4f}" if not math.isnan(v) else f"{k}: nan")
        else:
            print(f"{k}: {v}")
    print("=== Métricas macro ===")
    for k in sorted(macro_keys):
        v = metrics[k]
        if isinstance(v, float):
            print(f"{k}: {v:.4f}" if not math.isnan(v) else f"{k}: nan")
        else:
            print(f"{k}: {v}")


def log_pred_stats_multi(tag: str, y_pred_raw: np.ndarray, y_true_raw: np.ndarray,
                         target_cols: List[str], step: int):
    import mlflow
    agg = {}
    for i, col in enumerate(target_cols):
        pred = y_pred_raw[:, i]
        true = y_true_raw[:, i]
        stats = {
            "min": float(pred.min()),
            "p5": float(np.percentile(pred, 5)),
            "median": float(np.percentile(pred, 50)),
            "mean": float(pred.mean()),
            "p95": float(np.percentile(pred, 95)),
            "max": float(pred.max()),
            "std": float(pred.std()),
            "zero_ratio": float((pred <= 1e-6).mean()),
        }
        for k, v in stats.items():
            mlflow.log_metric(f"{tag}/pred/{col}/{k}", v, step=step)
        # True dist
        mlflow.log_metric(f"{tag}/true/{col}/mean", float(true.mean()), step=step)
        mlflow.log_metric(f"{tag}/true/{col}/std", float(true.std()), step=step)
        mlflow.log_metric(f"{tag}/true/{col}/p95", float(np.percentile(true, 95)), step=step)
        mlflow.log_metric(f"{tag}/true/{col}/zero_ratio", float((true <= 1e-6).mean()), step=step)

        for k, v in stats.items():
            agg.setdefault(k, []).append(v)

    # macro
    for k, vals in agg.items():
        mlflow.log_metric(f"{tag}/pred/macro/{k}", float(np.mean(vals)), step=step)


def run_epoch(loader: DataLoader,
              model: nn.Module,
              criterion,
              device: torch.device,
              optimizer=None,
              target_cols: List[str] = None,
              thresholds_map: Dict[str, float] = None,
              global_step: int = 0,
              log_pos_counts: bool = False):
    training = optimizer is not None
    model.train() if training else model.eval()
    losses = []
    all_true_tr, all_pred_tr = [], []
    with torch.set_grad_enabled(training):
        for images, y_tr, _ in tqdm(loader, desc="Batches", leave=False):
            images = images.to(device, non_blocking=True)
            y_tr = y_tr.to(device)
            preds_tr = model(images)
            loss = criterion(preds_tr, y_tr)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if training and log_pos_counts:
                try:
                    import mlflow

                    # Máscara de positivos (log1p > 0 implica original > 0)
                    pos_mask: torch.Tensor = (y_tr > 0)  # (B, T) bool
                    batch_pos_counts = torch.count_nonzero(pos_mask, dim=0)  # (T,)
                    batch_size_actual = y_tr.size(0)
                    batch_pos_ratio = batch_pos_counts.float() / batch_size_actual

                    for i, col in enumerate(target_cols):
                        mlflow.log_metric(
                            f"train/pos_count/{col}",
                            int(batch_pos_counts[i].item()),
                            step=global_step
                        )
                        mlflow.log_metric(
                            f"train/pos_ratio/{col}",
                            float(batch_pos_ratio[i].item()),
                            step=global_step
                        )

                        # Falsos positivos aproximados: pred_tr > 0 donde label == 0
                        preds_pos_mask = preds_tr[:, i] > 0  # (B,) bool
                        true_zero_mask = ~pos_mask[:, i]  # (B,) bool
                        fp_mask = preds_pos_mask & true_zero_mask
                        fp_like = torch.count_nonzero(fp_mask)
                        mlflow.log_metric(
                            f"train/fp_like_count/{col}",
                            int(fp_like.item()),
                            step=global_step
                        )

                    # Co-ocurrencia por muestra
                    pos_per_sample = torch.count_nonzero(pos_mask, dim=1)  # (B,)
                    mlflow.log_metric(
                        "train/avg_pos_targets_per_sample",
                        float(pos_per_sample.float().mean().item()),
                        step=global_step
                    )
                    mlflow.log_metric(
                        "train/median_pos_targets_per_sample",
                        float(pos_per_sample.float().median().item()),
                        step=global_step
                    )
                    mlflow.log_metric(
                        "train/num_targets_without_positives",
                        int(torch.count_nonzero(batch_pos_counts == 0).item()),
                        step=global_step
                    )
                    mlflow.log_metric(
                        "train/num_samples_all_zero_targets",
                        int(torch.count_nonzero(pos_per_sample == 0).item()),
                        step=global_step
                    )
                    mlflow.log_metric(
                        "train/num_samples_multi_positive",
                        int(torch.count_nonzero(pos_per_sample > 1).item()),
                        step=global_step
                    )
                except Exception:
                    pass
                global_step += 1

            losses.append(loss.item())
            all_true_tr.append(y_tr.detach().cpu())
            all_pred_tr.append(preds_tr.detach().cpu())
    avg_loss = float(np.mean(losses)) if losses else float("nan")
    y_true_tr = torch.cat(all_true_tr, dim=0)
    y_pred_tr = torch.cat(all_pred_tr, dim=0)
    y_true_raw = inverse_transform(y_true_tr).numpy()
    y_pred_raw = inverse_transform(y_pred_tr).numpy()
    metrics = compute_per_target_metrics(y_true_raw, y_pred_raw, target_cols, thresholds_map)
    metrics["loss_raw_space"] = avg_loss
    return metrics, y_true_raw, y_pred_raw, global_step


def evaluate_zero_baseline(df: pd.DataFrame, target_cols: List[str], thresholds_map: Dict[str, float]) -> Dict:
    metrics = {}
    eps = 1e-6
    mae_pos_list, rel_mae_list, medae_list, q90_list = [], [], [], []
    coverage_pos_list, mae_top10_list = [], []
    mae_list, r2_list, corr_list = [], [], []
    for col in target_cols:
        yt = df[col].values.astype("float32")
        yp = np.zeros_like(yt)

        mae = mean_absolute_error(yt, yp)
        mae_list.append(mae)

        mask_pos = yt > eps
        if mask_pos.any():
            yt_pos = yt[mask_pos]
            yp_pos = yp[mask_pos]
            mae_pos = mean_absolute_error(yt_pos, yp_pos)
            rel_mae = mae_pos / (yt_pos.mean() + eps)
            medae_pos = median_absolute_error(yt_pos, yp_pos)
            q90 = float(np.percentile(np.abs(yt_pos - yp_pos), 90))
            coverage_pos = coverage_within(yt_pos, yp_pos, tol=5.0)
            mae_top10 = mae_top_k(yt_pos, yp_pos, 0.10)
        else:
            mae_pos = rel_mae = medae_pos = q90 = coverage_pos = mae_top10 = float("nan")

        mae_pos_list.append(mae_pos)
        rel_mae_list.append(rel_mae)
        medae_list.append(medae_pos)
        q90_list.append(q90)
        coverage_pos_list.append(coverage_pos)
        mae_top10_list.append(mae_top10)

        try:
            r2 = r2_score(yt, yp)
        except Exception:
            r2 = float("nan")
        r2_list.append(r2)

        # Corr trivial (todo 0) => nan
        corr_list.append(float("nan"))

        thr = thresholds_map.get(col, 0.1)
        pred_pos = yp > thr
        true_pos = yt > thr
        tp = np.logical_and(pred_pos, true_pos).sum()
        fn = np.logical_and(~pred_pos, true_pos).sum()
        fp = np.logical_and(pred_pos, ~true_pos).sum()
        tn = np.logical_and(~pred_pos, ~true_pos).sum()
        recall_pos = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        recall_zero = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        metrics.update({
            f"{col}/mae": mae,
            f"{col}/mae_pos": mae_pos,
            f"{col}/rel_mae_pos": rel_mae,
            f"{col}/medae_pos": medae_pos,
            f"{col}/q90_abs_err_pos": q90,
            f"{col}/coverage_pm5_pos": coverage_pos,
            f"{col}/mae_top10pct_pos": mae_top10,
            f"{col}/r2": r2,
            f"{col}/recall_pos_thr{thr}": recall_pos,
            f"{col}/recall_zero_thr{thr}": recall_zero,
        })

    def _nanmean(a):
        b = [x for x in a if not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))]
        return float(np.mean(b)) if b else float("nan")

    metrics.update({
        "macro/mae": _nanmean(mae_list),
        "macro/mae_pos": _nanmean(mae_pos_list),
        "macro/rel_mae_pos": _nanmean(rel_mae_list),
        "macro/medae_pos": _nanmean(medae_list),
        "macro/q90_abs_err_pos": _nanmean(q90_list),
        "macro/coverage_pm5_pos": _nanmean(coverage_pos_list),
        "macro/mae_top10pct_pos": _nanmean(mae_top10_list),
        "macro/r2": _nanmean(r2_list),
    })
    return metrics


def mlflow_log_metrics(step: int, metrics: Dict, prefix: str = ""):
    import mlflow
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            mlflow.log_metric(prefix + k, float(v), step=step)


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4721, 0.4410, 0.3985],
                             std=[0.2312, 0.2230, 0.2147])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4721, 0.4410, 0.3985],
                             std=[0.2312, 0.2230, 0.2147])
    ])
    return train_tf, val_tf


def metrics_by_percentiles(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: List[str],
    thresholds_map: Dict[str, float],
    n_bins: int = 10
) -> pd.DataFrame:
    from scipy.stats import spearmanr
    rows = []
    eps = 1e-6
    for i, col in enumerate(target_cols):
        yt_full = y_true[:, i]
        yp_full = y_pred[:, i]
        thr = thresholds_map.get(col, 0.1)

        # Bin cero
        mask_zero = yt_full <= eps
        if mask_zero.any():
            yt_bin = yt_full[mask_zero]; yp_bin = yp_full[mask_zero]
            mae = mean_absolute_error(yt_bin, yp_bin)
            medae_val = median_absolute_error(yt_bin, yp_bin)
            q90 = float(np.percentile(np.abs(yt_bin - yp_bin), 90)) if len(yt_bin) else float("nan")
            pred_pos = yp_bin > thr
            true_pos = yt_bin > thr
            tp = np.logical_and(pred_pos, true_pos).sum()
            fn = np.logical_and(~pred_pos, true_pos).sum()
            fp = np.logical_and(pred_pos, ~true_pos).sum()
            tn = np.logical_and(~pred_pos, ~true_pos).sum()
            recall_pos = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            recall_zero = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            precision_pos = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            f1_pos = (2 * precision_pos * recall_pos / (precision_pos + recall_pos)
                      if precision_pos and recall_pos and (precision_pos + recall_pos) > 0 else np.nan)
            rows.append({
                "target": col,
                "bin_type": "zero",
                "bin_start": 0.0,
                "bin_end": 0.0,
                "threshold": thr,
                "n_samples": int(mask_zero.sum()),
                "mean_true": float(yt_bin.mean()),
                "mean_pred": float(yp_bin.mean()),
                "bias": float(yp_bin.mean() - yt_bin.mean()),
                "mae": mae,
                "medae": medae_val,
                "rel_mae": float("nan"),
                "nrm_mae": float("nan"),
                "q90_abs_err": q90,
                "spearman": float("nan"),
                "recall_pos": recall_pos,
                "recall_zero": recall_zero,
                "precision_pos": precision_pos,
                "f1_pos": f1_pos,
                "prevalence": (tp + fn) / max(len(yt_bin), 1),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            })

        # Bins positivos
        mask_pos = yt_full > eps
        if not mask_pos.any():
            continue
        yt_pos = yt_full[mask_pos]
        yp_pos = yp_full[mask_pos]
        percentiles = np.linspace(0, 100, n_bins + 1)
        cuts = np.unique(np.percentile(yt_pos, percentiles))
        if len(cuts) <= 1:
            continue

        for b in range(len(cuts) - 1):
            start, end = cuts[b], cuts[b + 1]
            m_bin = (yt_pos >= start) & (yt_pos <= end)
            if not m_bin.any():
                continue
            yt_bin = yt_pos[m_bin]; yp_bin = yp_pos[m_bin]
            bin_range = end - start
            mae = mean_absolute_error(yt_bin, yp_bin)
            medae_val = median_absolute_error(yt_bin, yp_bin)
            rel_mae = mae / (yt_bin.mean() + eps)
            nrm_mae = mae / (bin_range + eps) if bin_range > 0 else float("nan")
            q90 = float(np.percentile(np.abs(yt_bin - yp_bin), 90))
            if len(yt_bin) >= 8 and np.std(yt_bin) > 1e-6 and np.std(yp_bin) > 1e-6:
                try:
                    spearman_val = float(spearmanr(yt_bin, yp_bin).correlation)
                except Exception:
                    spearman_val = float("nan")
            else:
                spearman_val = float("nan")
            pred_pos = yp_bin > thr
            true_pos = yt_bin > thr
            tp = np.logical_and(pred_pos, true_pos).sum()
            fn = np.logical_and(~pred_pos, true_pos).sum()
            fp = np.logical_and(pred_pos, ~true_pos).sum()
            tn = np.logical_and(~pred_pos, ~true_pos).sum()
            recall_pos = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            recall_zero = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            precision_pos = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            f1_pos = (2 * precision_pos * recall_pos / (precision_pos + recall_pos)
                      if precision_pos and recall_pos and (precision_pos + recall_pos) > 0 else np.nan)
            rows.append({
                "target": col,
                "bin_type": "positive",
                "bin_start": float(start),
                "bin_end": float(end),
                "threshold": thr,
                "n_samples": int(m_bin.sum()),
                "mean_true": float(yt_bin.mean()),
                "mean_pred": float(yp_bin.mean()),
                "bias": float(yp_bin.mean() - yt_bin.mean()),
                "mae": mae,
                "medae": medae_val,
                "rel_mae": rel_mae,
                "nrm_mae": nrm_mae,
                "q90_abs_err": q90,
                "spearman": spearman_val,
                "recall_pos": recall_pos,
                "recall_zero": recall_zero,
                "precision_pos": precision_pos,
                "f1_pos": f1_pos,
                "prevalence": (tp + fn) / max(len(yt_bin), 1),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            })
    return pd.DataFrame(rows)


# -------------------------------
# MLflow
# -------------------------------
def maybe_init_mlflow(args, target_cols):
    if not args.mlflow:
        return None
    import mlflow
    jira_ticket_url = MLFlowNamer.define_jira_ticket_url(args.ticket) if args.ticket else None
    username, password, url = Secrets(aws_profile=None).get_mlflow_user()
    setup_mlflow(username, password, url, args.mlflow_experiment)
    run = mlflow.start_run(run_name=args.mlflow_run_name)
    mlflow.log_params({
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "img_size": args.img_size,
        "targets": ",".join(target_cols),
        "freeze_backbone": not args.no_freeze_backbone,
        "shared_dim": args.shared_dim,
        "head_hidden_dim": args.head_hidden_dim,
        "dropout": args.dropout,
        "zero_warmup_epochs": args.zero_warmup_epochs,
        "zero_weight": args.zero_weight,
        "lambda_zero_reg": args.lambda_zero_reg,
        "unfreeze_last_n_stages": args.unfreeze_last_n_stages,
        "pos_per_target": args.pos_per_target,
        "early_stopping_patience": args.early_stopping_patience,
    })
    if jira_ticket_url:
        mlflow.set_tag("jira_ticket", jira_ticket_url)
    return run


# -------------------------------
# Main
# -------------------------------
def main():
    set_seeds(45)
    parser = argparse.ArgumentParser(description="Multi-Head multi-output regression (log1p) con enmascarado de ceros.")
    data_src = parser.add_mutually_exclusive_group(required=True)
    data_src.add_argument("--geopackage", type=str, help="Ruta .gpkg")
    data_src.add_argument("--csv", type=str, help="Ruta CSV")

    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--model-name", type=str, default="convnext_tiny")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--shared-dim", type=int, default=384)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-last-n-stages", type=int, default=1)
    parser.add_argument("--zero-warmup-epochs", type=int, default=3)
    parser.add_argument("--zero-weight", type=float, default=0.05)
    parser.add_argument("--lambda-zero-reg", type=float, default=1e-3)
    parser.add_argument("--pos-per-target", type=int, default=0,
                        help="N positivos por target en cada batch (0 desactiva sampler balanceado)")
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--model-save", type=str, default="multihead_best.pth")
    parser.add_argument("--percentile-bins", type=int, default=12)
    parser.add_argument("--image-base-dir", type=str)
    parser.add_argument("--ticket", type=str, default=None)
    parser.add_argument("--no-shared", action="store_true",
                        help="Desactiva capa fully-connected compartida (usa directamente features del backbone).")

    # MLflow
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", type=str, default="multioutput/multihead")
    parser.add_argument("--mlflow-run-name", type=str)

    args = parser.parse_args()
    target_cols = args.targets
    freeze_backbone = (not args.no_freeze_backbone)

    thresholds_map = validate_thresholds(target_cols, THRESHOLDS)

    if args.mlflow and args.mlflow_run_name is None:
        args.mlflow_run_name = (
            f"MultiHead/{args.model_name}/lr={args.lr}/freeze={freeze_backbone}/zw={args.zero_weight}"
        )

    # Carga de datos
    if args.geopackage:
        if gpd is None:
            raise RuntimeError("geopandas no disponible.")
        layers = fiona.listlayers(args.geopackage)
        gdfs = []
        for lyr in layers:
            g = gpd.read_file(args.geopackage, layer=lyr)
            g["__source_layer"] = lyr
            gdfs.append(g)
        df = pd.concat(gdfs, ignore_index=True)
    else:
        df = pd.read_csv(args.csv)

    if ("image_path" not in df.columns) or df["image_path"].fillna("").eq("").all():
        if not args.image_base_dir:
            raise ValueError("Necesitas --image-base-dir para construir rutas de imagen.")
        df["image_path"] = (
            args.image_base_dir.rstrip("/") + "/" +
            df["split"].astype(str) + "/" +
            df.get("child_aoi", "").astype(str) + "/" +
            df["id"].astype(str) + ".png"
        )

    required_cols = {"split", "image_path"} | set(target_cols)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")
    print("Targets:", target_cols)

    zero_val_metrics = evaluate_zero_baseline(val_df, target_cols, thresholds_map)
    print("\nBaseline cero (val):")
    print_metrics(zero_val_metrics)

    train_tf, val_tf = build_transforms(args.img_size)
    train_ds = MultiTargetImageDataset(train_df, target_cols, transform=train_tf)
    val_ds = MultiTargetImageDataset(val_df, target_cols, transform=val_tf)

    if args.pos_per_target > 0:
        print(f"[INFO] Usando BalancedPositiveBatchSampler con pos_per_target={args.pos_per_target}")
        # train_loader = DataLoader(
        #     train_ds,
        #     batch_sampler=BalancedPositiveBatchSampler(
        #         train_df, target_cols, batch_size=args.batch_size, pos_per_target=args.pos_per_target
        #     ),
        #     num_workers=args.num_workers,
        #     pin_memory=True
        # )
        raise NotImplementedError
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = MultiOutputRegressorHybrid(
        model_name=args.model_name,
        n_targets=len(target_cols),
        hidden_shared=args.shared_dim,
        hidden_head=args.head_hidden_dim,
        dropout=args.dropout,
        pretrained=True,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n_stages=args.unfreeze_last_n_stages,
        disable_shared=args.no_shared
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,} | Entrenables: {trainable_params:,} "
          f"({100*trainable_params/total_params:.2f}%)")

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    mlflow_run = maybe_init_mlflow(args, target_cols)

    if args.mlflow:
        import mlflow
        mlflow_log_metrics(0, {f"baseline_zero/val/{k}": v for k, v in zero_val_metrics.items()})
        mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=0)

    best_score = float("inf")
    best_state = None
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # Construir loss dinámicamente (warmup => zero_weight=0)
        if epoch <= args.zero_warmup_epochs:
            zw = 0.0
        else:
            zw = args.zero_weight
        criterion = build_dynamic_loss_fn(zero_weight=zw, lambda_zero=args.lambda_zero_reg)

        print(f"\n==== Epoch {epoch}/{args.epochs} (zero_weight={zw}) ====")
        train_metrics, y_true_raw_train, y_pred_raw_train, global_step = run_epoch(
            train_loader, model, criterion, device, optimizer,
            target_cols, thresholds_map,
            global_step=global_step,
            log_pos_counts=False  # Activa logging
        )
        val_metrics, y_true_raw_val, y_pred_raw_val, _ = run_epoch(
            val_loader, model, criterion, device, None,
            target_cols, thresholds_map,
            global_step=global_step,
            log_pos_counts=False
        )

        # Métrica de selección: macro/mae_pos si existe, fallback macro/mae
        selection_metric = val_metrics.get("macro/mae_pos", val_metrics["macro/mae"])

        print("\n-- Train Metrics --")
        print_metrics(train_metrics)
        print("\n-- Val Metrics --")
        print_metrics(val_metrics)

        if args.mlflow:
            mlflow_log_metrics(epoch, train_metrics, prefix="train/")
            mlflow_log_metrics(epoch, val_metrics, prefix="val/")
            log_pred_stats_multi("train", y_pred_raw_train, y_true_raw_train, target_cols, step=epoch)
            log_pred_stats_multi("val", y_pred_raw_val, y_true_raw_val, target_cols, step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)
            mlflow.log_metric("zero_weight_effective", zw, step=epoch)

        if selection_metric < best_score:
            best_score = selection_metric
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_selection_metric": best_score,
                "args": vars(args),
                "target_cols": target_cols,
                "thresholds_map": thresholds_map
            }
            torch.save(best_state, args.model_save)
            print(f"*** Nuevo mejor modelo guardado ({args.model_save}) sel_metric={best_score:.4f}")
            if args.mlflow:
                import mlflow
                mlflow.log_artifact(args.model_save, artifact_path="model")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step(selection_metric)

        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping (sin mejora {args.early_stopping_patience} epochs).")
            break

    if best_state:
        model.load_state_dict(best_state["model"])

    print("\n==== Evaluación final (validación) ====")

    final_metrics, y_true_raw_val, y_pred_raw_val, _ = run_epoch(
        val_loader,
        model,
        build_dynamic_loss_fn(zero_weight=args.zero_weight, lambda_zero=args.lambda_zero_reg),
        device,
        None,
        target_cols, thresholds_map,
        global_step=global_step,
        log_pos_counts=False
    )

    print_metrics(final_metrics)

    delta_macro_mae = zero_val_metrics["macro/mae"] - final_metrics["macro/mae"]
    final_metrics["delta_vs_zero/macro_mae"] = delta_macro_mae
    print(f"\nDelta macro/mae vs baseline cero: {delta_macro_mae:.4f}")

    # Guardar predicciones
    val_pred_df = pd.DataFrame({"image_path": val_df["image_path"].values})
    for i, col in enumerate(target_cols):
        val_pred_df[f"{col}_true"] = y_true_raw_val[:, i]
        val_pred_df[f"{col}_pred"] = y_pred_raw_val[:, i]
    pred_csv = "val_predictions_multihead.csv"
    val_pred_df.to_csv(pred_csv, index=False)

    # Ordenar por macro error (promedio absoluto por fila)
    abs_errors = [np.abs(y_true_raw_val[:, i] - y_pred_raw_val[:, i]) for i in range(len(target_cols))]
    macro_abs_error = np.mean(np.stack(abs_errors, axis=1), axis=1)
    error_df = val_pred_df.copy()
    error_df["macro_abs_error"] = macro_abs_error
    for i, col in enumerate(target_cols):
        error_df[f"{col}_abs_error"] = abs_errors[i]
    error_df = error_df.sort_values("macro_abs_error", ascending=False).reset_index(drop=True)
    error_csv = "val_predictions_by_error.csv"
    error_df.to_csv(error_csv, index=False)

    # Métricas por percentiles
    df_bin_metrics = metrics_by_percentiles(
        y_true=y_true_raw_val,
        y_pred=y_pred_raw_val,
        target_cols=target_cols,
        thresholds_map=thresholds_map,
        n_bins=args.percentile_bins
    )
    bin_csv = "val_metrics_by_percentile.csv"
    df_bin_metrics.to_csv(bin_csv, index=False)

    print(f"Predicciones guardadas en: {pred_csv}")
    print(f"Predicciones ordenadas por error en: {error_csv}")
    print(f"Métricas por severidad en: {bin_csv}")

    if args.mlflow:
        import mlflow
        mlflow_log_metrics(0, {f"val_final/{k}": v for k, v in final_metrics.items()})
        log_pred_stats_multi("val_final", y_pred_raw_val, y_true_raw_val, target_cols, step=0)
        mlflow.log_artifact(pred_csv, artifact_path="predictions")
        mlflow.log_artifact(error_csv, artifact_path="predictions")
        mlflow.log_artifact(bin_csv, artifact_path="predictions")
        mlflow.end_run()

    print("\nListo. Entrenamiento multi-head completado.")


if __name__ == "__main__":
    main()