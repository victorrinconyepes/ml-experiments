import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import label, distance_transform_edt

# ===========================
# Métricas
# ===========================
def iou_score(preds, targets, threshold=0.5, logits=True):
    """Compute mean Intersection over Union (IoU) over a batch.

    Parameters
    - preds: torch.Tensor of shape (B, 1, H, W), model outputs (logits or probabilities).
    - targets: torch.Tensor of shape (B, 1, H, W), binary ground-truth masks {0,1}.
    - threshold: float, threshold for binarizing predictions if probs given.
    - logits: bool, whether preds are logits; if True, apply sigmoid first.

    Returns
    - float: mean IoU over the batch.
    """
    if logits: preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = ((preds + targets) > 0).sum(dim=(1,2,3))
    return (intersection / (union + 1e-8)).mean().item()

def dice_score(preds, targets, threshold=0.5, logits=True):
    """Compute mean Dice coefficient over a batch.

    Parameters
    - preds: torch.Tensor (B, 1, H, W), model outputs (logits or probabilities).
    - targets: torch.Tensor (B, 1, H, W), binary ground-truth masks {0,1}.
    - threshold: float, threshold for binarizing predictions if probs given.
    - logits: bool, whether preds are logits; if True, apply sigmoid first.

    Returns
    - float: mean Dice score over the batch.
    """
    if logits: preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    total = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    return (2 * intersection / (total + 1e-8)).mean().item()

def seg_precision_recall(preds, targets, threshold=0.5, logits=True):
    """Compute pixel-wise precision and recall for a batch.

    Parameters
    - preds: torch.Tensor (B, 1, H, W), model outputs (logits or probabilities).
    - targets: torch.Tensor (B, 1, H, W), binary ground-truth masks {0,1}.
    - threshold: float, threshold for binarizing predictions if probs given.
    - logits: bool, whether preds are logits; if True, apply sigmoid first.

    Returns
    - (precision, recall): tuple of floats.
    """
    if logits:
        preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return precision.item(), recall.item()

def find_optimal_threshold(y_true, y_probs):
    """Find classification threshold maximizing F1 on a PR curve.

    Parameters
    - y_true: array-like of shape (N,), binary labels {0,1}.
    - y_probs: array-like of shape (N,), predicted probabilities in [0,1].

    Returns
    - (best_thr, best_f1): threshold and corresponding F1 score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    f1_scores = f1_scores[:-1]
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def find_optimal_iou_threshold(y_true, y_pred, thresholds=np.linspace(0.1, 0.9, 9)):
    """
    Encuentra el threshold que maximiza IoU dado un batch completo.
    - y_true: tensor (B, 1, H, W) ground truth binario.
    - y_pred: tensor (B, 1, H, W) probabilidades [0,1].
    """
    best_thr = 0.5
    best_iou = 0.0

    # pasar a CPU/numpy si hace falta
    y_true_np = y_true.detach().cpu().numpy().astype(np.uint8)
    y_pred_np = y_pred.detach().cpu().numpy()

    for thr in thresholds:
        preds = (y_pred_np > thr).astype(np.uint8)
        inter = (preds & y_true_np).sum()
        union = (preds | y_true_np).sum()
        iou = inter / (union + 1e-7)

        if iou > best_iou:
            best_iou = iou
            best_thr = thr

    return best_thr, best_iou


def iou(mask1, mask2):
    """Compute IoU between two binary NumPy masks.

    Parameters
    - mask1, mask2: np.ndarray (H,W) bool or {0,1}.

    Returns
    - float: IoU value in [0,1].
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + 1e-8)

def hit_or_miss_iou(gt_mask, pred_mask, iou_thresh=0.1):
    """Hit-or-miss metric based on component-level IoU matching.

    A GT object is considered hit if any predicted component overlaps it with IoU > iou_thresh.
    The score is hits / number_of_gt_objects (or 1.0 if no GT objects).

    Parameters
    - gt_mask: np.ndarray (H,W), binary ground-truth mask.
    - pred_mask: np.ndarray (H,W), binary predicted mask.
    - iou_thresh: float, IoU threshold for a hit.

    Returns
    - float: hit-or-miss score in [0,1].
    """
    gt_labels, n_gt = label(gt_mask)
    pred_labels, n_pred = label(pred_mask)
    hits = 0
    for i in range(1, n_gt+1):
        real_obj = (gt_labels == i)
        for j in range(1, n_pred+1):
            pred_obj = (pred_labels == j)
            if iou(real_obj, pred_obj) > iou_thresh:
                hits += 1
                break
    return hits / n_gt if n_gt > 0 else 1.0


def oversampling_objects(gt_mask, pred_mask, iou_thresh=0.1):
    """Measure fraction of predicted components not matching any GT sufficiently.

    A predicted component is counted as oversampled if its best IoU with any GT object is < iou_thresh.

    Parameters
    - gt_mask: np.ndarray (H,W), binary ground-truth mask.
    - pred_mask: np.ndarray (H,W), binary predicted mask.
    - iou_thresh: float, IoU threshold to consider a prediction matched.

    Returns
    - float: oversampling ratio in [0,1]; 0 if no predictions.
    """
    gt_labels, n_gt = label(gt_mask)
    pred_labels, n_pred = label(pred_mask)

    if n_pred == 0:
        return 0.0  # no hay predicciones → no hay sobremuestreo

    oversampled = 0
    for j in range(1, n_pred + 1):
        pred_obj = (pred_labels == j)
        max_iou = 0.0
        for i in range(1, n_gt + 1):
            gt_obj = (gt_labels == i)
            max_iou = max(max_iou, iou(pred_obj, gt_obj))
        # si el mejor solape es < (1 - over_thresh), consideramos sobremuestreo
        if max_iou < iou_thresh:
            oversampled += 1

    return oversampled / n_pred


def coverage(gt_mask, pred_mask):
    """Compute pixel coverage: fraction of GT pixels covered by prediction.

    Parameters
    - gt_mask: np.ndarray (H,W), binary ground-truth mask.
    - pred_mask: np.ndarray (H,W), binary predicted mask.

    Returns
    - float: coverage ratio in [0,1] (1.0 if no GT pixels).
    """
    real_pixels = gt_mask > 0
    pred_pixels = pred_mask > 0
    covered_pixels = np.logical_and(real_pixels, pred_pixels).sum()
    total_pixels = real_pixels.sum()
    return covered_pixels / total_pixels if total_pixels > 0 else 1.0

def false_positive_rate(gt_mask, pred_mask):
    """Compute the fraction of predicted pixels that are false positives.

    Parameters
    - gt_mask: np.ndarray (H,W), binary ground-truth mask.
    - pred_mask: np.ndarray (H,W), binary predicted mask.

    Returns
    - float: FP / predicted positives (0.0 if no predicted positives).
    """
    pred_pixels = pred_mask > 0
    real_pixels = gt_mask > 0
    false_positives = np.logical_and(pred_pixels, np.logical_not(real_pixels)).sum()
    total_predicted = pred_pixels.sum()
    fp_rate = false_positives / total_predicted if total_predicted > 0 else 0.0
    return fp_rate

def average_minimum_distance(mask_gt, mask_pred):
    """Average minimum distance between GT and prediction boundaries via EDT.

    Computes mean of distances from predicted positives to nearest GT positive and vice versa,
    using Euclidean Distance Transform (EDT) on inverse masks.

    Parameters
    - mask_gt: np.ndarray (H,W), binary ground-truth mask.
    - mask_pred: np.ndarray (H,W), binary predicted mask.

    Returns
    - float: symmetric average minimum distance (pixels); inf if one side has no positives.
    """
    mask_gt = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)
    dt_gt = distance_transform_edt(~mask_gt)
    dt_pred = distance_transform_edt(~mask_pred)

    if mask_pred.sum() > 0:
        avg_pred_to_gt = dt_gt[mask_pred].mean()
    else:
        avg_pred_to_gt = np.inf
    if mask_gt.sum() > 0:
        avg_gt_to_pred = dt_pred[mask_gt].mean()
    else:
        avg_gt_to_pred = np.inf
    return (avg_pred_to_gt + avg_gt_to_pred) / 2

def pixel_recall(gt_mask, pred_mask):
    """Pixel-wise recall between two binary masks.

    Parameters
    - gt_mask: np.ndarray (H,W), binary ground-truth mask.
    - pred_mask: np.ndarray (H,W), binary predicted mask.

    Returns
    - float or np.nan: TP/(TP+FN); nan if GT has no positives.
    """
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    tp = np.logical_and(gt_mask, pred_mask).sum()
    fn = np.logical_and(gt_mask, np.logical_not(pred_mask)).sum()
    if (tp + fn) == 0:
        return np.nan
    return tp / (tp + fn)

def pixel_precision(gt_mask, pred_mask):
    """Pixel-wise precision between two binary masks.

    Parameters
    - gt_mask: np.ndarray (H,W), binary ground-truth mask.
    - pred_mask: np.ndarray (H,W), binary predicted mask.

    Returns
    - float or np.nan: TP/(TP+FP); nan if prediction has no positives.
    """
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    tp = np.logical_and(gt_mask, pred_mask).sum()
    fp = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()
    if (tp + fp) == 0:
        return np.nan
    return tp / (tp + fp)

def pixel_f1(gt_mask, pred_mask):
    """Pixel-wise F1 score between two binary masks.

    Returns nan if precision or recall are undefined (e.g., no positives).

    Parameters
    - gt_mask: np.ndarray (H,W), binary ground-truth mask.
    - pred_mask: np.ndarray (H,W), binary predicted mask.

    Returns
    - float or np.nan: harmonic mean of precision and recall.
    """
    rec = pixel_recall(gt_mask, pred_mask)
    prec = pixel_precision(gt_mask, pred_mask)
    if np.isnan(rec) or np.isnan(prec) or (rec + prec) == 0:
        return np.nan
    return 2 * (prec * rec) / (prec + rec)

def pick_threshold_for_precision(y_true_pix, y_prob_pix, target_precision=0.60):
    """Pick the lowest threshold that achieves a target pixel precision on the PR curve.

    Fallback to the threshold maximizing F1 if no point reaches the target precision.

    Parameters
    - y_true_pix: array-like (N,), binary pixel labels concatenated across images.
    - y_prob_pix: array-like (N,), predicted pixel probabilities in [0,1].
    - target_precision: float in (0,1], desired precision.

    Returns
    - float: selected threshold value in [0,1].
    """
    prec, rec, thr = precision_recall_curve(y_true_pix, y_prob_pix)

    # Solo consideramos hasta el penúltimo elemento, para que coincida con 'thr'
    valid = np.where(prec[:-1] >= target_precision)[0]
    if len(valid) > 0:
        # Nos quedamos con el threshold más bajo que cumple
        return float(thr[valid[0]])

    # optimizaacion por F1
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    f1 = f1[:-1]  # alinear con thr
    return float(thr[np.argmax(f1)])

def find_optimal_pixel_f1_threshold(y_true_masks, y_pred_probs):
    """Find the pixel threshold maximizing F1 from concatenated masks.

    Parameters
    - y_true_masks: list/iterable of np.ndarray (H,W), binary GT masks.
    - y_pred_probs: list/iterable of np.ndarray (H,W), predicted probabilities in [0,1].

    Returns
    - (best_thr, best_f1): tuple of floats with threshold and maximum pixel F1.
    """
    y_true = np.concatenate([m.ravel() for m in y_true_masks]).astype(np.uint8)
    y_prob = np.concatenate([p.ravel() for p in y_pred_probs]).astype(np.float32)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    f1 = f1[:-1]
    return float(thr[np.argmax(f1)]), float(f1.max())