# ===========================
# Inference y métricas (con puerta de clasificación + post-procesado)
# ===========================
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from damage.guided_segmentation.custom_metrics import hit_or_miss_iou, coverage, false_positive_rate, \
    average_minimum_distance, pixel_recall, pixel_precision, pixel_f1, oversampling_objects
from scipy.ndimage import label, binary_opening
from typing import Tuple, Dict, Any


def _save_triplet_panel(img_rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, out_path: str,
                        titles=("Imagen", "GT", "Predicción")):
    """
    img_rgb: (H,W,3) uint8
    gt_mask, pred_mask: (H,W) float/bool en [0..1]
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1,3,1); plt.imshow(img_rgb); plt.axis("off"); plt.title(titles[0])
    plt.subplot(1,3,2); plt.imshow(gt_mask, cmap="gray", vmin=0, vmax=1); plt.axis("off"); plt.title(titles[1])
    plt.subplot(1,3,3); plt.imshow(pred_mask, cmap="gray", vmin=0, vmax=1); plt.axis("off"); plt.title(titles[2])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

def _denormalize_image(img_chw: torch.Tensor, mean, std) -> np.ndarray:
    """
    img_chw: tensor (3,H,W) normalizado
    mean, std: listas/tuplas de 3 floats (como params['mean'], params['std'])
    return: np.uint8 (H,W,3) en [0,255]
    """
    if isinstance(mean, (list, tuple)):
        mean = np.array(mean, dtype=np.float32)
    if isinstance(std, (list, tuple)):
        std = np.array(std, dtype=np.float32)
    x = img_chw.detach().cpu().float().numpy()  # (C,H,W)
    x = (x * std[:, None, None]) + mean[:, None, None]
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).round().astype(np.uint8)    # (C,H,W)
    return np.transpose(x, (1, 2, 0))           # (H,W,3)

def postprocess_mask(prob, thr, keep_top=1, min_area=64, prob_weighted=True, apply_open=False, opening_iter=1):
    """
    prob: (H,W) en [0..1]. Devuelve máscara binaria post-procesada (float32 {0,1})
    - threshold
    - conservar K componentes con mayor "confianza" (suma de probabilidad o área)
    - filtrar por área mínima
    - opcional: apertura binaria
    """
    bin_mask = (prob > float(thr)).astype(np.uint8)
    lbl, n = label(bin_mask)
    if n == 0:
        return bin_mask.astype(np.float32)

    areas = np.bincount(lbl.ravel())
    scores = []
    for i in range(1, n + 1):
        if prob_weighted:
            scores.append(prob[lbl == i].sum())
        else:
            scores.append(areas[i])
    order = np.argsort(scores)[::-1]  # mayor a menor

    keep_ids = set(int(order[j]) + 1 for j in range(min(keep_top, n)))

    out = np.zeros_like(bin_mask, dtype=np.uint8)
    for i in range(1, n + 1):
        if i in keep_ids and areas[i] >= min_area:
            out[lbl == i] = 1

    if apply_open:
        out = binary_opening(out.astype(bool), iterations=opening_iter).astype(np.uint8)

    return out.astype(np.float32)

def load_state_dict_forgiving(
    model: torch.nn.Module,
    ckpt_path: str,
    map_location: str = "cpu",
    strict: bool = False
) -> Tuple[list, list]:
    """
    Carga un checkpoint eliminando prefijos comunes (_orig_mod., module., model., y 'state_dict' de Lightning).
    Devuelve (missing_keys, unexpected_keys) de load_state_dict.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state: Dict[str, Any] = ckpt.get("state_dict", ckpt)

    new_state = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    if missing:
        print(f"[load_state_dict_forgiving] Missing keys ({len(missing)}): {missing[:10]} ...")
    if unexpected:
        print(f"[load_state_dict_forgiving] Unexpected keys ({len(unexpected)}): {unexpected[:10]} ...")
    print(f"[load_state_dict_forgiving] Pesos cargados desde {ckpt_path}")
    return missing, unexpected

def inference_and_test_metrics(
    model,
    test_loader,
    best_threshold_cls_saved,
    threshold,
    save_visuals=False,
    viz_limit=50,
    viz_out_dir=None,
    viz_tag="",
    show_confusion_and_reports=True,
    postprocess = True,
    state_dict_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    element_index=1,
    save_dir=None,
    postprocess_keep_top=10,
    postprocess_min_area=100,
    postprocess_apply_opening=True,
    postprocess_opening_iter=2,
    params=None  # Parámetros de normalización (mean, std) para denormalizar imágenes
):
    """
    Extensión: aplica "gate" con la cabeza de clasificación y post-procesado de componentes para
    reducir falsos positivos manteniendo buen hit-or-miss.
    """
    # load_state_dict_forgiving(model, state_dict_path, map_location=device, strict=False)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()
    all_test_true_masks = []
    all_test_pred_probs = []
    all_test_pred_bin = []
    all_preds_cls, all_labels_cls = [], []

    # Directorio de visualizaciones
    if save_visuals:
        if viz_out_dir is None:
            safe_tag = viz_tag if viz_tag else "viz"
            viz_out_dir = os.path.join(
                save_dir,
                f"viz_element_{element_index}_{safe_tag}_thr{threshold:.2f}_cls{best_threshold_cls_saved:.2f}"
            )
        os.makedirs(viz_out_dir, exist_ok=True)

    dataset = getattr(test_loader, "dataset", None)
    sample_counter = 0
    saved = 0

    with torch.no_grad():
        for images, masks, labels in test_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device).unsqueeze(1)
            seg_out, cls_out = model(images)

            probs_cls = torch.sigmoid(cls_out)  # (B,1)
            preds_cls = (probs_cls > best_threshold_cls_saved).float()

            all_preds_cls.extend(preds_cls.cpu().numpy())
            all_labels_cls.extend(labels.cpu().numpy())
            all_test_true_masks.extend(masks[:,0].cpu().numpy())

            probs_seg = torch.sigmoid(seg_out).cpu().numpy()  # (B,1,H,W)
            all_test_pred_probs.extend(probs_seg[:,0])

            B = images.shape[0]
            for b in range(B):
                prob = probs_seg[b, 0]
                cls_p = float(probs_cls[b, 0].item())

                # Gate: si la prob de clase no supera el umbral, anula la máscara
                if cls_p <= float(best_threshold_cls_saved):
                    pred_bin = np.zeros_like(prob, dtype=np.float32)
                else:
                    if postprocess:
                        pred_bin = postprocess_mask(
                            prob=prob,
                            thr=threshold,
                            keep_top=postprocess_keep_top,
                            min_area=postprocess_min_area,
                            prob_weighted=True,
                            apply_open=postprocess_apply_opening,
                            opening_iter=postprocess_opening_iter,
                        )
                    else:
                        pred_bin = (prob > float(threshold)).astype(np.float32)
                all_test_pred_bin.append(pred_bin)

                # Visualizaciones
                if save_visuals and saved < viz_limit:
                    img_rgb = _denormalize_image(images[b], params["mean"], params["std"])
                    gt = masks[b, 0].detach().cpu().numpy().astype(np.float32)

                    if dataset is not None and hasattr(dataset, "df"):
                        try:
                            image_name = dataset.df.iloc[sample_counter]["image_name"]
                            stem = os.path.splitext(os.path.basename(image_name))[0]
                        except Exception:
                            stem = f"sample_{sample_counter:06d}"
                    else:
                        stem = f"sample_{sample_counter:06d}"

                    out_path = os.path.join(viz_out_dir, f"{stem}_panel.png")
                    _save_triplet_panel(
                        img_rgb, gt, pred_bin, out_path,
                        titles=("Imagen", "Máscara GT", f"Pred (thr={threshold:.2f}, gate={best_threshold_cls_saved:.2f})")
                    )
                    saved += 1
                sample_counter += 1

    # Para clasificación
    all_preds_cls = [int(p[0]) for p in all_preds_cls]
    all_labels_cls = [int(l[0]) for l in all_labels_cls]

    if show_confusion_and_reports:
        cm = confusion_matrix(all_labels_cls, all_preds_cls, normalize='true')
        if save_visuals:
            disp = ConfusionMatrixDisplay(cm, display_labels=[f"No Element {element_index}", f"Element {element_index}"])
            disp.plot(cmap="Blues")
            plt.title("Matriz de Confusión - Clasificación Test (con gate)")
            plt.show()
            print("Reporte de clasificación en Test (con gate):")
            print(classification_report(all_labels_cls, all_preds_cls,
                                        target_names=[f"No Element {element_index}", f"Element {element_index}"]))

    # Métricas de segmentación con máscara post-procesada
    iou_list, dice_list, amd_list = [], [], []
    hit_or_miss_0_list, hit_or_miss_1_list = [], []
    hit_or_miss_25_list, hit_or_miss_5_list, hit_or_miss_75_list = [], [], []
    oversampling_objects_0_list, oversampling_objects_1_list = [], []
    oversampling_objects_25_list, oversampling_objects_50_list, oversampling_objects_75_list = [], [], []
    coverage_list, fp_rate_list = [], []
    recall_list, precision_list, f1_list = [], [], []
    empty_gt_total = 0
    empty_gt_with_pred = 0
    non_empty_gt_without_pred = 0
    non_empty_gt_total = 0
    for mask, pred_mask, prob in zip(all_test_true_masks, all_test_pred_bin, all_test_pred_probs):
        if mask.sum() == 0: # Gt vacio
            empty_gt_total += 1
            if pred_mask.sum() > 0:
                empty_gt_with_pred += 1
        else:
            non_empty_gt_total += 1
            if pred_mask.sum() == 0: # Gt no vacio, pred vacio
                non_empty_gt_without_pred += 1



        intersection = (pred_mask * mask).sum()
        union = ((pred_mask + mask) > 0).sum()
        iou = intersection / (union + 1e-8)
        iou_list.append(iou)
        dice = 2 * intersection / (pred_mask.sum() + mask.sum() + 1e-8)
        dice_list.append(dice)
        hit_or_miss_0_list.append(hit_or_miss_iou(mask, pred_mask, iou_thresh=0.0))
        hit_or_miss_1_list.append(hit_or_miss_iou(mask, pred_mask, iou_thresh=0.1))
        hit_or_miss_25_list.append(hit_or_miss_iou(mask, pred_mask, iou_thresh=0.25))
        hit_or_miss_5_list.append(hit_or_miss_iou(mask, pred_mask, iou_thresh=0.5))
        hit_or_miss_75_list.append(hit_or_miss_iou(mask, pred_mask, iou_thresh=0.75))
        oversampling_objects_0_list.append(oversampling_objects(mask, pred_mask, iou_thresh=0.0))
        oversampling_objects_1_list.append(oversampling_objects(mask, pred_mask, iou_thresh=0.1))
        oversampling_objects_25_list.append(oversampling_objects(mask, pred_mask, iou_thresh=0.25))
        oversampling_objects_50_list.append(oversampling_objects(mask, pred_mask, iou_thresh=0.5))
        oversampling_objects_75_list.append(oversampling_objects(mask,pred_mask, iou_thresh=0.75))
        coverage_list.append(coverage(mask, pred_mask))
        fp_rate_list.append(false_positive_rate(mask, pred_mask))
        amd_list.append(average_minimum_distance(mask, pred_mask))
        recall_list.append(pixel_recall(mask, pred_mask))
        precision_list.append(pixel_precision(mask, pred_mask))
        f1_list.append(pixel_f1(mask, pred_mask))

    if empty_gt_total > 0:
        empty_gt_fp_rate = empty_gt_with_pred / empty_gt_total
    else:
        empty_gt_fp_rate = 0.0

    if non_empty_gt_total > 0:
        non_empty_gt_fn_rate = non_empty_gt_without_pred / non_empty_gt_total
    else:
        non_empty_gt_fn_rate = 0.0

    metrics = {
        "IoU": np.array(iou_list),
        "Dice": np.array(dice_list),
        "Hit0": np.array(hit_or_miss_0_list),
        "Hit1": np.array(hit_or_miss_1_list),
        "Hit25": np.array(hit_or_miss_25_list),
        "Hit5": np.array(hit_or_miss_5_list),
        "Hit75": np.array(hit_or_miss_75_list),
        "OversamplingObjects0": np.array(oversampling_objects_0_list),
        "OversamplingObjects1": np.array(oversampling_objects_1_list),
        "OversamplingObjects25": np.array(oversampling_objects_25_list),
        "OversamplingObjects50": np.array(oversampling_objects_50_list),
        "OversamplingObjects75": np.array(oversampling_objects_75_list),
        "Coverage": np.array(coverage_list),
        "FPRate": np.array(fp_rate_list),
        "AMD": np.array(amd_list),
        "PixelRecall": np.array(recall_list),
        "PixelPrecision": np.array(precision_list),
        "PixelF1": np.array(f1_list),
        "EmptyGT_FPRate": empty_gt_fp_rate,
        "NonEmptyGT_FNRate": non_empty_gt_fn_rate,
        "confusion_matrix_cls": cm if show_confusion_and_reports else None,
    }

    for mname in ["IoU", "Dice", "Coverage", "FPRate", "AMD", "PixelRecall", "PixelPrecision", "PixelF1",
                  "Hit0", "Hit1", "Hit25", "Hit5", "Hit75", "OversamplingObjects0", "OversamplingObjects1",
                  "OversamplingObjects25", "OversamplingObjects50", "OversamplingObjects75"]:
        arr = metrics[mname]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            print(f"{mname}: sin datos finitos (NaN/Inf).")
            continue
        print(
            f"{mname}: media={arr.mean():.4f}, mediana={np.median(arr):.4f}, p90={np.percentile(arr, 90):.4f}, p95={np.percentile(arr, 95):.4f}"
        )
    print("Images-level metrics")
    print(f"Empty GT with positives detections: {empty_gt_fp_rate:.4f}, Non-empty GT with no detections: {non_empty_gt_fn_rate:.4f}")
    return metrics
