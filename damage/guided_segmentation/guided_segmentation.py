import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, accuracy_score
)
import segmentation_models_pytorch as smp

from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, TverskyLoss
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import label, center_of_mass, distance_transform_edt

from damage.datasets.dataset import CSVCroppedImagesDataset

# ===========================
# Parámetros
# ===========================
IMAGE_DIR = "/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/cropped_images"
MASK_DIR = "/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/cropped_masks"
CSV_PATH = "/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/crop_damage_classes.csv"

BATCH_SIZE = 8
NUM_EPOCHS = 30
LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ELEMENT_INDEX = "7" # Seguir probando con "7"
NUM_WORKERS = 16
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 7
IMAGE_SIZE = 320
classification_loss_weight = 0.1

BEST_MODEL_DIR = f"/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/test_guided_segmentation_element_{ELEMENT_INDEX}/fix"
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, f"best_model_element_{ELEMENT_INDEX}.pth")
backbone = 'efficientnet-b0'

# ===========================
# Transformaciones
# ===========================
params = smp.encoders.get_preprocessing_params(backbone)
train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.432, saturation=0.4, hue=0.1, p=0.5),
    A.Normalize(mean=params['mean'], std=params['std']),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=params['mean'], std=params['std']),
    ToTensorV2()
])


#Nueva funcion de perdida
def gaussian_kernel(kernel_size=7, sigma=2, channels=1):
    """Crea un kernel gaussiano 2D para convolución"""
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel

class GaussianDiceLoss(nn.Module):
    def __init__(self, kernel_size=7, sigma=2, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = None

    def _kernel(self, C, device, dtype):
        ax = torch.arange(-self.kernel_size // 2 + 1., self.kernel_size // 2 + 1., device=device, dtype=dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        k = torch.exp(-(xx**2 + yy**2) / (2. * (float(self.sigma) ** 2)))
        k = k / (k.sum() + 1e-12)
        k = k.view(1, 1, self.kernel_size, self.kernel_size).repeat(C, 1, 1, 1)
        return k

    def forward(self, pred, target):
        # Trata siempre pred como logits
        pred = torch.sigmoid(pred)
        B, C, H, W = pred.shape
        if (self.kernel is None) or (self.kernel.shape[0] != C) or (self.kernel.device != pred.device) or (self.kernel.dtype != pred.dtype):
            self.kernel = self._kernel(C, pred.device, pred.dtype)

        pred_s = F.conv2d(pred, self.kernel, padding=self.kernel_size//2, groups=C)
        targ_s = F.conv2d(target, self.kernel, padding=self.kernel_size//2, groups=C)

        inter = (pred_s * targ_s).sum(dim=(1,2,3))
        total = pred_s.sum(dim=(1,2,3)) + targ_s.sum(dim=(1,2,3))
        dice = (2 * inter + self.smooth) / (total + self.smooth)
        return 1 - dice.mean()

class CombinedSegLoss(nn.Module):
    """
    Sube precisión penalizando FP: GaussianDice + BCEWithLogits.
    Ajusta lambda_bce según tu dataset (p.ej., 0.5–1.5).
    """
    def __init__(self, lambda_bce=1.0, pos_weight=None):
        super().__init__()
        self.dice = GaussianDiceLoss(kernel_size=7, sigma=2)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.lambda_bce = float(lambda_bce)

    def forward(self, seg_logits, seg_target):
        return self.dice(seg_logits, seg_target) + self.lambda_bce * self.bce(seg_logits, seg_target)



# ===========================
# Métricas
# ===========================
def iou_score(preds, targets, threshold=0.5, logits=True):
    if logits: preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    union = ((preds + targets) > 0).sum(dim=(1,2,3))
    return (intersection / (union + 1e-8)).mean().item()

def dice_score(preds, targets, threshold=0.5, logits=True):
    if logits: preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    total = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    return (2 * intersection / (total + 1e-8)).mean().item()

def seg_precision_recall(preds, targets, threshold=0.5, logits=True):
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
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    f1_scores = f1_scores[:-1]
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def find_optimal_iou_threshold(y_true_masks, y_pred_probs):
    thresholds = np.linspace(0, 1, 50)
    mean_ious = []
    for th in thresholds:
        ious = []
        for yt, yp in zip(y_true_masks, y_pred_probs):
            pred = (yp > th).astype(np.float32)
            intersection = (pred * yt).sum()
            union = ((pred + yt) > 0).sum()
            iou = intersection / (union + 1e-8)
            ious.append(iou)
        mean_ious.append(np.mean(ious))
    best_th = thresholds[np.argmax(mean_ious)]
    best_iou = np.max(mean_ious)
    return best_th, best_iou

def hit_or_miss_iou(gt_mask, pred_mask, iou_thresh=0.1):
    def iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / (union + 1e-8)

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

def coverage(gt_mask, pred_mask):
    # gt_mask: Máscara real (binaria)
    # pred_mask: Máscara predicha (binaria, tras aplicar threshold)
    real_pixels = gt_mask > 0
    pred_pixels = pred_mask > 0
    # Píxeles de la máscara real cubiertos por alguna predicción
    covered_pixels = np.logical_and(real_pixels, pred_pixels).sum()
    total_pixels = real_pixels.sum()
    return covered_pixels / total_pixels if total_pixels > 0 else 1.0

def false_positive_rate(gt_mask, pred_mask):
    # Píxeles predichos como positivos
    pred_pixels = pred_mask > 0
    # Píxeles reales (ground truth)
    real_pixels = gt_mask > 0
    # Falsos positivos: píxeles predichos positivos que no corresponden a objeto real
    false_positives = np.logical_and(pred_pixels, np.logical_not(real_pixels)).sum()
    total_predicted = pred_pixels.sum()
    # Porcentaje de píxeles predichos que son falsos positivos
    fp_rate = false_positives / total_predicted if total_predicted > 0 else 0.0
    return fp_rate

def average_minimum_distance(mask_gt, mask_pred):
    # Ambos binarios (0/1)
    mask_gt = mask_gt.astype(bool)
    mask_pred = mask_pred.astype(bool)
    # Distancia desde cada pixel a lo más cercano de la otra máscara
    dt_gt = distance_transform_edt(~mask_gt)  # distancia de cada pixel a objeto real
    dt_pred = distance_transform_edt(~mask_pred)

    # Para cada pixel predicho positivo, su distancia a objeto real
    if mask_pred.sum() > 0:
        avg_pred_to_gt = dt_gt[mask_pred].mean()
    else:
        avg_pred_to_gt = np.inf
    # Para cada pixel real positivo, su distancia a predicho
    if mask_gt.sum() > 0:
        avg_gt_to_pred = dt_pred[mask_gt].mean()
    else:
        avg_gt_to_pred = np.inf

    # Promedio de ambos sentidos
    return (avg_pred_to_gt + avg_gt_to_pred) / 2

def pixel_recall(gt_mask, pred_mask):
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    tp = np.logical_and(gt_mask, pred_mask).sum()
    fn = np.logical_and(gt_mask, np.logical_not(pred_mask)).sum()
    if (tp + fn) == 0:
        return np.nan
    return tp / (tp + fn)

def pixel_precision(gt_mask, pred_mask):
    gt_mask = gt_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)
    tp = np.logical_and(gt_mask, pred_mask).sum()
    fp = np.logical_and(np.logical_not(gt_mask), pred_mask).sum()
    if (tp + fp) == 0:
        return np.nan
    return tp / (tp + fp)

def pixel_f1(gt_mask, pred_mask):
    rec = pixel_recall(gt_mask, pred_mask)
    prec = pixel_precision(gt_mask, pred_mask)
    if np.isnan(rec) or np.isnan(prec) or (rec + prec) == 0:
        return np.nan
    return 2 * (prec * rec) / (prec + rec)

def pick_threshold_for_precision(y_true_pix, y_prob_pix, target_precision=0.60):
    prec, rec, thr = precision_recall_curve(y_true_pix, y_prob_pix)
    m = prec >= target_precision
    if m.any():
        return float(thr[m.argmax()])
    # fallback: F1 máximo
    f1 = 2*prec*rec/(prec+rec+1e-8); f1 = f1[:-1]
    return float(thr[np.argmax(f1)])

def find_optimal_pixel_f1_threshold(y_true_masks, y_pred_probs):
    # y_true_masks y y_pred_probs son listas de (1,H,W) en [0/1] y [0..1]
    y_true = np.concatenate([m.ravel() for m in y_true_masks]).astype(np.uint8)
    y_prob = np.concatenate([p.ravel() for p in y_pred_probs]).astype(np.float32)
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    f1 = f1[:-1]  # alinear con thr
    return float(thr[np.argmax(f1)]), float(f1.max())

def calculate_segmentation_pos_weight(train_loader):
    pos_pix = 0
    neg_pix = 0
    sample_batches = max(1, len(train_loader) // 5)  # muestrea ~20% para acelerar
    with torch.no_grad():
        for i, (_, masks, _) in enumerate(train_loader):
            m = masks  # (B,1,H,W) en {0,1}
            pos_pix += m.sum().item()
            neg_pix += (m.numel() - m.sum()).item()
            if i + 1 >= sample_batches:
                break
    if pos_pix == 0:
        seg_pos_weight = None
    else:
        seg_pos_weight = torch.tensor([neg_pix / max(pos_pix, 1)], device=DEVICE, dtype=torch.float32)

    return seg_pos_weight


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


class DualUNetPlusPlusGuided(nn.Module):
    """
    U-Net++ (SMP) con:
      - Cabeza de clasificación auxiliar desde f5 (GAP+FC).
      - Guía espacial: f5 -> conv1x1 -> upsample -> fusiona con decoder -> convs -> sigmoid -> gating residual.
      - Guía de canal (FiLM): GAP(f5) -> MLP -> gamma,beta -> y = (1+gamma)*x + beta.
    Devuelve (seg_map_logits, cls_logits).
    """
    def __init__(self, backbone_name='efficientnet-b0', pretrained=True, in_channels=3, seg_classes=1,
                 cls_classes=1, use_spatial_attn=True, use_film=True, film_hidden=512):
        super().__init__()
        self.seg_model = smp.UnetPlusPlus(
            encoder_name=backbone_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=seg_classes,
            activation=None
        )
        self.encoder = self.seg_model.encoder
        self.decoder = self.seg_model.decoder
        self.segmentation_head = self.seg_model.segmentation_head

        c5 = self.encoder.out_channels[-1]
        # cabeza cls
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc = nn.Linear(c5, cls_classes)

        # descubrir canales del decoder
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 256, 256)
            feats = self.encoder(dummy)
            dfeat = self.decoder(feats)
            cdec = dfeat.shape[1]
        self.cdec = cdec

        self.use_spatial_attn = use_spatial_attn
        self.use_film = use_film

        if use_spatial_attn:
            # proyección de contexto desde f5 a cdec y fusión con decoder
            self.ctx_proj = nn.Conv2d(c5, cdec, kernel_size=1)
            self.attn_gen = nn.Sequential(
                nn.Conv2d(cdec*2, cdec, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(cdec, cdec, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

        if use_film:
            self.film = nn.Sequential(
                nn.Linear(c5, film_hidden), nn.ReLU(inplace=True),
                nn.Linear(film_hidden, 2*cdec)
            )

    def forward(self, x):
        feats = self.encoder(x)
        f5 = feats[-1]                         # (B, C5, H5, W5)
        dec = self.decoder(feats)              # (B, Cdec, H, W)

        # cls auxiliar
        cls_logits = self.cls_fc(self.global_pool(f5).flatten(1))

        # FiLM (canal)
        if self.use_film:
            ctx_vec = self.global_pool(f5).flatten(1)  # (B, C5)
            gb = self.film(ctx_vec)                    # (B, 2*Cdec)
            gamma, beta = torch.chunk(gb, 2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta  = beta.unsqueeze(-1).unsqueeze(-1)
            dec = (1 + gamma) * dec + beta

        # Atención espacial (pixel-wise)
        if self.use_spatial_attn:
            ctx = self.ctx_proj(f5)                                        # (B, Cdec, H5, W5)
            ctx = F.interpolate(ctx, size=dec.shape[2:], mode='bilinear', align_corners=False)
            attn_in = torch.cat([dec, ctx], dim=1)                         # (B, 2*Cdec, H, W)
            attn_map = self.attn_gen(attn_in)                              # (B, Cdec, H, W)
            dec = dec * (1 + attn_map)                                     # gating residual

        seg_logits = self.segmentation_head(dec)  # sin activación (logits)
        return seg_logits, cls_logits



def inference_and_test_metrics(
    model,
    test_loader,
    best_threshold_cls_saved,
    threshold,
    save_visuals=False,
    viz_limit=50,
    viz_out_dir=None,
    viz_tag="",
    show_confusion_and_reports=True
):
    """
    Extensión: si save_visuals=True, guarda paneles (imagen, GT, pred) de hasta viz_limit muestras.
    viz_out_dir por defecto: BEST_MODEL_DIR/viz_element_{ELEMENT_INDEX}_{viz_tag}_thr{threshold:.2f}
    """
    # ===========================
    # Evaluación en test con el umbral óptimo de IoU
    # ===========================
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()
    all_test_true_masks = []
    all_test_pred_probs = []
    all_preds_cls, all_labels_cls = [], []

    # Directorio de visualizaciones
    if save_visuals:
        if viz_out_dir is None:
            safe_tag = viz_tag if viz_tag else "viz"
            viz_out_dir = os.path.join(
                BEST_MODEL_DIR,
                f"viz_element_{ELEMENT_INDEX}_{safe_tag}_thr{threshold:.2f}"
            )
        os.makedirs(viz_out_dir, exist_ok=True)

    # Para recuperar nombres de archivo (DataLoader con shuffle=False)
    dataset = getattr(test_loader, "dataset", None)
    sample_counter = 0
    saved = 0

    with torch.no_grad():
        for images, masks, labels in test_loader:
            images, masks, labels = images.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            seg_out, cls_out = model(images)

            probs_cls = torch.sigmoid(cls_out)
            preds_cls = (probs_cls > best_threshold_cls_saved).float()
            all_preds_cls.extend(preds_cls.cpu().numpy())
            all_labels_cls.extend(labels.cpu().numpy())
            all_test_true_masks.extend(masks.cpu().numpy())
            all_test_pred_probs.extend(torch.sigmoid(seg_out).cpu().numpy())

            # Guardar visualizaciones
            if save_visuals and saved < viz_limit:
                B = images.shape[0]
                for b in range(B):
                    if saved >= viz_limit:
                        break

                    # Denormalizar imagen y preparar máscaras
                    img_rgb = _denormalize_image(images[b], params["mean"], params["std"])
                    gt = masks[b, 0].detach().cpu().numpy().astype(np.float32)
                    prob = torch.sigmoid(seg_out[b, 0]).detach().cpu().numpy().astype(np.float32)
                    pred_bin = (prob > float(threshold)).astype(np.float32)

                    # Nombre de salida (usar image_name del dataset si está disponible)
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
                        titles=("Imagen", "Máscara GT", f"Predicción (thr={threshold:.2f})")
                    )
                    saved += 1
                    sample_counter += 1
            else:
                sample_counter += images.shape[0]

    # Para clasificación
    all_preds_cls = [int(p[0]) for p in all_preds_cls]
    all_labels_cls = [int(l[0]) for l in all_labels_cls]

    if show_confusion_and_reports:
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
        cm = confusion_matrix(all_labels_cls, all_preds_cls, normalize='true')
        disp = ConfusionMatrixDisplay(cm, display_labels=[f"No Element {ELEMENT_INDEX}", f"Element {ELEMENT_INDEX}"])
        disp.plot(cmap="Blues")
        plt.title("Matriz de Confusión - Clasificación Test")
        plt.show()
        print("Reporte de clasificación en Test:")
        print(classification_report(all_labels_cls, all_preds_cls,
                                    target_names=[f"No Element {ELEMENT_INDEX}", f"Element {ELEMENT_INDEX}"]))

    # Para segmentación: calcula métricas usando el threshold elegido
    iou_list, dice_list, amd_list = [], [], []
    hit_or_miss_0_list, hit_or_miss_1_list = [], []
    hit_or_miss_25_list, hit_or_miss_5_list, hit_or_miss_75_list = [], [], []
    coverage_list, fp_rate_list = [], []
    recall_list, precision_list, f1_list = [], [], []

    for mask, prob in zip(all_test_true_masks, all_test_pred_probs):
        pred_mask = (prob > threshold).astype(np.float32)
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
        coverage_list.append(coverage(mask, pred_mask))
        fp_rate_list.append(false_positive_rate(mask, pred_mask))
        amd_list.append(average_minimum_distance(mask, pred_mask))
        recall_list.append(pixel_recall(mask, pred_mask))
        precision_list.append(pixel_precision(mask, pred_mask))
        f1_list.append(pixel_f1(mask, pred_mask))

    metrics = {
        "IoU": np.array(iou_list),
        "Dice": np.array(dice_list),
        "Hit0": np.array(hit_or_miss_0_list),
        "Hit1": np.array(hit_or_miss_1_list),
        "Hit25": np.array(hit_or_miss_25_list),
        "Hit5": np.array(hit_or_miss_5_list),
        "Hit75": np.array(hit_or_miss_75_list),
        "Coverage": np.array(coverage_list),
        "FPRate": np.array(fp_rate_list),
        "AMD": np.array(amd_list),
        "PixelRecall": np.array(recall_list),
        "PixelPrecision": np.array(precision_list),
        "PixelF1": np.array(f1_list)
    }

    for mname in ["IoU", "Dice", "Coverage", "FPRate", "AMD", "PixelRecall", "PixelPrecision", "PixelF1", "Hit0", "Hit1", "Hit25", "Hit5", "Hit75"]:
        arr = metrics[mname]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            print(f"{mname}: sin datos finitos (NaN/Inf).")
            continue
        print(
            f"{mname}: media={arr.mean():.4f}, mediana={np.median(arr):.4f}, p90={np.percentile(arr, 90):.4f}, p95={np.percentile(arr, 95):.4f}"
        )

# ===========================
# Entrenamiento y métricas por epoch
# ===========================
def main():
    df = pd.read_csv(CSV_PATH)
    df_pos = df[df[ELEMENT_INDEX] == True]
    df_neg = df[df[ELEMENT_INDEX] == False]
    df_neg_sampled = df_neg.sample(n=int(len(df_pos)*2), random_state=42)
    df_balanced = pd.concat([df_pos, df_neg_sampled]).sample(frac=1, random_state=42)

    df_trainval, df_test = train_test_split(
        df_balanced, test_size=0.2, random_state=42, stratify=df_balanced[ELEMENT_INDEX]
    )
    df_train, df_val = train_test_split(
        df_trainval, test_size=0.25, random_state=42, stratify=df_trainval[ELEMENT_INDEX]
    )

    train_dataset = CSVCroppedImagesDataset(df_train, IMAGE_DIR, MASK_DIR, transform=train_transform, element_index=ELEMENT_INDEX)
    val_dataset = CSVCroppedImagesDataset(df_val, IMAGE_DIR, MASK_DIR, transform=val_transform, element_index=ELEMENT_INDEX)
    test_dataset = CSVCroppedImagesDataset(df_test, IMAGE_DIR, MASK_DIR, transform=val_transform, element_index=ELEMENT_INDEX)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    seg_pos_weight = calculate_segmentation_pos_weight(train_loader)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    model = DualUNetPlusPlusGuided(backbone_name='efficientnet-b0', pretrained=True).to(DEVICE)
    model = torch.compile(model)
    torch.set_float32_matmul_precision('high')

    # seg_loss_fn = CombinedLoss(DiceLoss(mode='binary'), FocalLoss(mode='binary'))
    # seg_loss_fn = DiceLoss(mode='binary')
    # seg_loss_fn = TverskyLoss(mode='binary', alpha=0.25, beta=0.75)
    seg_loss_fn = GaussianDiceLoss(kernel_size=7, sigma=2)
    # seg_loss_fn = CombinedSegLoss(lambda_bce=1.0, pos_weight=seg_pos_weight)
    N_pos = len(df_pos)
    N_neg = len(df_neg_sampled)
    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([N_neg/N_pos], device=DEVICE))

    # optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    dec_params = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
    cls_params = list(model.cls_fc.parameters())
    enc_params = list(model.encoder.parameters())

    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": 1e-5, "weight_decay": 1e-5},
        {"params": dec_params + cls_params, "lr": 1e-4, "weight_decay": 1e-5},
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    metrics_history = {
        "train_loss": [], "val_loss": [], "val_iou": [], "val_dice": [], "val_prec_seg": [], "val_rec_seg": [],
        "val_acc_cls": [], "val_f1_cls": [], "val_roc_auc_cls": [], "val_pr_auc_cls": []
    }

    best_val_iou = 0
    epochs_no_improve = 0
    best_threshold = 0.5
    best_threshold_iou_saved = 0.5  # para guardar el mejor umbral de segmentación
    best_threshold_cls_saved = 0.5  # para guardar el mejor umbral de clasificación
    best_threshold_f1_saved = 0.5
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0
        for images, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images, masks, labels = images.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            seg_out, cls_out = model(images)
            loss = seg_loss_fn(seg_out, masks) + classification_loss_weight * cls_loss_fn(cls_out, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        metrics_history["train_loss"].append(running_loss / len(train_loader.dataset))

        # Validación
        model.eval()
        val_loss = 0
        val_iou, val_dice, val_prec_seg, val_rec_seg = 0, 0, 0, 0
        y_true_cls, y_probs_cls = [], []
        all_val_true_masks = []
        all_val_pred_probs = []

        with torch.no_grad():
            for images, masks, labels in val_loader:
                images, masks, labels = images.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                seg_out, cls_out = model(images)
                loss = seg_loss_fn(seg_out, masks) + classification_loss_weight * cls_loss_fn(cls_out, labels)
                val_loss += loss.item() * images.size(0)

                val_iou += iou_score(seg_out, masks) * images.size(0)
                val_dice += dice_score(seg_out, masks) * images.size(0)
                prec, rec = seg_precision_recall(seg_out, masks)
                val_prec_seg += prec * images.size(0)
                val_rec_seg += rec * images.size(0)

                y_true_cls.extend(labels.cpu().numpy())
                y_probs_cls.extend(torch.sigmoid(cls_out).cpu().numpy())
                all_val_true_masks.extend(masks.cpu().numpy())
                all_val_pred_probs.extend(torch.sigmoid(seg_out).cpu().numpy())

        n_val = len(val_loader.dataset)
        val_loss /= n_val
        val_iou /= n_val
        val_dice /= n_val
        val_prec_seg /= n_val
        val_rec_seg /= n_val

        metrics_history["val_loss"].append(val_loss)

        y_true_cls = np.array(y_true_cls).flatten()
        y_probs_cls = np.array(y_probs_cls).flatten()
        val_preds_cls = (y_probs_cls > best_threshold).astype(int)

        metrics_history["val_iou"].append(val_iou)
        metrics_history["val_dice"].append(val_dice)
        metrics_history["val_prec_seg"].append(val_prec_seg)
        metrics_history["val_rec_seg"].append(val_rec_seg)
        metrics_history["val_acc_cls"].append(accuracy_score(y_true_cls, val_preds_cls))
        metrics_history["val_f1_cls"].append(f1_score(y_true_cls, val_preds_cls))
        metrics_history["val_roc_auc_cls"].append(roc_auc_score(y_true_cls, y_probs_cls))
        metrics_history["val_pr_auc_cls"].append(average_precision_score(y_true_cls, y_probs_cls))

        # Buscar threshold óptimo para IoU
        best_threshold_iou, best_iou = find_optimal_iou_threshold(all_val_true_masks, all_val_pred_probs)
        print(f"Epoch {epoch+1}: Loss(val)={val_loss:.4f}, IoU={val_iou:.4f}, Dice={val_dice:.4f}, Prec(seg)={val_prec_seg:.4f}, Rec(seg)={val_rec_seg:.4f}, "
              f"Acc(cls)={metrics_history['val_acc_cls'][-1]:.3f}, F1(cls)={metrics_history['val_f1_cls'][-1]:.3f}, Threshold_cls={best_threshold:.3f}, Threshold_iou={best_threshold_iou:.3f}")

        scheduler.step(val_iou)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            epochs_no_improve = 0
            # Calcula y guarda el mejor threshold para clasificación también
            best_threshold, best_f1 = find_optimal_threshold(y_true_cls, y_probs_cls)
            thr_f1, max_f1 = find_optimal_pixel_f1_threshold(all_val_true_masks, all_val_pred_probs)
            best_threshold_cls_saved = best_threshold
            best_threshold_iou_saved = best_threshold_iou
            best_threshold_f1_saved = thr_f1
            print(f"✅ Mejor modelo guardado con Val IoU: {val_iou:.4f},"
                  f" Best Threshold Clasificación: {best_threshold:.3f}, Best Threshold IoU: {best_threshold_iou:.3f}, Best Threshold Pixel-F1: {thr_f1:.3f}")

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"🛑 Early stopping en epoch {epoch+1}")
            break

    # ===========================
    # Graficar métricas
    # ===========================
    epochs_arr = range(1, len(metrics_history["val_iou"]) + 1)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_arr, metrics_history["val_loss"], label="Val Loss")
    plt.plot(epochs_arr, metrics_history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss de Entrenamiento y Validación")

    plt.subplot(1, 3, 2)
    plt.plot(epochs_arr, metrics_history["val_iou"], label="IoU")
    plt.plot(epochs_arr, metrics_history["val_dice"], label="Dice")
    plt.plot(epochs_arr, metrics_history["val_prec_seg"], label="Precision(seg)")
    plt.plot(epochs_arr, metrics_history["val_rec_seg"], label="Recall(seg)")
    plt.xlabel("Epoch")
    plt.ylabel("Métrica Segmentación")
    plt.legend()
    plt.title("Segmentación")

    plt.subplot(1, 3, 3)
    plt.plot(epochs_arr, metrics_history["val_acc_cls"], label="Accuracy")
    plt.plot(epochs_arr, metrics_history["val_f1_cls"], label="F1")
    plt.plot(epochs_arr, metrics_history["val_roc_auc_cls"], label="ROC-AUC")
    plt.plot(epochs_arr, metrics_history["val_pr_auc_cls"], label="PR-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Métrica Clasificación")
    plt.legend()
    plt.title("Clasificación")

    plt.tight_layout()
    plt.show()

    inference_and_test_metrics(
        model=model,
        test_loader=test_loader,
        best_threshold_cls_saved=best_threshold_cls_saved,
        threshold=best_threshold_iou_saved,
        save_visuals=True,
        viz_limit=100,
        viz_tag="iou_opt"  # etiqueta para la carpeta
    )

    # Guarda hasta 100 paneles con el umbral óptimo por F1-pixel
    inference_and_test_metrics(
        model=model,
        test_loader=test_loader,
        best_threshold_cls_saved=best_threshold_cls_saved,
        threshold=best_threshold_f1_saved,
        save_visuals=True,
        viz_limit=100,
        viz_tag="f1pix_opt"
    )

if __name__ == "__main__":
    import random
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(42)

    main()
