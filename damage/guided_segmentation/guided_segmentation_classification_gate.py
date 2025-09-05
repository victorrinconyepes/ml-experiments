import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_curve,f1_score,
    roc_auc_score, average_precision_score, accuracy_score
)
import segmentation_models_pytorch as smp

import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

import mlflow
from pyvexcelutils.aws.secrets import Secrets
from pyvexcelutils.mlflow.mlflow_helper import setup_mlflow, MLFlowNamer

from damage.datasets.dataset import CSVCroppedImagesDataset
from damage.guided_segmentation.custom_metrics import iou_score, dice_score, seg_precision_recall, \
    find_optimal_iou_threshold
from damage.guided_segmentation.models import DualUNetPlusPlusGuided, DualFPNGuided
import cv2

# ===========================
# Parámetros
# ===========================
IMAGE_DIR = "/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/cropped_images/"
MASK_DIR = "/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/cropped_masks"
# CSV_PATH = "/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/crop_damage_classes.csv"
ELEMENT_INDEX = "7"
SPLIT_DIR = f"/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/split/element_{ELEMENT_INDEX}"

BATCH_SIZE = 16
NUM_EPOCHS = 30
LR = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 16
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 7
IMAGE_SIZE = 320
classification_loss_weight = 0.1

jira_ticket = "COPPER-2400"
MLFLOW_RUN_NAME = f"GaussianDiceLoss(kernel_size=7, sigma=2, smooth=1e-6)) + {classification_loss_weight} * WeightedBCE/AdamW/{LR}/DualFPNGuided/efficientnet-b0"
REMOVE_GRAYSKY = False

BEST_MODEL_DIR = f"/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/test_guided_segmentation_element_{ELEMENT_INDEX}/fix"
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, f"best_model_element_{ELEMENT_INDEX}.pth")
backbone = 'efficientnet-b0'

# menos FP manteniendo buen hit-or-miss
TARGET_PIXEL_PRECISION = 0.65        # precisión-píxel deseada para elegir umbral de segmentación
CLS_GATE_TARGET_PRECISION = 0.85     # precisión deseada para la puerta de clasificación

#Posprocesado
KEEP_TOP_COMPONENTS = 1              # nº de componentes a conservar (si esperas 1 elemento por crop)
MIN_COMPONENT_AREA = 64              # área mínima de componente (px)
APPLY_BINARY_OPENING = False         # aplicar apertura binaria para reducir ruido fino
OPENING_ITER = 1


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
# train_transform = A.Compose([
#     A.SmallestMaxSize(max_size=IMAGE_SIZE, interpolation=cv2.INTER_LINEAR),
#     A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE,
#                   border_mode=cv2.BORDER_REFLECT_101),
#
#     A.RandomCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
#
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101,
#              interpolation=cv2.INTER_LINEAR, p=0.5),
#
#     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#     A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.3),
#     A.Sharpen(alpha=(0.05, 0.2), lightness=(0.9, 1.1), p=0.2),
#
#     A.Normalize(mean=params['mean'], std=params['std']),
#     ToTensorV2()
# ])
# val_transform = A.Compose([
#     A.SmallestMaxSize(max_size=IMAGE_SIZE, interpolation=cv2.INTER_LINEAR),
#     A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE,
#                   border_mode=cv2.BORDER_REFLECT_101),
#     A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE),
#     A.Normalize(mean=params['mean'], std=params['std']),
#     ToTensorV2()
# ])



# nueva funcion de pérdida: suavizado gaussiano
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
    Sube precisión penalizando FP: GaussianDice + BCE ponderado por píxel.
    Ajusta lambda_bce y pesos según tu dataset (p.ej., w_neg 2–4 para castigar FP).
    """
    def __init__(self, lambda_bce=1.0, w_pos=1.0, w_neg=2.0):
        super().__init__()
        # Opcional: reducir smoothing si hay "bleeding" excesivo
        self.dice = GaussianDiceLoss(kernel_size=5, sigma=1)
        self.lambda_bce = float(lambda_bce)
        self.w_pos = float(w_pos)
        self.w_neg = float(w_neg)

    def forward(self, seg_logits, seg_target):
        dice = self.dice(seg_logits, seg_target)
        bce_per_pix = F.binary_cross_entropy_with_logits(seg_logits, seg_target, reduction='none')
        weights = torch.where(seg_target > 0.5,
                              torch.as_tensor(self.w_pos, device=seg_target.device, dtype=seg_target.dtype),
                              torch.as_tensor(self.w_neg, device=seg_target.device, dtype=seg_target.dtype))
        bce = (bce_per_pix * weights).mean()
        return dice + self.lambda_bce * bce






# ===========================
# Umbral de clasificación por precisión objetivo y post-procesado de máscaras
# ===========================
def find_threshold_for_cls_precision(y_true, y_prob, target_precision=0.85):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    m = prec[:-1] >= target_precision  # thr tiene len-1
    if m.any():
        return float(thr[m.argmax()])
    # Fallback: F1 máximo
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
    return float(thr[np.argmax(f1)])



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

# ===========================
# Entrenamiento y métricas por epoch
# ===========================
def main():
    # df = pd.read_csv(CSV_PATH)
    # df_pos = df[df[ELEMENT_INDEX] == True]
    # df_neg = df[df[ELEMENT_INDEX] == False]
    # df_neg_sampled = df_neg.sample(n=int(len(df_pos)*2), random_state=42)
    # df_balanced = pd.concat([df_pos, df_neg_sampled]).sample(frac=1, random_state=42)
    #
    # df_trainval, df_test = train_test_split(
    #     df_balanced, test_size=0.2, random_state=42, stratify=df_balanced[ELEMENT_INDEX]
    # )
    # df_train, df_val = train_test_split(
    #     df_trainval, test_size=0.25, random_state=42, stratify=df_trainval[ELEMENT_INDEX]
    # )
    df_train = pd.read_csv(os.path.join(SPLIT_DIR, 'train.csv'))
    df_val = pd.read_csv(os.path.join(SPLIT_DIR, 'val.csv'))
    df_test = pd.read_csv(os.path.join(SPLIT_DIR, 'test.csv'))

    #Remove graysky
    if REMOVE_GRAYSKY:
        df_train = df_train[~df_train['image_name'].str.contains('graysky', na=False)]
        df_val = df_val[~df_val['image_name'].str.contains('graysky', na=False)]
        df_test = df_test[~df_test['image_name'].str.contains('graysky', na=False)]

    train_dataset = CSVCroppedImagesDataset(df_train, IMAGE_DIR, MASK_DIR, transform=train_transform, element_index=ELEMENT_INDEX)
    val_dataset = CSVCroppedImagesDataset(df_val, IMAGE_DIR, MASK_DIR, transform=val_transform, element_index=ELEMENT_INDEX)
    test_dataset = CSVCroppedImagesDataset(df_test, IMAGE_DIR, MASK_DIR, transform=val_transform, element_index=ELEMENT_INDEX)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    seg_pos_weight = calculate_segmentation_pos_weight(train_loader)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # model = DualUNetPlusPlusGuided(backbone_name='efficientnet-b0', pretrained=True).to(DEVICE)
    model = DualFPNGuided(backbone_name='efficientnet-b0', pretrained=True).to(DEVICE)
    model = torch.compile(model)
    torch.set_float32_matmul_precision('high')

    # PÉRDIDAS
    # seg_loss_fn = TverskyLoss(mode='binary', alpha=0.7, beta=0.3)
    seg_loss_fn = GaussianDiceLoss(kernel_size=7, sigma=2)
    # seg_loss_fn = CombinedSegLoss(lambda_bce=1.0, w_pos=1.0, w_neg=2.0)

    N_pos = len(df_train[df_train[ELEMENT_INDEX] == True])
    N_neg = len(df_train) - N_pos
    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([N_neg/N_pos], device=DEVICE))

    # # OPTIMIZADOR
    # dec_params = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
    # cls_params = list(model.cls_fc.parameters())
    # enc_params = list(model.encoder.parameters())

    # optimizer = torch.optim.AdamW([
    #     {"params": enc_params, "lr": 1e-5, "weight_decay": 1e-5},  # encoder
    #     {"params": dec_params + cls_params, "lr": 1e-4, "weight_decay": 1e-5},  # FPN + cabeza
    # ])


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    metrics_history = {
        "train_loss": [], "val_loss": [],
        "train_seg_loss": [], "train_cls_loss": [],
        "val_seg_loss": [], "val_cls_loss": [],
        "val_iou": [], "val_dice": [], "val_prec_seg": [], "val_rec_seg": [],
        "val_acc_cls": [], "val_f1_cls": [], "val_roc_auc_cls": [], "val_pr_auc_cls": []
    }

    best_val_iou = 0
    epochs_no_improve = 0
    best_threshold = 0.5
    best_threshold_iou_saved = 0.5   # mejor umbral de segmentación por IoU
    best_threshold_cls_saved = 0.5   # mejor umbral de clasificación (por F1)
    best_threshold_f1_saved = 0.5    # mejor umbral de segmentación por F1-pixel
    best_threshold_pix_prec_saved = 0.5  # mejor umbral de segmentación por precisión-píxel objetivo
    best_threshold_cls_prec_saved = 0.5  # mejor umbral de clasificación por precisión objetivo (gate)

    # ===========================
    # MLflow setup
    # ===========================
    jira_ticket_url=None
    if jira_ticket is not None:
        jira_ticket_url = MLFlowNamer.define_jira_ticket_url(
            jira_ticket
        )
    experiment_name = "property-damage/guided-segmentation-by-classification/binary"
    username, password, url = Secrets(aws_profile=None).get_mlflow_user()
    setup_mlflow(username, password, url, experiment_name)
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME, description=jira_ticket_url):
        # Log params
        mlflow.log_params({
            "image_dir": IMAGE_DIR,
            "mask_dir": MASK_DIR,
            "split_dir": SPLIT_DIR,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "device": DEVICE,
            "element_index": ELEMENT_INDEX,
            "num_workers": NUM_WORKERS,
            "weight_decay": WEIGHT_DECAY,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "image_size": IMAGE_SIZE,
            "classification_loss_weight": classification_loss_weight,
            "backbone": backbone,
            "target_pixel_precision": TARGET_PIXEL_PRECISION,
            "cls_gate_target_precision": CLS_GATE_TARGET_PRECISION,
            "keep_top_components": KEEP_TOP_COMPONENTS,
            "min_component_area": MIN_COMPONENT_AREA,
            "apply_binary_opening": APPLY_BINARY_OPENING,
            "opening_iter": OPENING_ITER,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "test_size": len(test_dataset),
            "cls_pos_weight": float(N_neg/N_pos),
            "seg_pos_weight": float(seg_pos_weight.item()) if seg_pos_weight is not None else None,
        })

        for epoch in range(NUM_EPOCHS):
            model.train()
            running_total_loss = 0.0
            running_seg_loss = 0.0
            running_cls_loss = 0.0
            for images, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
                images, masks, labels = images.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                seg_out, cls_out = model(images)

                seg_loss = seg_loss_fn(seg_out, masks)
                cls_loss = cls_loss_fn(cls_out, labels)
                loss = seg_loss + classification_loss_weight * cls_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_size = images.size(0)
                running_total_loss += loss.item() * batch_size
                running_seg_loss += seg_loss.item() * batch_size
                running_cls_loss += cls_loss.item() * batch_size

            # metrics_history["train_loss"].append(running_loss / len(train_loader.dataset))
            train_loss_epoch = running_total_loss / len(train_loader.dataset)
            train_seg_loss_epoch = running_seg_loss / len(train_loader.dataset)
            train_cls_loss_epoch = running_cls_loss / len(train_loader.dataset)

            metrics_history["train_loss"].append(train_loss_epoch)
            metrics_history["train_seg_loss"].append(train_seg_loss_epoch)
            metrics_history["train_cls_loss"].append(train_cls_loss_epoch)

            # Validación
            model.eval()
            val_total_loss = 0.0
            val_seg_component = 0.0
            val_cls_component = 0.0
            val_iou, val_dice, val_prec_seg, val_rec_seg = 0, 0, 0, 0
            y_true_cls, y_probs_cls = [], []
            all_val_true_masks = []
            all_val_pred_probs = []

            with torch.no_grad():
                for images, masks, labels in val_loader:
                    images, masks, labels = images.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                    seg_out, cls_out = model(images)

                    seg_loss = seg_loss_fn(seg_out, masks)
                    cls_loss = cls_loss_fn(cls_out, labels)
                    loss = seg_loss + classification_loss_weight * cls_loss

                    batch_size = images.size(0)
                    val_total_loss += loss.item() * batch_size
                    val_seg_component += seg_loss.item() * batch_size
                    val_cls_component += cls_loss.item() * batch_size


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
            val_loss_epoch = val_total_loss / n_val
            val_seg_loss_epoch = val_seg_component / n_val
            val_cls_loss_epoch = val_cls_component / n_val
            val_iou /= n_val
            val_dice /= n_val
            val_prec_seg /= n_val
            val_rec_seg /= n_val

            metrics_history["val_loss"].append(val_loss_epoch)
            metrics_history["val_seg_loss"].append(val_seg_loss_epoch)
            metrics_history["val_cls_loss"].append(val_cls_loss_epoch)

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

            # Log metrics per epoch
            mlflow.log_metric("train_loss", train_loss_epoch, step=epoch)
            mlflow.log_metric("train_seg_loss", train_seg_loss_epoch, step=epoch)
            mlflow.log_metric("train_cls_loss", train_cls_loss_epoch, step=epoch)
            mlflow.log_metric("val_loss", val_loss_epoch, step=epoch)
            mlflow.log_metric("val_seg_loss", val_seg_loss_epoch, step=epoch)
            mlflow.log_metric("val_cls_loss", val_cls_loss_epoch, step=epoch)

            mlflow.log_metric("val_iou", val_iou, step=epoch)
            mlflow.log_metric("val_dice", val_dice, step=epoch)
            mlflow.log_metric("val_prec_seg", val_prec_seg, step=epoch)
            mlflow.log_metric("val_rec_seg", val_rec_seg, step=epoch)
            mlflow.log_metric("val_acc_cls", metrics_history["val_acc_cls"][-1], step=epoch)
            mlflow.log_metric("val_f1_cls", metrics_history["val_f1_cls"][-1], step=epoch)
            mlflow.log_metric("val_roc_auc_cls", metrics_history["val_roc_auc_cls"][-1], step=epoch)
            mlflow.log_metric("val_pr_auc_cls", metrics_history["val_pr_auc_cls"][-1], step=epoch)
            # Log LRs
            if len(optimizer.param_groups) >= 2:
                mlflow.log_metric("lr_enc", optimizer.param_groups[0]['lr'], step=epoch)
                mlflow.log_metric("lr_heads", optimizer.param_groups[1]['lr'], step=epoch)

            # Buscar threshold óptimo para IoU
            best_threshold_iou, best_iou = find_optimal_iou_threshold(all_val_true_masks, all_val_pred_probs)
            print(
                f"Epoch {epoch + 1}: "
                f"TrainLoss={train_loss_epoch:.4f} (Seg={train_seg_loss_epoch:.4f}, Cls={train_cls_loss_epoch:.4f}) | "
                f"ValLoss={val_loss_epoch:.4f} (Seg={val_seg_loss_epoch:.4f}, Cls={val_cls_loss_epoch:.4f}) | "
                f"IoU={val_iou:.4f}, Dice={val_dice:.4f}, Prec(seg)={val_prec_seg:.4f}, Rec(seg)={val_rec_seg:.4f} | "
                f"Acc(cls)={metrics_history['val_acc_cls'][-1]:.3f}, F1(cls)={metrics_history['val_f1_cls'][-1]:.3f} | "
                f"Thr_cls={best_threshold:.3f}, Thr_iou={best_threshold_iou:.3f}"
            )
            scheduler.step(val_iou)

            if val_iou > best_val_iou:
                best_val_iou = val_iou
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                epochs_no_improve = 0

                # Thresholds por distintas óptimas
                # best_threshold, best_f1 = find_optimal_threshold(y_true_cls, y_probs_cls)  # clasificación por F1
                # thr_f1, max_f1 = find_optimal_pixel_f1_threshold(all_val_true_masks, all_val_pred_probs)  # pixel-F1
                # best_threshold_cls_saved = best_threshold
                # best_threshold_iou_saved = best_threshold_iou
                # best_threshold_f1_saved = thr_f1

                # Umbral por precisión-píxel objetivo (segmentación)
                # y_true_pix = np.concatenate([m.ravel() for m in all_val_true_masks]).astype(np.uint8)
                # y_prob_pix = np.concatenate([p.ravel() for p in all_val_pred_probs]).astype(np.float32)
                # best_threshold_pix_prec_saved = pick_threshold_for_precision(
                #     y_true_pix, y_prob_pix, target_precision=TARGET_PIXEL_PRECISION
                # )

                # Umbral de clasificación por precisión objetivo (gate)
                # best_threshold_cls_prec_saved = find_threshold_for_cls_precision(
                #     y_true_cls, y_probs_cls, target_precision=CLS_GATE_TARGET_PRECISION
                # )

                # Log best thresholds
                # best_params = {
                #     "best_threshold_cls_f1": float(best_threshold_cls_saved),
                #     "best_threshold_iou": float(best_threshold_iou_saved),
                #     "best_threshold_seg_f1": float(best_threshold_f1_saved),
                #     "best_threshold_pix_precision": float(best_threshold_pix_prec_saved),
                #     "best_threshold_cls_precision": float(best_threshold_cls_prec_saved),
                #     "best_val_iou": float(best_val_iou)
                # }

                # Log checkpoint artifact
                if os.path.exists(BEST_MODEL_PATH):
                    mlflow.log_artifact(BEST_MODEL_PATH, artifact_path="checkpoints")

                # print(f"✅ Mejor modelo guardado con Val IoU: {val_iou:.4f}, "
                #       f"Best Thr Clasificación(F1): {best_threshold:.3f}, Best Thr IoU: {best_threshold_iou:.3f}, "
                #       f"Best Thr Pixel-F1: {thr_f1:.3f}")
                # print(f"🔧 Thresholds precisión: PixelPrecision={best_threshold_pix_prec_saved:.3f}, "
                #       f"Cls(gate)={best_threshold_cls_prec_saved:.3f}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"🛑 Early stopping en epoch {epoch+1}")
                break

        # mlflow.log_params(best_params)
        # metrics = inference_and_test_metrics(
        #     model=model,
        #     test_loader=test_loader,
        #     best_threshold_cls_saved=best_threshold_cls_saved,
        #     threshold=best_threshold_iou_saved,
        #     save_visuals=False,
        #     viz_limit=100,
        #     viz_out_dir=None,
        #     viz_tag="iou_opt",
        #     postprocess = False,
        #     params=params,
        #     save_dir=BEST_MODEL_DIR,
        #     state_dict_path=os.path.join(BEST_MODEL_DIR, f"best_model_element_{ELEMENT_INDEX}.pth")
        # )
        # for mname in ["IoU", "Dice", "Coverage", "FPRate", "AMD", "PixelRecall", "PixelPrecision", "PixelF1",
        #               "Hit0", "Hit1", "Hit25", "Hit5", "Hit75", "OversamplingObjects0", "OversamplingObjects1",
        #               "OversamplingObjects25", "OversamplingObjects50", "OversamplingObjects75"]:
        #     arr = metrics[mname]
        #     arr = arr[np.isfinite(arr)]
        #     if arr.size == 0:
        #         print(f"{mname}: sin datos finitos (NaN/Inf).")
        #         mlflow.log_metric(f"{mname}", float("nan"))
        #         continue
        #     mean = arr.mean()
        #     median = np.median(arr)
        #     p90 = np.percentile(arr, 90)
        #     p95 = np.percentile(arr, 95)
        #     print(
        #         f"{mname}: media={arr.mean():.4f}, mediana={np.median(arr):.4f},"
        #         f" p90={np.percentile(arr, 90):.4f}, p95={np.percentile(arr, 95):.4f}"
        #     )
        #     mlflow.log_metric(f"{mname}_mean", float(mean))
        #     mlflow.log_metric(f"{mname}_median", float(median))
        #     mlflow.log_metric(f"{mname}_p90", float(p90))
        #     mlflow.log_metric(f"{mname}_p95", float(p95))
        # Matriz de confusión clasificación
        # cm = metrics['confusion_matrix_cls']
        # cm = np.round(cm, 2)
        # np.savetxt("confusion_matrix_cls.csv", cm, delimiter=",")
        # mlflow.log_artifact("confusion_matrix_cls.csv")

if __name__ == "__main__":
    import random
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(42)

    main()