import os
import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from mlflow.artifacts import download_artifacts
import mlflow
from urllib.parse import quote
from pyvexcelutils.aws.secrets import Secrets

from damage.datasets.dataset import CSVCroppedImagesDataset
from damage.guided_segmentation.models import DualUNetPlusPlusGuided
from damage.guided_segmentation.utils import inference_and_test_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 320
BATCH_SIZE = 8
NUM_WORKERS = 8
BACKBONE = "efficientnet-b0"

# Datos
ELEMENT_INDEX = "7"
IMAGE_DIR = "/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/cropped_images/"
MASK_DIR = "/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/cropped_masks"
CSV_TEST = f"/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/split/element_{ELEMENT_INDEX}/test.csv"

# Checkpoint en MLflow
ARTIFACT_URI = (
    "mlflow-artifacts:/property-damage/guided-segmentation-by-classification/binary/"
    "4d0634f9cf924a98a0e66659ebb47d04/artifacts/checkpoints/best_model_element_7.pth"
)

# Malla de thresholds
CLS_GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.798, 0.85, 0.9]
SEG_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# Definición del objetivo: max Hit15_with_gt_mean y penaliza oversampling y FP en no_gt
ALPHA_OVERSAMP = 0.5    # peso penalización de oversampling
BETA_FPR_NO_GT = 0.2    # penalización de FPR en no-GT

# Restricción (opcional): filtra configuraciones con oversampling50_with_gt_mean <= este valor
MAX_OVERSAMP50_MEAN = 0.6

SAVE_CSV = f"./grid_search_element_{ELEMENT_INDEX}.csv"

def resolve_mlflow_artifact(artifact_uri: str) -> str:
    username, password, url = Secrets().get_mlflow_user()
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    mlflow.set_tracking_uri(url)
    local_path = download_artifacts(artifact_uri=artifact_uri)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"No se pudo descargar el artifact desde {artifact_uri}")
    print(f"Checkpoint descargado en: {local_path}")
    return local_path

def main():
    # Dataset completo (no filtrar a solo positivos, para medir no_gt)
    params = smp.encoders.get_preprocessing_params(BACKBONE)
    transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=params['mean'], std=params['std']),
        ToTensorV2()
    ])
    df_test = pd.read_csv(CSV_TEST)
    test_dataset = CSVCroppedImagesDataset(df_test, IMAGE_DIR, MASK_DIR, transform=transform, element_index=ELEMENT_INDEX)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = DualUNetPlusPlusGuided(
        backbone_name=BACKBONE, pretrained=True, in_channels=3, seg_classes=1, cls_classes=1
    ).to(DEVICE)

    state_dict_path = resolve_mlflow_artifact(ARTIFACT_URI)

    rows = []
    for thr_seg in SEG_GRID:
        for thr_cls in CLS_GRID:
            print(f"\n=== Evaluando cls={thr_cls:.3f} seg={thr_seg:.3f} ===")
            metrics = inference_and_test_metrics(
                model=model,
                test_loader=test_loader,
                best_threshold_cls_saved=thr_cls,
                threshold=thr_seg,
                save_visuals=False,
                viz_limit=0,
                viz_tag=f"grid_cls{thr_cls:.2f}_seg{thr_seg:.2f}",
                postprocess=False,
                state_dict_path=state_dict_path,
                device=DEVICE,
                element_index=int(ELEMENT_INDEX),
                save_dir=".",
                postprocess_keep_top=1,
                postprocess_min_area=64,
                postprocess_apply_opening=False,
                postprocess_opening_iter=1,
                params=params,
                hit_iou_thresholds=(0.1, 0.15, 0.25, 0.5),
                oversampling_iou_thresholds=(0.1, 0.15, 0.25, 0.5),
            )
            sbg = metrics["summary_by_gt"]  # dict por métrica -> with_gt/no_gt stats

            # Lectura robusta de estadísticas (devuelve np.nan si falta)
            def stat(mname, subset, statname):
                try:
                    return float(sbg[mname][subset][statname])
                except Exception:
                    return float("nan")

            # Métricas objetivo
            hit15_mean_wg = stat("Hit15", "with_gt", "mean")
            oversamp50_mean_wg = stat("OversamplingObjects50", "with_gt", "mean")
            fpr_no_gt_mean = stat("FPRate", "no_gt", "mean")

            # Otras útiles para inspección
            hit25_mean_wg = stat("Hit25", "with_gt", "mean")
            pixelf1_mean_wg = stat("PixelF1", "with_gt", "mean")
            iou_mean_wg = stat("IoU", "with_gt", "mean")

            # Penalización / restricción
            if not np.isnan(oversamp50_mean_wg) and oversamp50_mean_wg <= MAX_OVERSAMP50_MEAN:
                score = (hit15_mean_wg
                         - ALPHA_OVERSAMP * (oversamp50_mean_wg if not np.isnan(oversamp50_mean_wg) else 0.0)
                         - BETA_FPR_NO_GT * (fpr_no_gt_mean if not np.isnan(fpr_no_gt_mean) else 0.0))
            else:
                score = -np.inf  # descarta configuraciones que violan la restricción

            rows.append({
                "thr_cls": thr_cls,
                "thr_seg": thr_seg,
                "score": score,
                "Hit15_with_gt_mean": hit15_mean_wg,
                "Hit25_with_gt_mean": hit25_mean_wg,
                "Oversamp50_with_gt_mean": oversamp50_mean_wg,
                "FPR_no_gt_mean": fpr_no_gt_mean,
                "PixelF1_with_gt_mean": pixelf1_mean_wg,
                "IoU_with_gt_mean": iou_mean_wg,
            })

    df = pd.DataFrame(rows)
    df.sort_values("score", ascending=False, inplace=True)
    print("\nTop 10 configuraciones por score:")
    print(df.head(10).to_string(index=False))
    df.to_csv(SAVE_CSV, index=False)
    print(f"\nGuardado CSV: {SAVE_CSV}")

if __name__ == "__main__":
    import random
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(42)
    main()