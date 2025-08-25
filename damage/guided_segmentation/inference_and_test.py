import segmentation_models_pytorch as smp
import albumentations as A
from albumentations import ToTensorV2

from damage.datasets.dataset import CSVCroppedImagesDataset
from damage.guided_segmentation.models import DualUNetPlusPlusGuided
from damage.guided_segmentation.utils import inference_and_test_metrics
import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
import mlflow

IMAGE_SIZE = 320
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Rutas (ajusta a tu entorno)
IMAGE_DIR = "/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/cropped_images/"
MASK_DIR = "/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/cropped_masks"
CSV_INFERENCE_PATH = "/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/split/element_4/test.csv"
# STATE_DICT_PATH = "/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/test_guided_segmentation_element_4/fix/best_model_element_4.pth"
# BEST_MODEL_DIR = f"/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/test_guided_segmentation_element_4/fix"

# Elemento que estás segmentando/clasificando
ELEMENT_INDEX = "4"

IMAGE_SIZE = 320
BATCH_SIZE = 8
NUM_WORKERS = 8

BACKBONE = 'efficientnet-b0'
def main():
    params = smp.encoders.get_preprocessing_params(BACKBONE)
    transform =  A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=params['mean'], std=params['std']),
        ToTensorV2()
    ])
    df = pd.read_csv(CSV_INFERENCE_PATH)
    df_pos = df[df[ELEMENT_INDEX] == True].reset_index(drop=True)
    print(f"Total test={len(df)}, solo positivos={len(df_pos)}")

    test_dataset = CSVCroppedImagesDataset(
        df_pos, IMAGE_DIR, MASK_DIR, transform=transform, element_index=ELEMENT_INDEX
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    model = DualUNetPlusPlusGuided(
        backbone_name=BACKBONE, pretrained=True, in_channels=3, seg_classes=1, cls_classes=1
    ).to(DEVICE)


    # Umbrales óptimos obtenidos en validación
    # Umbrales
    # Por ejemplo, los "precision-first":
    BEST_THRESHOLD_CLS_PREC_SAVED = 0.86  # Cls(gate)
    BEST_THRESHOLD_PIX_PREC_SAVED = 0.7  # PixelPrecision

    threshols_cls = [0.5, 0.6, 0.7, 0.8, 0.86, 0.9]
    threshols_pix = [0.5, 0.6, 0.7, 0.8, 0.9]

    # Post-procesado
    POSTPROCESSING = False
    POST_KEEP_TOP = 1
    POST_MIN_AREA = 64
    POST_APPLY_OPENING = False
    POST_OPENING_ITER = 1
    SAVE_DIR = (f"/ceph04/ml/property_damage_elements/datasets/ds-25-06-09_all_classes_MEDIUM_wrong_nir_fixes/"
                f"test_guided_segmentation_element_4/fix/inference_outputs_element_{ELEMENT_INDEX}/testthreshold_pixprec")
    os.makedirs(SAVE_DIR, exist_ok=True)

    SAVE_VISUALS = False


    for thr_pixel_precision in threshols_pix:
        for thr_cls in threshols_cls:
            print(f"Testing {thr_pixel_precision} pixel precision - {thr_cls} cls")
            metrics = inference_and_test_metrics(
                model=model,
                test_loader=test_loader,
                best_threshold_cls_saved=thr_cls,
                threshold=thr_pixel_precision,
                save_visuals=SAVE_VISUALS,
                viz_limit=100,
                viz_tag="pix_prec_opt",
                postprocess=POSTPROCESSING,
                state_dict_path=STATE_DICT_PATH,   # IMPORTANTE: aquí cargas los pesos
                device=DEVICE,
                element_index=ELEMENT_INDEX,
                save_dir=SAVE_DIR,
                postprocess_keep_top=POST_KEEP_TOP,
                postprocess_min_area=POST_MIN_AREA,
                postprocess_apply_opening=POST_APPLY_OPENING,
                postprocess_opening_iter=POST_OPENING_ITER,
                params=params,  # Para denormalizar imágenes en los paneles
            )




if __name__ == '__main__':
    import random
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(42)
    main()
