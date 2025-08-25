import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class CSVCroppedImagesDataset(Dataset):
    def __init__(self, df, image_dir, mask_dir, transform=None, element_index=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.element_index = element_index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_name'])
        mask_path = os.path.join(self.mask_dir, row['image_name'].replace(".png", "_mask.png"))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask_raw = np.array(Image.open(mask_path))  # NO .convert("L") si pierdes IDs
        mask = (mask_raw == int(self.element_index)).astype(np.float32)

        label = row[self.element_index]
        label = 1 if label in [True, "True", 1] else 0
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image'].float()
            mask = augmented['mask'].float().unsqueeze(0)

        return image, mask, label
