
import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ── Augmentation pipelines ───────────────────────────────────────────────────

def get_train_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
                 rotate=(-15, 15), p=0.5),
        A.ElasticTransform(alpha=1, sigma=10, p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1, p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ── Dataset class ────────────────────────────────────────────────────────────

class DefectDataset(Dataset):
    def __init__(self, data_root, category, split='train',
                 img_size=256, val_ratio=0.2, random_seed=42):

        self.transform = (get_train_transforms(img_size) if split == 'train'
                          else get_val_transforms(img_size))

        image_paths, mask_paths = self._collect_pairs(data_root, category)

        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            image_paths, mask_paths,
            test_size=val_ratio,
            random_state=random_seed
        )

        if split == 'train':
            self.image_paths = train_imgs
            self.mask_paths  = train_masks
        else:
            self.image_paths = val_imgs
            self.mask_paths  = val_masks

        print(f"[{split}] {len(self.image_paths)} samples "
              f"({sum(m is not None for m in self.mask_paths)} with defects)")

    def _collect_pairs(self, data_root, category):
        image_paths, mask_paths = [], []
        test_root = Path(data_root) / category / 'test'
        gt_root   = Path(data_root) / category / 'ground_truth'

        for defect_type in sorted(test_root.iterdir()):
            for img_path in sorted(defect_type.glob('*.png')):
                image_paths.append(img_path)

                if defect_type.name == 'good':
                    mask_paths.append(None)
                else:
                    # ← FIX: use Path() constructor instead of + operator
                    mask_path = gt_root / defect_type.name / (img_path.stem + '_mask.png')
                    mask_paths.append(mask_path if mask_path.exists() else None)

        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_path = self.mask_paths[idx]
        if mask_path is None:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask  = augmented['mask'].unsqueeze(0)

        return image, mask


# ── DataLoader factory ───────────────────────────────────────────────────────

def get_dataloaders(data_root, category, img_size=256,
                    batch_size=8, num_workers=2, val_ratio=0.2):
    train_dataset = DefectDataset(data_root, category, split='train',
                                  img_size=img_size, val_ratio=val_ratio)
    val_dataset   = DefectDataset(data_root, category, split='val',
                                  img_size=img_size, val_ratio=val_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader
