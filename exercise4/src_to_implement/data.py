from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv
import torchvision.transforms as T
import pandas as pd
from typing import Union
from typing import Tuple
from tqdm.autonotebook import tqdm
from PIL import Image

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]

class InactiveContrastBoost:
    def __init__(self, active_label):
        self.active_label = active_label

    def __call__(self, img):
        if self.active_label == 1:
            return self.boost_dark_regions(img)
        return img

    def boost_dark_regions(self, img):
        np_img = np.array(img).astype(np.float32)
        dark_mask = np_img < 50
        np_img[dark_mask] *= 0.5  # tiefer verdunkeln
        np_img = np.clip(np_img, 0, 255).astype(np.uint8)
        return Image.fromarray(np_img)



class ChallengeDataset(Dataset):
    """
    Custom PyTorch dataset for solar panel defect detection.

    Args
    ----
    data : pandas.DataFrame
        Should contain at least the columns "image_path" and "label".
    mode : str
        Either "train" or "val", determines whether augmentations are applied.
    """

    def __init__(self, data: pd.DataFrame, mode: str = "train") -> None:
        super().__init__()
        if mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'")
        self.data = data.reset_index(drop=True)
        self.mode = mode

        # 1) Define image transformations
        pil_tfm = [
            T.ToPILImage()
        ]

        base_tfms = [
            T.ToTensor(),
            T.Normalize(mean=train_mean, std=train_std),
        ]

        if mode == "train":
            # Apply data augmentation in training mode
            # maybe add opening + closing and gaussian blurring here
            aug_tfms = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
            ]
            self.transform = T.Compose(pil_tfm + base_tfms)
        else:
            # No augmentation in validation mode
            self.transform = T.Compose(pil_tfm + base_tfms)

    def __len__(self) -> int:
        # Return number of samples
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get row from DataFrame
        record = self.data.iloc[index]

        # Load image from file
        img_path = Path(record["filename"])
        img = imread(img_path)

        # Boost if inactive == 1
        if self.mode == "train" and record["inactive"] == 1:
            pil_img = Image.fromarray(img).convert("L")  # Ensure grayscale
            boosted = InactiveContrastBoost(active_label=1)(pil_img)
            img = np.array(boosted)

            save_dir = Path("boosted_previews")
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"boosted_{img_path.stem}.png"
            boosted.save(save_path)

        # Convert grayscale image to RGB if necessary
        if img.ndim == 2 or img.shape[-1] == 1:
            img = gray2rgb(img)

        # Ensure image is in uint8 format (required by ToPILImage)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)

        # Load label and convert to tensor
        label_crack: Union[int, list, np.ndarray] = record["crack"]
        label_inactive: Union[int, list, np.ndarray] = record["inactive"]
        label_tensor = torch.as_tensor((label_crack, label_inactive), dtype=torch.long)

        # Apply transformations to image
        img_tensor = self.transform(img)


        return img_tensor, label_tensor
    

class DataAugmenter:
    def __init__(self, output_dir="augmented", num_augs=1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.num_augs = num_augs

        self.aug_tfs = [
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.2),
        ]
        self.transform = T.Compose(self.aug_tfs)

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        new_rows = []

        for idx, row in df.iterrows():
            filename = Path(row["filename"])
            img = imread(filename)

            # Convert to RGB if needed
            if img.ndim == 2 or img.shape[-1] == 1:
                img = gray2rgb(img)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)

            for i in range(self.num_augs):
                img_aug = self.transform(img)

                # Create new filename
                stem = filename.stem
                suffix = filename.suffix if filename.suffix else ".png"
                new_name = f"{stem}_aug{i+1}{suffix}"
                save_path = self.output_dir / new_name

                # Save image
                img_aug.save(save_path)

                # Append new row
                new_rows.append({
                    "filename": str(save_path),
                    "crack": row["crack"],
                    "inactive": row["inactive"],
                    "class_combo": row["class_combo"]
                })

        return pd.DataFrame(new_rows)