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

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


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
    """
    Augmentiert unterrepräsentierte Klassen (crack oder inactive) durch gezielte Bildtransformationen.
    Speichert die neuen Bilder ab und gibt ein erweitertes DataFrame zurück.
    """

    def __init__(self, df: pd.DataFrame, base_path: str = "", output_dir: str = "augmented"):
        self.df = df.copy()
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.augmentations = [
            ("flip_h", T.RandomHorizontalFlip(p=1.0)),
            ("flip_v", T.RandomVerticalFlip(p=1.0)),
            ("rot_15", T.RandomRotation(degrees=15)),
        ]

    def augment_minority_classes(self) -> pd.DataFrame:
        # Filter nur crack oder inactive == 1
        minority_df = self.df[(self.df["crack"] == 1) | (self.df["inactive"] == 1)]

        new_rows = []
        for _, row in tqdm(minority_df.iterrows(), total=len(minority_df), desc="Augmenting"):
            img_path = self.base_path / row["filename"]
            image = imread(img_path)

            # Graustufenbilder zu RGB
            if image.ndim == 2 or image.shape[-1] == 1:
                image = gray2rgb(image)

            # Konvertieren in PIL-Format für torchvision Transforms
            pil_image = T.ToPILImage()(image)

            for suffix, aug in self.augmentations:
                transformed = aug(pil_image)
                new_name = img_path.stem + f"_{suffix}.png"
                save_path = self.output_dir / new_name
                transformed.save(save_path)

                # Neue Zeile für DataFrame
                new_rows.append({
                    "filename": str(save_path),
                    "crack": row["crack"],
                    "inactive": row["inactive"]
                })

        # Kombinieren mit Original
        df_augmented = pd.DataFrame(new_rows)
        df_combined = pd.concat([self.df, df_augmented], ignore_index=True)
        return df_combined
