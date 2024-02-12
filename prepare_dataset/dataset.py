import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class MultiClassCarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))

        
        mask_paths = [os.path.join(self.mask_dir, self.images[index].replace(".jpg", f"_class{class_id}_mask.gif")) for class_id in range(num_classes)]
        
        masks = [np.array(Image.open(mask_path).convert("L"), dtype=np.float32) for mask_path in mask_paths]
        masks = [mask / 255.0 for mask in masks]  # Normalize masks to [0, 1]

        if self.transform is not None:
            # Apply the same transformation to both image and masks
            augmentations = self.transform(image=image, masks=masks)
            image = augmentations["image"]
            masks = augmentations["masks"]

        return image, masks
