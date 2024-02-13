import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

class MultiClassDataset(Dataset):
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

        
        mask_paths = [os.path.join(self.mask_dir, self.images[index].replace(".jpg", f"_mask_{class_id}.png")) for class_id in range(num_classes)]
        
        masks = [np.array(Image.open(mask_path).convert("L"), dtype=np.float32) for mask_path in mask_paths]
        masks = [mask / 255.0 for mask in masks]  # Normalize masks to [0, 1]

        if self.transform is not None:
            # Apply the same transformation to both image and masks
            augmentations = self.transform(image=image, masks=masks)
            image = augmentations["image"]
            masks = augmentations["masks"]

        return image, masks


if __name__ == "__main__":


    # Assuming num_classes is defined globally for simplicity
    num_classes = 8  # Update this based on your actual number of classes

    # Initialize your dataset
    dataset = MultiClassDataset(
        image_dir=r'image_test',  # Update this path
        mask_dir=r'mask_test',    # Update this path
        transform=A.Compose([
            A.Resize(height=512, width=512),
            A.Rotate(limit=30, p=1.0),
            # A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ])
    )

    # dataset = MultiClassDataset(
    #     image_dir=r'image_test',  # Update this path
    #     mask_dir=r'mask_test'   # Update this path
    #     )

    # Function to display images and masks
    def show_image_and_masks(image, masks):
        plt.figure(figsize=(15, 10))
        plt.subplot(1, num_classes + 1, 1)
        plt.imshow(image)
        plt.title('Image')
        plt.axis('off')
        
        for i, mask in enumerate(masks, start=1):
            plt.subplot(1, num_classes + 1, i + 1)
            plt.imshow(mask, cmap='gray')
            plt.title(f'Mask {i-1}')
            plt.axis('off')
        
        plt.show()

    # Fetch the first item
    image, masks = dataset[1]

    # Convert image back to PIL Image for plotting
    image = transforms.ToPILImage()(image)

    # Show the image and masks
    show_image_and_masks(image, masks)

