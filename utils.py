import torch
import torchvision
from dataset import MultiClassDataset
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_dataset = MultiClassDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_dataset = MultiClassDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader




def check_accuracy(loader, model, device="DEVICE"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    num_classes = 8  # Number of classes

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)  # Assuming y is in the same format as model predictions

            # Forward pass
            preds = model(x)

            # Convert logits to predictions by taking argmax along the class dimension
            preds = torch.argmax(preds, dim=1)

            # Calculate accuracy
            num_correct += torch.sum(preds == y).item()  # Convert to Python number
            num_pixels += torch.numel(preds)

            # Calculate Dice score for each class
            for cls in range(num_classes):
                intersection = torch.sum((preds == cls) & (y == cls))
                cls_union = torch.sum(preds == cls) + torch.sum(y == cls)
                dice_score += (2. * intersection + 1e-8) / (cls_union + 1e-8)

    # Calculate overall accuracy and average Dice score across all classes
    accuracy = num_correct / num_pixels * 100
    dice_score /= (len(loader) * num_classes)

    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"Average Dice score: {dice_score.item()}")

    model.train()


# def save_predictions_as_imgs(
#     loader, model, folder="saved_images/", device="cuda"
# ):
#     model.eval()
#     for idx, (x, y) in enumerate(loader):
#         x = x.to(device=device)
#         with torch.no_grad():
#             preds = torch.sigmoid(model(x))
#             preds = (preds > 0.5).float()
#         torchvision.utils.save_image(
#             preds, f"{folder}/pred_{idx}.png"
#         )
#         torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

#     model.train()
    

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="DEVICE"):
    model.eval()
    color_map = [  # Define color map for each class
        [0, 0, 0],      # Class 0: Background (Black)
        [255, 0, 0],    # Class 1: Red
        [0, 255, 0],    # Class 2: Green
        [0, 0, 255],    # Class 3: Blue
        [255, 255, 0],  # Class 4: Yellow
        [255, 0, 255],  # Class 5: Magenta
        [0, 255, 255],  # Class 6: Cyan
        [128, 128, 128] # Class 7: Gray
        # Add colors for additional classes as needed
    ]

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = torch.argmax(preds, dim=1)  # Convert logits to class labels

        # Create color-coded prediction image
        pred_img = torch.zeros_like(x)
        for cls, color in enumerate(color_map):
            pred_img[preds == cls] = torch.tensor(color, dtype=torch.float32) / 255.0

        # Save prediction and ground truth images
        torchvision.utils.save_image(pred_img, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.float(), f"{folder}/{idx}.png")  # Assuming y is already in image format

    model.train()



# if __name__ == "__main__":
