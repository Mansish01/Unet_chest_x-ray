import os
import random
import shutil

def split_train_test(source_folder, train_folder, test_folder, split_ratio=0.8):
    # List all image files in the source folder
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    # Shuffle the list of image files
    random.shuffle(image_files)
    
    # Calculate the number of images for training and testing based on the split ratio
    num_train = int(len(image_files) * split_ratio)
    
    # Split the list of image files into training and testing sets
    train_images = image_files[:num_train]
    test_images = image_files[num_train:]
    
    # Copy images to train folder
    for image in train_images:
        shutil.copy(os.path.join(source_folder, image), os.path.join(train_folder, image))
    
    # Copy images to test folder
    for image in test_images:
        shutil.copy(os.path.join(source_folder, image), os.path.join(test_folder, image))

# Example usage
source_folder = "resized_train"
train_folder = "train_data"
test_folder = "test_data"

# Create train and test folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Split data
split_train_test(source_folder, train_folder, test_folder, split_ratio=0.8)
