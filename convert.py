import json
import numpy as np
from PIL import Image
import os

def rle_decode(mask_rle, shape):
    """
    Decodes an RLE encoded mask.
    
    Args:
        mask_rle (str): The RLE mask string.
        shape (tuple): The shape of the array to be returned (height, width).
    
    Returns:
        numpy.ndarray: The decoded mask.
    """
    s = list(map(int, mask_rle.split()))
    starts, lengths = s[::2], s[1::2]
    starts = np.array(starts) - 1
    ends = starts + lengths
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    return mask.reshape(shape).T  # Note: .T might be needed based on image/mask orientation

# Path to your JSON file
json_path = r'diagnostic_per_image.json'

# Path to your images directory
images_dir = r'resized_train'

# Path to save the mask images
masks_folder = r'resized_train_masks'

# Load JSON data
with open(json_path, 'r') as file:
    data = json.load(file)

for item in data:
    image_id = item['image_id']
    category_ids = item['CategoryId']
    encoded_pixels = item['EncodedPixels']
    
    # Assuming your images are named using the image_id and have a .jpg extension
    image_path = os.path.join(images_dir, f"{image_id}.jpg")
    with Image.open(image_path) as img:
        width, height = img.size
    
    # Initialize an empty mask for each category
    category_masks = {cat_id: np.zeros((height, width), dtype=np.uint8) for cat_id in range(8)}  # Assuming there are 8 categories in total
    
    # Decode and merge masks for each category
    for cat_id, mask_rle in zip(category_ids, encoded_pixels):
        decoded_mask = rle_decode(mask_rle, (height, width))
        category_masks[cat_id] += decoded_mask
    
    # Save the merged masks for each category
    for cat_id, merged_mask in category_masks.items():
        mask_img = Image.fromarray(merged_mask * 255)  # Convert to uint8
        mask_img_path = os.path.join(masks_folder, f"{image_id}_mask_{cat_id}.png")
        mask_img.save(mask_img_path)

print("Decoding complete.")
