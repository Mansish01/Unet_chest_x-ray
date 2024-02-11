import numpy as np
from PIL import Image

def rle_decode(mask_rle, shape):
    """
    Decodes an RLE encoded mask.
    
    Args:
        mask_rle (str): The RLE mask string.
        shape (tuple): The shape of the array to be returned (height, width).
    
    Returns:
        numpy.ndarray: The decoded mask.
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Note: .T might be needed based on image orientation

# Example data from your JSON (first element only)
image_id = "47ed17dcb2cbeec15182ed335a8b5a9e"
encoded_pixels = "57192 20 57704 20 58216 20 58728 20 59240 20 59752 20 60264 20 60776 20 61288 20 61800 20 62312 20 62824 20 63336 20 63848 20 64360 20 64872 20 65384 20 65896 20 66408 20 66920 20 67432 20 67944 20 68456 20"

# Assuming you know the image size, for example:
width, height = 512, 512  # Replace with your actual image dimensions

decoded_mask = rle_decode(encoded_pixels, (height, width))

# Convert to an image and save
mask_img = Image.fromarray(decoded_mask * 255)  # Scale values for visualization
mask_img.save(f"{image_id}_mask.png")

print("Mask image saved.")
