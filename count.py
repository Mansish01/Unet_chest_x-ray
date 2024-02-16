import os
import json

# Step 1: Read the JSON file and extract category IDs
json_file_path = r'diagnostic_per_image.json'
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

category_counts = {}  # Dictionary to store category counts

# Step 2: Get the list of image files in the image directory
image_dir = r'resized_train'
image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

# Step 3: Count unique category IDs associated with images
for entry in data:
    image_id = entry.get("image_id")
    category_ids = entry.get("CategoryId", [])

    # Check if the image is present in the image directory
    if f"{image_id}.jpg" in image_files:  # Assuming image files have ".jpg" extension
        # Count unique category IDs for each image
        unique_category_ids = set(category_ids)
        for category_id in unique_category_ids:
            category_counts[category_id] = category_counts.get(category_id, 0) + 1

# Print the result
print("Category Counts:")
for category_id, count in category_counts.items():
    print(f"Category {category_id}: {count} images")
