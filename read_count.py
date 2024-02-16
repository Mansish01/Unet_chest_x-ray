import json

# Replace 'your_file_path.json' with the actual path to your JSON file
file_path = r'diagnostic_per_image.json'

# Read and load JSON data
with open(file_path, 'r') as file:
    data = json.load(file)


# Count occurrences of each unique (image_id, CategoryId) pair
category_count = {}

for entry in data:
    image_id = entry["image_id"]
    category_ids = set(entry["CategoryId"])
    for category_id in category_ids:
        category_count.setdefault(category_id, set()).add(image_id)

# Print the counts
for category_id, image_ids in category_count.items():
    print(f"Category {category_id}: {len(image_ids)} occurrences")