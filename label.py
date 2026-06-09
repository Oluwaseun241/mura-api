import os
import csv

def generate_labels_csv(base_dir="images", output_csv="labels.csv"):
    rows = []
    for class_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, class_folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                rows.append([f"{class_folder}/{filename}", class_folder])

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(rows)

    print(f"✅ Generated '{output_csv}' with {len(rows)} entries")

generate_labels_csv()
