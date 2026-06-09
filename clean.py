from PIL import Image
import os

VALID_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

def clean_image_folder(base_dir, min_width=500, min_height=400):
    total_removed = 0
    for class_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, class_folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if not is_image(file):
                continue
            file_path = os.path.join(folder_path, file)
            try:
                with Image.open(file_path) as img:
                    if img.width < min_width or img.height < min_height:
                        os.remove(file_path)
                        total_removed += 1
            except Exception:
                os.remove(file_path)
                total_removed += 1
    print(f"🧹 Removed {total_removed} low-quality/corrupt images")

def resize_images(base_dir, size=(512, 512)):
    for class_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, class_folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if not is_image(file):
                continue
            file_path = os.path.join(folder_path, file)
            try:
                with Image.open(file_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(size)
                    img.save(file_path)
            except Exception:
                os.remove(file_path)

def is_image(file):
    return any(file.lower().endswith(ext) for ext in VALID_EXTENSIONS)


def count_images_per_class(base_dir):
    total = 0
    for class_folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, class_folder)
        if os.path.isdir(folder_path):
            count = len([f for f in os.listdir(folder_path) if is_image(f)])
            print(f"{class_folder}: {count} images")
            total += count
    print(f"\n📦 Total images: {total}")

def main():
    base_dir = "images"
    print("🧹 Cleaning low-quality and corrupt images...")
    clean_image_folder(base_dir)
    
    print("📐 Resizing all images to 512x512...")
    resize_images(base_dir)

    print("📊 Counting images per class...")
    count_images_per_class(base_dir)

if __name__ == '__main__':
    main()
