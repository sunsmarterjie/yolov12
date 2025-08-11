import os
import random
import shutil

def split_class_folders(
    source_dir="datasets/beans",
    output_dir="datasets/beans_split",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
    random.seed(seed)

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)

        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:],
        }

        for split_name, files in splits.items():
            split_cls_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_cls_dir, exist_ok=True)

            for file in files:
                src_file = os.path.join(cls_path, file)
                dst_file = os.path.join(split_cls_dir, file)
                shutil.copy(src_file, dst_file)

    print("Dataset split by class folders completed.")

if __name__ == "__main__":
    split_class_folders()
