import os
import json
import shutil
from tqdm import tqdm

def filter_ade20k_dataset(src_img_dir, src_ann_dir, dst_img_dir, dst_ann_dir, classes_to_keep):
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_ann_dir, exist_ok=True)
    ann_files = [f for f in os.listdir(src_ann_dir) if f.endswith('.json')]
    count = 0
    print(f"Filtering for classes: {classes_to_keep}")
    print(f"Total annotation files to process: {len(ann_files)}")

    for ann_file in tqdm(ann_files, desc="Filtering annotations"):
        ann_path = os.path.join(src_ann_dir, ann_file)
        try:
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
            objects = ann_data.get('objects', [])
            image_classes = set()
            for obj in objects:
                if isinstance(obj, dict):
                    class_title = obj.get('classTitle', '').lower()
                    if class_title:
                        image_classes.add(class_title)

            if classes_to_keep.intersection(image_classes):
                # Extract image filename by stripping '.json' from annotation file name
                img_filename = ann_file[:-5]

                src_img_path = os.path.join(src_img_dir, img_filename)
                dst_img_path = os.path.join(dst_img_dir, img_filename)
                dst_ann_path = os.path.join(dst_ann_dir, ann_file)

                if os.path.exists(src_img_path):
                    shutil.copy2(src_img_path, dst_img_path)
                    shutil.copy2(ann_path, dst_ann_path)
                    count += 1
                else:
                    print(f"WARNING: Image file missing: {src_img_path}")

        except Exception as e:
            print(f"ERROR processing {ann_file}: {e}")

    print(f"Finished filtering. Total matched images: {count}")

if __name__ == "__main__":
    base_path = ""
    src_train_img = os.path.join(base_path, "ADE20K_Dataset/training/img")
    src_train_ann = os.path.join(base_path, "ADE20K_Dataset/training/ann")
    src_val_img = os.path.join(base_path, "ADE20K_Dataset/validation/img")
    src_val_ann = os.path.join(base_path, "ADE20K_Dataset/validation/ann")

    #out_base = os.path.join(base_path, "ADE20K_Filtered_Dataset")
    out_base = os.path.join(base_path, "ADE20K_Filtered_Floor_Dataset")
    out_train_img = os.path.join(out_base, "training/img")
    out_train_ann = os.path.join(out_base, "training/ann")
    out_val_img = os.path.join(out_base, "validation/img")
    out_val_ann = os.path.join(out_base, "validation/ann")

    #filter_classes = {"wall", "floor", "ceiling", "bed", "window"}

    filter_classes = {"floor"}

    print("Filtering TRAINING set:")
    filter_ade20k_dataset(src_train_img, src_train_ann, out_train_img, out_train_ann, filter_classes)

    print("Filtering VALIDATION set:")
    filter_ade20k_dataset(src_val_img, src_val_ann, out_val_img, out_val_ann, filter_classes)
