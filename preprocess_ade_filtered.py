import os
import cv2
import numpy as np
from tqdm import tqdm
import random
from torchvision import transforms
from PIL import Image
import torch

IMG_SIZE = 512

def augment_image(img_pil, is_window_dataset=False):
    aug_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
    ]
    if is_window_dataset:
        aug_list.append(transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5
        ))
    else:
        aug_list.append(transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ))
    augment = transforms.Compose(aug_list)
    img_aug = augment(img_pil)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return to_tensor(img_aug)

def unnormalize(tensor):
    tensor = tensor.clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0,1)
    return tensor

def process_images_in_folder(input_folder, output_folder, is_window_dataset=False):
    os.makedirs(output_folder, exist_ok=True)
    img_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img_file in tqdm(img_files, desc=f"Processing {os.path.basename(input_folder)}"):
        img_path = os.path.join(input_folder, img_file)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_tensor = augment_image(img, is_window_dataset)
        img_unnorm = unnormalize(img_tensor)
        img_np = (img_unnorm.numpy() * 255).astype(np.uint8)
        img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC format
        save_path = os.path.join(output_folder, img_file)
        cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

def process_dataset_root(dataset_root, output_root, is_window_dataset=False):
    for split in ['training', 'validation']:
        input_img_folder = os.path.join(dataset_root, split, 'img')
        output_img_folder = os.path.join(output_root, split, 'img')
        if os.path.exists(input_img_folder):
            print(f"Processing {dataset_root} - {split}")
            process_images_in_folder(input_img_folder, output_img_folder, is_window_dataset)
        else:
            print(f"Warning: {input_img_folder} does not exist")


def main():
    base_path = "/home/piyush/Desktop/AI Furnishing Visualizer"
    datasets = {
        "ADE20K_Filtered_Bed_Dataset": False,
        "ADE20K_Filtered_Floor_Dataset": False,
        "ADE20K_Filtered_Window_Dataset": True,
        "ADE20K_Filtered_Ceiling_Dataset": False,
        "ADE20K_Filtered_Wall_Dataset": False,
    }
    output_root = os.path.join(base_path, "Preprocessed_Datasets")
    for dataset_name, is_window in datasets.items():
        dataset_path = os.path.join(base_path, dataset_name)
        output_path = os.path.join(output_root, dataset_name)
        os.makedirs(output_path, exist_ok=True)
        process_dataset_root(dataset_path, output_path, is_window)
    print("All datasets processed.")

if __name__ == "__main__":
    main()
