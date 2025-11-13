import os
import cv2
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset


class RRGDataset(Dataset):
    def __init__(self, images, tokenized_reports, labels=None):
        self.images = images
        self.tokenized_reports = tokenized_reports
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        report = self.tokenized_reports[idx]
        if self.labels is not None:
            label = self.labels[idx]
            return image, report, label
        else:
            return image, report

def preprocess_and_load_rrg_data(csv_path, img_dir, tokenizer, img_size=(256, 256), max_length=100):
    print("Preprocessing data and loading images for RRG...")
    df = pd.read_excel(csv_path) if csv_path.endswith(('.xlsx', '.xls')) else pd.read_csv(csv_path)
    df['LABEL'] = df['LABEL'].astype(str).str.strip().str.replace(' ', '')
    df = df[df['LABEL'] != 'CND'].reset_index(drop=True)
    unique_labels = sorted(df['LABEL'].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    images, reports, labels = [], [], []
    df['IMG_ID_PADDED'] = df['IMG_ID'].astype(str).str.zfill(4)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading RRG Data"):
        padded_id = row['IMG_ID_PADDED']
        report_text = str(row.get('report_maira_2', row.get('REPORT', "No findings.")))
        label = row['LABEL']
        found_file = False
        
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(img_dir, f"{padded_id}{ext}")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB').resize(img_size)
                    img_array = np.array(img).astype(np.float32) / 255.0
                    images.append(torch.from_numpy(img_array).permute(2, 0, 1).float())
                    
                    formatted_text = f"{tokenizer.cls_token} {report_text} {tokenizer.sep_token}"
                    tokenized = tokenizer.encode_plus(
                        formatted_text, 
                        padding='max_length', 
                        truncation=True,
                        max_length=max_length, 
                        return_tensors='pt',
                        add_special_tokens=False
                    )
                    reports.append(tokenized['input_ids'].squeeze(0))
                    labels.append(label_map[label])
                    found_file = True
                    break
                except Exception as e:
                    print(f"Error for ID {padded_id}: {e}")
        if not found_file:
            print(f"Warning: Image file for ID {padded_id} not found.")
    
    if not images:
        return None, None, None
    return torch.stack(images), torch.stack(reports), torch.tensor(labels, dtype=torch.long)

def get_class_info(csv_path):
    try:
        df = pd.read_excel(csv_path) if csv_path.endswith(('.xlsx', '.xls')) else pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file at {csv_path} was not found.")
        return 0, []
    if 'LABEL' not in df.columns:
        print("Error: 'LABEL' column not found.")
        return 0, []
    df['LABEL'] = df['LABEL'].astype(str).str.strip().str.replace(' ', '')
    df = df[df['LABEL'] != 'CND']
    unique_labels = sorted(df['LABEL'].unique())
    return len(unique_labels), unique_labels

def preprocess_and_load_data(csv_path, img_dir, img_size=(256, 256)):
    print("Preprocessing data and loading images...")

    if csv_path.endswith('.xlsx') or csv_path.endswith('.xls'):
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)

    print(f"Step 1: Initial number of rows in CSV: {len(df)}")

    df['LABEL'] = df['LABEL'].astype(str).str.strip().str.replace(' ', '')
    
    df = df[df['LABEL'] != 'CND'].reset_index(drop=True)

    print(f"Step 2: Number of rows after removing 'CND' labels: {len(df)}")

    df['IMG_ID_PADDED'] = df['IMG_ID'].astype(str).str.zfill(4)

    unique_labels = sorted(df['LABEL'].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    print(f"Detected {num_classes} classes: {unique_labels}")

    images = []
    labels = []

    print(f"Attempting to load images from: {img_dir}")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading Classification Data"):
        padded_id = row['IMG_ID_PADDED']
        label = row['LABEL']

        found_file = False
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(img_dir, f"{padded_id}{ext}")
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path).convert('RGB').resize(img_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    images.append(img_array)
                    labels.append(label_map[label])
                    found_file = True
                    break
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

        if not found_file:
            print(f"Warning: Image file for ID {padded_id} not found. Skipping.")

    print(f"Step 3: Final number of images successfully loaded: {len(images)}")

    if not images:
        return np.array([]), np.array([]), 0, []

    return np.stack(images), np.array(labels), num_classes, unique_labels

class LungClassificationDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

def preprocess_and_load_data_seg(image_path, mask_path, img_size=(256, 256)):
    print("Preprocessing data and loading images for segmentation...")
    images, masks = [], []
    valid_exts = (".png", ".jpg", ".jpeg")
    # print(len([f for f in os.listdir(image_path) if f.lower().endswith(('.png'))]))
    # print(len([f for f in os.listdir(mask_path) if f.lower().endswith(('.png'))]))
    image_dict = {os.path.splitext(f)[0]: f for f in os.listdir(image_path) if f.lower().endswith(valid_exts)}
    mask_dict  = {os.path.splitext(f)[0]: f for f in os.listdir(mask_path) if f.lower().endswith(valid_exts)}
    matching_names = sorted(image_dict.keys() & mask_dict.keys())
    print(f"Found {len(matching_names)} matching image–mask pairs")

    for name in tqdm(matching_names, desc="Loading Segmentation Data"):
        try:
            image = cv2.imread(os.path.join(image_path, image_dict[name]), cv2.IMREAD_COLOR)
            mask = cv2.imread(os.path.join(mask_path, mask_dict[name]), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None: continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, img_size, interpolation=cv2.INTER_LINEAR) / 255.0
            mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)

            if mask.sum() == 0: continue

            images.append(image)
            masks.append(mask)
        except Exception as e:
            print(f"Skipping {name} due to error: {e}")

    return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)

class LungSegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float()
        mask = torch.from_numpy(self.masks[idx]).unsqueeze(0).float()
        return image, mask