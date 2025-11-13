import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np
from PIL import Image
import cv2
import csv
import math
import json, re
import pandas as pd
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..utils.prediction import analyze_segmentation_mask, predict_single_image_rrg
from ..utils.vqa_utils import get_vqa_answers_from_llm
from ..utils.data_utils import get_class_info
from ..models import UnifiedMultiTaskModel

def run_full_pipeline_for_single_image(
    image_path,
    seg_model, class_model, rrg_model,
    llm_model, llm_tokenizer,
    rrg_tokenizer, class_labels,
    fixed_questions, icl_examples,
    device, img_size,
    mask_output_dir=None
):
    print("-" * 50)
    print(f"Processing image: {os.path.basename(image_path)}")
    seg_model.eval()
    class_model.eval()
    rrg_model.eval()
    
    try:
        image = Image.open(image_path).convert('RGB').resize(img_size)
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None

    result = {"image": os.path.basename(image_path)}
    
    with torch.no_grad():
        print("1. Running Segmentation...")
        seg_output, _ = seg_model(image_tensor, mode='seg')
        binary_mask = (torch.sigmoid(seg_output).squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        location_info = analyze_segmentation_mask(binary_mask)
        
        if mask_output_dir:
            os.makedirs(mask_output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(mask_output_dir, f"{base}_mask.png")
            cv2.imwrite(mask_path, binary_mask)
            result["mask_path"] = mask_path
        else:
            result["mask_path"] = "Not saved"

        print("2. Running Classification...")
        _, logits = class_model(image_tensor, mode='class')
        probs = F.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        predicted_class_label = class_labels[predicted_class_idx]
        result["predicted_class"] = predicted_class_label
        result["confidence"] = probs.max().item()

        print("3. Generating Initial Report...")
        initial_report = predict_single_image_rrg(
            rrg_model, image_path, device, img_size, rrg_tokenizer, 
            max_length=100, class_label_idx=predicted_class_idx
        )
        if not initial_report: initial_report = "Report could not be generated."
        
        print("4. Enhancing Report with Findings...")
        enhanced_report = (
            f"{initial_report} "
            f"Diagnostic analysis suggests the presence of {predicted_class_label}."
            f"{f' Based on the segmentation mask, {location_info}.' if predicted_class_label != 'Normal' else ''}"
        )
        result["enhanced_report"] = enhanced_report

        print("5. Performing VQA on Enhanced Report...")
        predicted_answers = get_vqa_answers_from_llm(
            llm_model, llm_tokenizer, enhanced_report, fixed_questions, icl_examples, device
        )
        
        for q, a in zip(fixed_questions, predicted_answers):
            result[q] = a
        print("   -> VQA complete.")

    return result

def run_predict_full_mode(args, device):
    print(f"🚀 Starting Full Prediction Mode on folder: {args.image_folder}")

    FIXED_QUESTIONS = [
        "Which lung is affected by this abnormality?",
        "What is the extent of infected region compared to total lung area?",
        "What is the severity of disease based on the % of lung involvement?",
        "What abnormalities are detected in this image?",
        "What disease or condition is identified in this image?",
        "What are the key findings from report?",
        "What additional observation about lung field, heart size or pleural space?",
        "What is final diagnostic decision?"
    ]
    print(f"Using {len(FIXED_QUESTIONS)} fixed questions for VQA.")

    print("\n--- Loading Models ---")
    rrg_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_classes, class_labels = get_class_info(args.csv_file_path)

    print("Loading Seg, Class, and RRG models...")
    seg_model = UnifiedMultiTaskModel(1, 1, 1, 1, args).to(device)
    class_model = UnifiedMultiTaskModel(1, num_classes, 1, 1, args).to(device)
    rrg_model = UnifiedMultiTaskModel(1, num_classes, rrg_tokenizer.vocab_size, rrg_tokenizer.vocab_size, args).to(device)

    try:
        seg_model.load_state_dict(torch.load(args.seg_checkpoint_path, map_location=device))
        class_model.load_state_dict(torch.load(args.class_checkpoint_path, map_location=device))
        rrg_model.load_state_dict(torch.load(args.rrg_checkpoint_path, map_location=device))
    except FileNotFoundError as e:
        print(f"❌ FATAL ERROR: Checkpoint not found. Please ensure all model checkpoints exist. Error: {e}")
        return

    LLM_NAME = "Qwen/Qwen1.5-1.8B-Chat"
    print(f"Loading LLM for VQA: {LLM_NAME}...")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME, torch_dtype="auto", device_map=device, trust_remote_code=True
        ).eval()
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load the LLM '{LLM_NAME}'. Check model name and internet connection. Error: {e}")
        return

    try:
        with open(args.examples_path, "r", encoding="utf-8") as f:
            icl_examples = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ FATAL ERROR: Could not load or parse ICL examples from '{args.examples_path}'. Error: {e}")
        return
    print("✅ All models and data loaded successfully.")

    mask_output_dir = os.path.join(args.output_path, "masks")
    csv_output_path = os.path.join(args.output_path, "full_pipeline_predictions.csv")
    os.makedirs(args.output_path, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(args.image_folder, "*.png")) + \
                glob.glob(os.path.join(args.image_folder, "*.jpg")) + \
                glob.glob(os.path.join(args.image_folder, "*.jpeg")))
    
    if not image_paths:
        print(f"❌ No images found in '{args.image_folder}'. Please check the path.")
        return
    print(f"\nFound {len(image_paths)} images to process.")

    all_results = []
    for img_path in tqdm(image_paths, desc="Full Pipeline Evaluation"):
        result = run_full_pipeline_for_single_image(
            image_path=img_path,
            seg_model=seg_model,
            class_model=class_model,
            rrg_model=rrg_model,
            llm_model=llm_model,
            llm_tokenizer=llm_tokenizer,
            rrg_tokenizer=rrg_tokenizer,
            class_labels=class_labels,
            fixed_questions=FIXED_QUESTIONS,
            icl_examples=icl_examples,
            device=device,
            img_size=(args.img_size, args.img_size),
            mask_output_dir=mask_output_dir
        )
        if result:
            all_results.append(result)

    if not all_results:
        print("No results were generated. Exiting.")
        return
        
    fieldnames = ["image", "mask_path", "predicted_class", "confidence", "enhanced_report"] + FIXED_QUESTIONS
    
    print(f"\nSaving {len(all_results)} results to CSV...")
    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n✅ All predictions saved to: {csv_output_path}")

def run_3_task_pipeline_for_single_image(
    image_path,
    seg_model, class_model, rrg_model,
    rrg_tokenizer, class_labels,
    device, img_size,
    mask_output_dir=None
):
    print("-" * 50)
    print(f"Processing image: {os.path.basename(image_path)}")
    seg_model.eval()
    class_model.eval()
    rrg_model.eval()
    
    try:
        image = Image.open(image_path).convert('RGB').resize(img_size)
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Error loading image {image_path}: {e}")
        return None

    result = {"image": os.path.basename(image_path)}
    
    with torch.no_grad():
        print("1. Running Segmentation...")
        seg_output, _ = seg_model(image_tensor, mode='seg')
        binary_mask = (torch.sigmoid(seg_output).squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        location_info = analyze_segmentation_mask(binary_mask)
        
        if mask_output_dir:
            os.makedirs(mask_output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            mask_path = os.path.join(mask_output_dir, f"{base}_mask.png")
            cv2.imwrite(mask_path, binary_mask)
            result["mask_path"] = mask_path
        else:
            result["mask_path"] = "Not saved"

        print("2. Running Classification...")
        _, logits = class_model(image_tensor, mode='class')
        probs = F.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(logits, dim=1).item()
        predicted_class_label = class_labels[predicted_class_idx]
        result["predicted_class"] = predicted_class_label
        result["confidence"] = probs.max().item()

        print("3. Generating Initial Report...")
        initial_report = predict_single_image_rrg(
            rrg_model, image_path, device, img_size, rrg_tokenizer, 
            max_length=100, class_label_idx=predicted_class_idx
        )
        if not initial_report: initial_report = "Report could not be generated."
        
        print("4. Enhancing Report with Findings...")
        enhanced_report = (
            f"{initial_report} "
            f"Diagnostic analysis suggests the presence of {predicted_class_label}."
            f"{f' Based on the segmentation mask, {location_info}.' if predicted_class_label != 'Normal' else ''}"
        )
        result["enhanced_report"] = enhanced_report

    return result

def run_3_task_prediction_mode(args, device):
    print(f"🚀 Starting 3-Task Prediction Mode on folder: {args.image_folder}")

    print("\n--- Loading Models ---")
    rrg_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_classes, class_labels = get_class_info(args.csv_file_path) 

    print("Loading Seg, Class, and RRG models...")
    seg_model = UnifiedMultiTaskModel(1, 1, 1, 1, args).to(device)
    class_model = UnifiedMultiTaskModel(1, num_classes, 1, 1, args).to(device)
    rrg_model = UnifiedMultiTaskModel(1, num_classes, rrg_tokenizer.vocab_size, rrg_tokenizer.vocab_size, args).to(device)

    try:
        seg_model.load_state_dict(torch.load(args.seg_checkpoint_path, map_location=device))
        class_model.load_state_dict(torch.load(args.class_checkpoint_path, map_location=device))
        rrg_model.load_state_dict(torch.load(args.rrg_checkpoint_path, map_location=device))
    except FileNotFoundError as e:
        print(f"❌ FATAL ERROR: Checkpoint not found. Please ensure all model checkpoints exist. Error: {e}")
        return
    
    print("✅ Seg, Class, and RRG models loaded successfully.")

    mask_output_dir = os.path.join(args.output_path, "masks_3task")
    csv_output_path = os.path.join(args.output_path, "3_task_predictions.csv")
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True) 

    image_paths = sorted(glob.glob(os.path.join(args.image_folder, "*.png")) + \
                        glob.glob(os.path.join(args.image_folder, "*.jpg")) + \
                        glob.glob(os.path.join(args.image_folder, "*.jpeg")))
    
    if not image_paths:
        print(f"❌ No images found in '{args.image_folder}'. Please check the path.")
        return
    print(f"\nFound {len(image_paths)} images to process.")

    all_results = []
    for img_path in tqdm(image_paths, desc="3-Task Pipeline Evaluation"):
        result = run_3_task_pipeline_for_single_image(
            image_path=img_path,
            seg_model=seg_model,
            class_model=class_model,
            rrg_model=rrg_model,
            rrg_tokenizer=rrg_tokenizer,
            class_labels=class_labels,
            device=device,
            img_size=(args.img_size, args.img_size),
            mask_output_dir=mask_output_dir
        )
        if result:
            all_results.append(result)

    if not all_results:
        print("No results were generated. Exiting.")
        return
        
    fieldnames = ["image", "mask_path", "predicted_class", "confidence", "enhanced_report"]
    
    print(f"\nSaving {len(all_results)} results to CSV...")
    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n✅ All 3-task predictions saved to: {csv_output_path}")
