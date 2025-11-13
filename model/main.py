import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import csv
import math
import json, re
from collections import defaultdict, Counter
import pandas as pd
from models import (
    DualEncoderUNet, 
    ClassifierHead, 
    RRGDecoder, 
    UnifiedMultiTaskModel
)
from args import get_args
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from torchmetrics.text import BLEUScore, ROUGEScore
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import sacrebleu
from nltk.translate.meteor_score import meteor_score
import nltk
from losses import (
    RRGMetrics,
    dice_score,
    iou_score,
    FocalLoss,
    CombinedLoss,
    DiceLoss
)
from load_data import (
    RRGDataset,
    preprocess_and_load_rrg_data,
    get_class_info,
    preprocess_and_load_data,
    preprocess_and_load_data,
    LungClassificationDataset,
    preprocess_and_load_data_seg,
    LungSegmentationDataset
)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, checkpoint_path, patience, mode='seg'):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    best_loss = float('inf')
    patience_counter = 0

    if mode == 'rrg':
        metrics_calculator = RRGMetrics()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        if mode == 'seg':
            train_metrics = {'iou': 0.0, 'dice': 0.0}
        elif mode == 'class':
            train_metrics = {'accuracy': 0.0}
        else: # rrg
            train_metrics = {'perplexity': 0.0}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            if mode == 'seg':
                images, masks = batch_data
                images, masks = images.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs, _ = model(images, mode='seg')
                
                if outputs.dim() == 4 and masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                
                outputs_sigmoid = torch.sigmoid(outputs)
                loss = criterion(outputs_sigmoid, masks)
                
                with torch.no_grad():
                    train_metrics['iou'] += iou_score(masks, outputs_sigmoid)
                    train_metrics['dice'] += dice_score(masks, outputs_sigmoid)
                
            elif mode == 'class':
                images, labels = batch_data
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                probs, logits = model(images, mode='class')
                loss = criterion(logits, labels)
                
                with torch.no_grad():
                    _, predicted = torch.max(logits, 1)
                    train_metrics['accuracy'] += (predicted == labels).sum().item() / labels.size(0)
                
            elif mode == 'rrg':
                images, reports, labels = batch_data
                images, reports, labels = images.to(device), reports.to(device), labels.to(device)
                
                input_reports = reports[:, :-1]
                target_reports = reports[:, 1:]
                
                optimizer.zero_grad()
                outputs = model(images, mode='rrg', rrg_targets=input_reports, class_labels=labels)
                
                outputs = outputs.reshape(-1, outputs.size(-1))
                target_reports = target_reports.reshape(-1)
                
                loss = criterion(outputs, target_reports)
                
                with torch.no_grad():
                    train_metrics['perplexity'] += torch.exp(loss).item()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})
        
        model.eval()
        val_loss = 0.0
        if mode == 'seg':
            val_metrics = {'iou': 0.0, 'dice': 0.0}
        elif mode == 'class':
            val_metrics = {'accuracy': 0.0}
        else:
            val_metrics = {'perplexity': 0.0}

        val_generated_reports = []
        val_reference_reports = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]")
            
            for batch_data in progress_bar:
                if mode == 'seg':
                    images, masks = batch_data
                    images, masks = images.to(device), masks.to(device)
                    
                    outputs, _ = model(images, mode='seg')
                    
                    if outputs.dim() == 4 and masks.dim() == 3:
                        masks = masks.unsqueeze(1)
                    
                    outputs_sigmoid = torch.sigmoid(outputs)
                    loss = criterion(outputs_sigmoid, masks)
                    
                    val_metrics['iou'] += iou_score(masks, outputs_sigmoid)
                    val_metrics['dice'] += dice_score(masks, outputs_sigmoid)
                    
                elif mode == 'class':
                    images, labels = batch_data
                    images, labels = images.to(device), labels.to(device)
                    
                    probs, logits = model(images, mode='class')
                    loss = criterion(logits, labels)
                    
                    _, predicted = torch.max(logits, 1)
                    val_metrics['accuracy'] += (predicted == labels).sum().item() / labels.size(0)
                    
                elif mode == 'rrg':
                    images, reports, labels = batch_data
                    images, reports, labels = images.to(device), reports.to(device), labels.to(device)

                    input_reports = reports[:, :-1]
                    target_reports = reports[:, 1:]
                    
                    outputs = model(images, mode='rrg', rrg_targets=input_reports, class_labels=labels)
                    outputs_flat = outputs.reshape(-1, outputs.size(-1))
                    target_reports_flat = target_reports.reshape(-1)
                    
                    loss = criterion(outputs_flat, target_reports_flat)
                    val_metrics['perplexity'] += torch.exp(loss).item()
                    
                    if len(val_generated_reports) < 50:  
                        _, features, _ = model.encoder(images)
                        pooled_features = model.global_pool(features)
                        flattened_features = model.flatten(pooled_features)
                        
                        generated_ids = model.rrg_decoder.generate_report(
                            flattened_features, labels, tokenizer, max_length=55
                        )
                        
                        for i in range(min(generated_ids.size(0), 50 - len(val_generated_reports))):
                            pred_text = tokenizer.decode(generated_ids[i].cpu().numpy(), skip_special_tokens=True)
                            target_text = tokenizer.decode(reports[i].cpu().numpy(), skip_special_tokens=True)
                            
                            val_generated_reports.append(pred_text)
                            val_reference_reports.append(target_text)
                
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
            val_metrics[key] /= len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if mode == 'seg':
            print(f"Train IoU: {train_metrics['iou']:.4f}, Train Dice: {train_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f}, Val Dice: {val_metrics['dice']:.4f}")
        elif mode == 'class':
            print(f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        elif mode == 'rrg':
            print(f"Train Perplexity: {train_metrics['perplexity']:.4f}, Val Perplexity: {val_metrics['perplexity']:.4f}")
            
            if val_generated_reports and val_reference_reports:
                metrics = metrics_calculator.compute_all_metrics(val_reference_reports, val_generated_reports, tokenizer)
                print(f"\nText Generation Metrics:")
                print(f"BLEU-1: {metrics['bleu1']:.4f}")
                print(f"BLEU-2: {metrics['bleu2']:.4f}")
                print(f"BLEU-3: {metrics['bleu3']:.4f}")
                print(f"BLEU-4: {metrics['bleu4']:.4f}")
                print(f"ROUGE-1: {metrics['rouge1']:.4f}")
                print(f"ROUGE-2: {metrics['rouge2']:.4f}")
                print(f"ROUGE-L: {metrics['rougeL']:.4f}")
                print(f"METEOR: {metrics['meteor']:.4f}")
                print(f"CIDEr: {metrics['cider']:.4f}")
        
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

def test_model(model, test_loader, device, mode='seg', class_labels=None, tokenizer=None, args=None):
    model.eval()
    all_predictions = []
    all_targets = []
    test_loss = 0.0

    generated_reports = []
    reference_reports = []

    if mode == 'rrg':
        metrics_calculator = RRGMetrics()
    
    criterion = {
        'seg': CombinedLoss(dice_weight=0.7, bce_weight=0.3),
        'class': nn.CrossEntropyLoss(),
        'rrg': nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer else 0)
    }[mode]
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        
        for batch_data in progress_bar:
            if mode == 'seg':
                images, masks = batch_data
                images, masks = images.to(device), masks.to(device)
                
                outputs, _ = model(images, mode='seg')
                loss = criterion(torch.sigmoid(outputs), masks)
                
                predictions = torch.sigmoid(outputs).cpu().numpy()
                all_predictions.extend(predictions)
                all_targets.extend(masks.cpu().numpy())
                
            elif mode == 'class':
                images, labels = batch_data
                images, labels = images.to(device), labels.to(device)
                
                probs, logits = model(images, mode='class')
                loss = criterion(logits, labels)
                
                _, predicted = torch.max(logits, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
            elif mode == 'rrg':
                images, reports, gt_labels = batch_data
                images, reports, gt_labels = images.to(device), reports.to(device), gt_labels.to(device)
                
                if args and not args.use_labels_for_rrg:
                    num_classes, _ = get_class_info(args.csv_file_path_class)
                    model_class = UnifiedMultiTaskModel(1, num_classes, 1, 1, args).to(device)
                    model_class.eval()
                    _, logits = model_class(images, mode='class')
                    conditioning_labels = torch.argmax(logits, 1)
                    if progress_bar.n == 0: 
                        print("Using PREDICTED labels for RRG testing.")
                else:
                    conditioning_labels = gt_labels
                    if progress_bar.n == 0: 
                        print("Using GROUND TRUTH labels for RRG testing.")
                
                input_reports = reports[:, :-1]
                target_reports = reports[:, 1:]
                
                outputs = model(images, mode='rrg', rrg_targets=input_reports, class_labels=conditioning_labels)
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = target_reports.reshape(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                
                _, features, _ = model.encoder(images)
                pooled_features = model.global_pool(features)
                flattened_features = model.flatten(pooled_features)
                
                generated_ids = model.rrg_decoder.generate_report(
                    flattened_features, conditioning_labels, tokenizer, max_length=55
                )
                
                for i in range(generated_ids.size(0)):
                    pred_text = tokenizer.decode(generated_ids[i].cpu().numpy(), skip_special_tokens=True)
                    target_text = tokenizer.decode(reports[i].cpu().numpy(), skip_special_tokens=True)
                    
                    generated_reports.append(pred_text)
                    reference_reports.append(target_text)
            
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    
    if mode == 'seg':
        preds_tensor = torch.from_numpy(np.array(all_predictions))
        targets_tensor = torch.from_numpy(np.array(all_targets))

        mean_iou = iou_score(targets_tensor, preds_tensor)
        mean_dice = dice_score(targets_tensor, preds_tensor)

        print(f"\nTest Results:\nTest Loss: {test_loss:.4f}\nMean IoU: {mean_iou:.4f}\nMean Dice: {mean_dice:.4f}")
        
    elif mode == 'class':
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        
        print(f"\nTest Results:\nTest Loss: {test_loss:.4f}\nAccuracy: {accuracy:.4f}\nF1 Score: {f1:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}")

        save_path = "/home/cvsingh007/Neerupam/Model/class_validation/results/OURS"
        os.makedirs(save_path, exist_ok=True)

        cm = confusion_matrix(all_targets, all_predictions)

        fig, ax = plt.subplots(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels if class_labels else None)
        disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
        plt.title("Confusion Matrix - Classification Results")
        plt.tight_layout()

        cm_path = os.path.join(save_path, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close(fig)

        print(f"\nConfusion matrix saved at: {cm_path}")
        
        if class_labels:
            from sklearn.metrics import classification_report
            print("\nClassification Report:\n", classification_report(all_targets, all_predictions, target_names=class_labels, zero_division=0))
            
    elif mode == 'rrg':
        perplexity = np.exp(test_loss)
        print(f"\nTest Results:\nTest Loss: {test_loss:.4f}\nPerplexity: {perplexity:.4f}")
        
        if generated_reports and reference_reports:
            print("\nComputing text generation metrics...")
            metrics = metrics_calculator.compute_all_metrics(reference_reports, generated_reports, tokenizer)
            
            print(f"\nText Generation Metrics:")
            print(f"BLEU-1: {metrics['bleu1']:.4f}")
            print(f"BLEU-2: {metrics['bleu2']:.4f}")
            print(f"BLEU-3: {metrics['bleu3']:.4f}")
            print(f"BLEU-4: {metrics['bleu4']:.4f}")
            print(f"ROUGE-1: {metrics['rouge1']:.4f}")
            print(f"ROUGE-2: {metrics['rouge2']:.4f}")
            print(f"ROUGE-L: {metrics['rougeL']:.4f}")
            print(f"METEOR: {metrics['meteor']:.4f}")
            print(f"CIDEr: {metrics['cider']:.4f}")


def analyze_segmentation_mask(mask, threshold=0.05):
    if mask is None or mask.ndim != 2:
        return "Invalid mask provided."

    mask = (mask > 0).astype(np.uint8)

    height, width = mask.shape
    midpoint = width // 2

    left_half = mask[:, :midpoint]
    right_half = mask[:, midpoint:]

    left_pixels = np.sum(left_half)
    right_pixels = np.sum(right_half)

    total = left_pixels + right_pixels
    if total == 0:
        return "No affected regions detected."

    left_ratio = left_pixels / total
    right_ratio = right_pixels / total

    if abs(left_ratio - right_ratio) > threshold:
        if left_ratio > right_ratio:
            is_left_affected, is_right_affected = 1, 0
        else:
            is_left_affected, is_right_affected = 0, 1
    else:
        is_left_affected = is_right_affected = 1

    if is_left_affected and is_right_affected:
        return "Both lungs appear to be affected."
    elif is_left_affected:
        return "The left lung appears to be affected."
    elif is_right_affected:
        return "The right lung appears to be affected."
    else:
        return "No specific region of affection was detected."

def predict_single_image_seg(model, image_path, device, img_size):
    model.eval()
    try:
        image = Image.open(image_path).convert('RGB').resize(img_size)
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    except Exception as e:
        return print(f"Error loading image: {e}")
    
    with torch.no_grad():
        output, _ = model(image_tensor, mode='seg')
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
    
    binary_mask = (prediction > 0.5).astype(np.uint8) * 255
    output_path = image_path.replace('.', '_segmentation.')
    cv2.imwrite(output_path, binary_mask)
    print(f"Segmentation saved to: {output_path}")

def predict_single_image_class(model, image_path, device, img_size, class_labels):
    model.eval()
    try:
        image = Image.open(image_path).convert('RGB').resize(img_size)
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    with torch.no_grad():
        probs, logits = model(image_tensor, mode='class')
        predicted_class = torch.argmax(logits, dim=1).item()
        confidence = torch.softmax(logits, dim=1).max().item()
    
    print(f"Predicted Class: {class_labels[predicted_class]}\nConfidence: {confidence:.4f}")
    return predicted_class

def clean_generated_text(text, tokenizer):
    special_tokens = [
        tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, 
        tokenizer.unk_token, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '<pad>', '<unk>'
    ]
    
    for token in special_tokens:
        if token:
            text = text.replace(token, '')
    
    text = ' '.join(text.split())
    
    if text:
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        if not text.endswith(('.', '!', '?')):
            text += '.'
    
    return text

def predict_single_image_rrg(model, image_path, device, img_size, tokenizer, max_length=100, class_label_idx=None):
    model.eval()
    try:
        image = Image.open(image_path).convert('RGB').resize(img_size)
        image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    with torch.no_grad():
        try:
            _, features, _ = model.encoder(image_tensor)
            pooled_features = model.global_pool(features)
            flattened_features = model.flatten(pooled_features)

            if class_label_idx is not None:
                label_tensor = torch.tensor([class_label_idx], dtype=torch.long).to(device)
            else:
                label_tensor = None
                
            generated_tokens = model.rrg_decoder.generate_report(
                flattened_features, label_tensor, tokenizer, max_length=max_length
            ).squeeze(0)
            
            report_text = tokenizer.decode(generated_tokens.cpu().numpy(), skip_special_tokens=True)
            report_text = clean_generated_text(report_text, tokenizer)
            
            return report_text
            
        except Exception as e:
            print(f"❌ Error during report generation: {e}")
            import traceback
            traceback.print_exc()
            return None


def run_seg_mode(args, device):
    print(f"Running Segmentation Mode: {args.mode}")
    model = UnifiedMultiTaskModel(1, 1, 1, 1, args).to(device)

    if 'train' in args.mode or 'test' in args.mode:
        images, masks = preprocess_and_load_data_seg(args.seg_image_dir, args.seg_mask_dir, (args.img_size, args.img_size))
        if images.size == 0:
            return print("No segmentation data. Exiting.")

        dataset = LungSegmentationDataset(images, masks)
        total = len(dataset)
        train_size = int(0.7 * total)
        val_size = int(0.1 * total)
        test_size = total - train_size - val_size

        train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        if 'train' in args.mode:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            criterion = CombinedLoss(args.dice_weight, args.bce_weight)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience)

            train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                        device, args.epochs, args.seg_checkpoint_path, args.patience, 'seg')

            print(f"Saving pre-trained encoder to {args.seg_encoder_checkpoint_path}")
            torch.save(model.encoder.state_dict(), args.seg_encoder_checkpoint_path)

        elif 'test' in args.mode:
            if not os.path.exists(args.seg_checkpoint_path):
                return print("No checkpoint found for testing.")

            model.load_state_dict(torch.load(args.seg_checkpoint_path, map_location=device))
            test_model(model, test_loader, device, 'seg')

    elif 'pred' in args.mode:
        if not os.path.exists(args.seg_checkpoint_path):
            return print("No checkpoint found for prediction.")

        model.load_state_dict(torch.load(args.seg_checkpoint_path, map_location=device))
        predict_single_image_seg(model, args.image_to_predict_path, device, (args.img_size, args.img_size))

def run_class_mode(args, device):
    print(f"Running Classification Mode: {args.mode}")
    num_classes, class_labels = get_class_info(args.csv_file_path_class)
    if num_classes == 0:
        return

    model = UnifiedMultiTaskModel(1, num_classes, 1, 1, args).to(device)

    if 'train' in args.mode or 'test' in args.mode:
        images, labels, _, _ = preprocess_and_load_data(
            args.csv_file_path_class, args.image_folder_path_class, (args.img_size, args.img_size)
        )
        if images.size == 0:
            return

        dataset = LungClassificationDataset(images, labels)

        total = len(dataset)
        train_size = int(0.7 * total)
        val_size = int(0.1 * total)
        test_size = total - train_size - val_size

        train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        if 'train' in args.mode:
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience)

            train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                        device, args.epochs, args.class_checkpoint_path, args.patience, 'class')

            print(f"Saving fine-tuned encoder to {args.class_encoder_checkpoint_path}")
            torch.save(model.encoder.state_dict(), args.class_encoder_checkpoint_path)

        elif 'test' in args.mode:
            if not os.path.exists(args.class_checkpoint_path):
                return print("No checkpoint found.")

            model.load_state_dict(torch.load(args.class_checkpoint_path, map_location=device))
            test_model(model, test_loader, device, 'class', class_labels)

    elif 'pred' in args.mode:
        if not os.path.exists(args.class_checkpoint_path):
            return print("No checkpoint found.")

        model.load_state_dict(torch.load(args.class_checkpoint_path, map_location=device))
        predict_single_image_class(model, args.image_to_predict_path, device,
                                (args.img_size, args.img_size), class_labels)

def run_rrg_mode(args, device):
    print(f"Running RRG Mode: {args.mode}")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    num_classes, _ = get_class_info(args.reports_path)
    if num_classes == 0:
        return

    model = UnifiedMultiTaskModel(1, num_classes,
                                tokenizer.vocab_size,
                                tokenizer.vocab_size, args).to(device)

    if 'train' in args.mode or 'test' in args.mode:
        images, reports, labels = preprocess_and_load_rrg_data(
            args.reports_path, args.image_folder_path,
            tokenizer, (args.img_size, args.img_size), args.max_len
        )
        if images is None:
            return

        dataset = RRGDataset(images, reports, labels)

        total = len(dataset)
        train_size = int(0.7 * total)
        val_size = int(0.1 * total)
        test_size = total - train_size - val_size

        train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)

        if 'train' in args.mode:
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
            )
            criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience)

            train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                        device, args.epochs, args.rrg_checkpoint_path, args.patience, 'rrg')

        elif 'test' in args.mode:
            if not os.path.exists(args.rrg_checkpoint_path):
                return print("No RRG checkpoint found.")

            model.load_state_dict(torch.load(args.rrg_checkpoint_path, map_location=device))
            test_model(model, test_loader, device, 'rrg', tokenizer=tokenizer, args=args)

    elif 'pred' in args.mode:
        if not os.path.exists(args.rrg_checkpoint_path) or not os.path.exists(args.class_checkpoint_path):
            return print("RRG or Classification checkpoint not found.")

        model.load_state_dict(torch.load(args.rrg_checkpoint_path, map_location=device))

        class_idx = predict_single_image_class(
            model, args.image_to_predict_path, device,
            (args.img_size, args.img_size),
            get_class_info(args.csv_file_path_class)[1]
        )

        if class_idx is not None:
            print(class_idx)
            report = predict_single_image_rrg(
                model, args.image_to_predict_path, device,
                (args.img_size, args.img_size),
                tokenizer, args.max_len, class_label_idx=class_idx
            )
            print(f"Generated Report: {report}")



def build_icl_prompt_multi_q(examples, report, questions):
    prompt = (
        "You are an expert radiology assistant. Your task is to analyze the given REPORT "
        "and provide a concise answer for each question in the numbered list. "
        "Provide ONLY the answers, with each answer on a new line.\n\n"
    )
    for i, ex in enumerate(examples):
        prompt += f"--- EXAMPLE {i+1} ---\n"
        prompt += f"REPORT:\n{ex['report']}\n\n"
        prompt += "QUESTIONS & ANSWERS:\n"
        qa_block = ""
        for num, qa in enumerate(ex["qas"], 1):
            qa_block += f"{num}. {qa['q']}\n{qa['a']}\n"
        prompt += qa_block + "\n"
    prompt += f"--- TASK ---\n"
    prompt += f"REPORT:\n{report}\n\n"
    prompt += "QUESTIONS & ANSWERS:\n"
    for i, q in enumerate(questions, 1):
        prompt += f"{i}. {q}\n"
    return prompt

def generate_text_icl(model, tokenizer, prompt, device, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    input_ids_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = outputs[0][input_ids_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

def parse_multi_answers(generated_text, num_qs, original_questions):
    question_set = {q.strip().lower() for q in original_questions}
    lines = [line.strip() for line in generated_text.strip().split('\n') if line.strip()]
    answers = []
    for line in lines:
        if re.match(r"^---", line): continue
        cleaned_line = re.sub(r'^\d+[\.\)]?\s*', '', line).strip()
        if cleaned_line.lower() in question_set: continue
        cleaned_line = re.sub(r'^(answer|solution|response)\s*[:\-]?\s*', '', cleaned_line, flags=re.IGNORECASE).strip()
        if cleaned_line:
            answers.append(cleaned_line)
    final_answers = answers[:num_qs]
    while len(final_answers) < num_qs:
        final_answers.append("")
    return final_answers

def get_vqa_answers_from_llm(llm_model, llm_tokenizer, report_text, questions, icl_examples, device):
    if not report_text:
        print("⚠️ Warning: Report text is empty. Skipping VQA.")
        return [""] * len(questions)
        
    prompt = build_icl_prompt_multi_q(icl_examples, report_text, questions)
    output_text = generate_text_icl(llm_model, llm_tokenizer, prompt, device)
    predicted_answers = parse_multi_answers(output_text, len(questions), questions)
    
    return predicted_answers

def get_text_answer_from_deepseek(model, processor, report_text, questions):
    prompt = (
        "You are a radiology assistant. Based on the following report, answer each question briefly.\n\n"
        f"REPORT:\n{report_text}\n\n"
    )
    for i, q in enumerate(questions, 1):
        prompt += f"{i}. {q}\n"
    
    inputs = processor.tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.language_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    return processor.tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_vqa_icl_evaluation(args, device):
    print(f"🚀 Starting VQA ICL Evaluation Mode on file: {args.csv_file_path_vqa}")

    TEST_IMAGE_IDS = [
        3815, 4515, 929, 1573, 1993, 2402, 4180, 5332, 1618, 3139, 5804, 3221, 5759, 4642, 478, 5655, 4043, 2223, 4675, 4034, 911, 2483,
        3358, 4906, 621, 5602, 6020, 5723, 2076, 2899, 97, 5942, 5552, 651, 4079, 5485, 956, 830, 291, 5229, 5205, 2493, 5117, 2457,
        2419, 724, 4572, 1306, 3252, 927, 4329, 2394, 1702, 2191, 3014, 5108, 2324, 694, 941, 2332, 5512, 859, 5647, 735, 2752, 4940,
        23, 3421, 378, 5030, 2227, 1264, 2360, 3664, 1195, 2896, 5629, 1495, 2408, 5433, 1849, 4659, 62, 3543, 2216, 4757, 5218, 1926,
        895, 6021, 5053, 3020, 4332, 3487, 1018, 1352, 4494, 563, 4678, 5305, 444, 4559, 3595, 2125, 3887, 484, 1252, 797, 1628, 1612,
        668, 5038, 809, 2327, 998, 1121, 5751, 2459, 4626, 169, 3461, 5439, 3300, 2115, 5840, 4289, 467, 3788, 516, 867, 3397, 1999,
        4453, 53, 2083, 4552, 2248, 3625, 5922, 1291, 3760, 4169, 4734, 3507, 4749, 2280, 872, 4987, 3341, 168, 5126, 326, 3449, 1813,
        663, 4344, 900, 491, 4990, 4879, 1587, 537, 1637, 5752, 814, 489, 1716, 26, 4162, 47, 942, 4013, 3195, 4041, 3218, 4450, 3627,
        6019, 5100, 3508, 5556, 5172, 1057, 3102, 3128, 1224, 2800, 4446, 356, 5097, 1416, 2645, 5109, 3463, 191, 2426, 635, 2695, 226,
        237, 6030, 4384, 5202, 2792, 995, 4287, 1508, 1742, 4833, 2286, 1877, 4475, 2044, 4830, 3728, 2082, 4664, 2307, 3492, 1157,
        2705, 1392, 3861, 2634, 4679, 5607, 4789, 2167, 5444, 4621, 5505, 1415, 158, 5096, 617, 5525, 3512, 240, 3902, 1867, 2690,
        221, 3708, 406, 5344, 5447, 1865, 2274, 4565, 450, 1258, 2904, 3391, 5182, 3824, 4667, 552, 1131, 1954, 526, 2995, 127, 950,
        5941, 5537, 4714, 5123, 1165, 1150, 3619, 2830, 1583, 4052, 4875, 2421, 793, 2005, 4474, 4367, 6013, 4347, 257, 4761, 3938,
        4256, 964, 5892, 2600, 1619, 4896, 4073, 1200, 2846, 2170, 4639, 140, 261, 3872, 2226, 2022, 1626, 5741, 838, 15, 2060, 1070,
        2368, 4969, 1697, 3737, 4560, 4571, 3753, 4596, 481, 1855, 1569, 3215, 332, 3624, 5526, 3481, 2244, 3890, 2622, 153, 3548, 2306,
        4775, 2879, 5706, 2858, 389, 5770, 1655, 2538, 3658, 5304, 3314, 3324, 2501, 1678, 316, 1952, 5909, 606, 3075, 5985, 4121,
        5974, 2303, 657, 1900, 4143, 4966, 28, 2497, 557, 973, 1796, 1769, 946, 123, 2592, 2607, 3430, 138, 3584, 292, 1938, 1120, 5355,
        778, 2438, 368, 4483, 3035, 301, 4878, 4031, 1068, 1535, 4827, 2373, 1842, 5378, 2909, 4136, 5226, 288, 1924, 3007, 235, 4350,
        4549, 3621, 2481, 835, 238, 3439, 3284, 2940, 5712, 4485, 936, 5021, 5844, 507, 5445, 3799, 2572, 1530, 4802, 4197, 5020, 3374,
        1343, 2161, 2176, 1767, 1358, 2963, 494, 5887, 5478, 3897, 2821, 352, 5290, 5699, 597, 5535, 2243, 3470, 836, 5385, 5249, 2386,
        3189, 1543, 5658, 1391, 4975, 2444, 2923, 5786, 2105, 3360, 5829, 2537, 3093, 5547, 4751, 3928, 1459, 5140, 2827, 358, 5705, 767,
        3216, 4039, 3054, 3877, 1408, 5523, 585, 2732, 1490, 2699, 3025, 5075, 4252, 142, 965, 3434, 439, 91, 1925, 3602, 1956, 2229,
        4547, 584, 4699, 2230, 776, 4886, 1594, 1832, 4301, 4091, 5737, 2043, 4605, 3782, 696, 3988, 1088, 1911, 3557, 250, 6011, 41,
        2510, 869, 5807, 1692, 3560, 2737, 4388, 1096, 1919, 1335, 464, 3756, 50, 4165, 4357, 5002, 89, 912, 4602, 2956, 1261, 2977,
        2154, 1589, 362, 1164, 5795, 3326, 463, 3725, 5371, 4144, 2876, 4522, 1833, 3479, 3525, 5994, 357, 5036, 4213, 2168, 5156, 218,
        1864, 883, 193, 5, 3297, 671, 2317, 4216, 1602, 5668, 1151, 4313, 1869, 223, 197, 1020, 4189, 5436, 4613, 2144, 2204, 3850, 2123,
        2351, 4433, 3806, 2160, 6010, 5187, 688, 4745, 2550, 2345, 4414, 3281, 5745, 4099, 39, 4154, 5800, 2277, 1454, 1485, 769, 5707,
        1763, 4314, 2668, 2447, 4258, 5081, 2625
    ]

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
    print(f"Using {len(FIXED_QUESTIONS)} fixed questions for VQA evaluation.")

    print("\n--- Loading Models ---")
    rrg_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_classes, class_labels = get_class_info(args.csv_file_path)

    print("Loading Segmentation, Classification, and RRG models...")
    seg_model = UnifiedMultiTaskModel(1, 1, 1, 1, args).to(device)
    class_model = UnifiedMultiTaskModel(1, num_classes, 1, 1, args).to(device)
    rrg_model = UnifiedMultiTaskModel(1, num_classes, rrg_tokenizer.vocab_size, rrg_tokenizer.vocab_size, args).to(device)

    try:
        seg_model.load_state_dict(torch.load(args.seg_checkpoint_path, map_location=device))
        class_model.load_state_dict(torch.load(args.class_checkpoint_path, map_location=device))
        rrg_model.load_state_dict(torch.load(args.rrg_checkpoint_path, map_location=device))
    except FileNotFoundError as e:
        print(f"❌ FATAL ERROR: A required checkpoint was not found. Please ensure all model checkpoints exist. Error: {e}")
        return

    seg_model.eval()
    class_model.eval()
    rrg_model.eval()
    
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

    try:
        full_qa_df = pd.read_excel(args.csv_file_path_vqa) if args.csv_file_path_vqa.endswith(('.xlsx', '.xls')) else pd.read_csv(args.csv_file_path_vqa)
        qa_df = full_qa_df[full_qa_df['IMG_ID'].isin(TEST_IMAGE_IDS)].reset_index(drop=True)

    except FileNotFoundError:
        print(f"❌ FATAL ERROR: VQA ground truth file not found at '{args.csv_file_path_vqa}'")
        return
    
    if len(qa_df) == 0:
        print("❌ FATAL ERROR: No matching IMG_IDs found in the QA file. Please check your test ID list.")
        return
        
    print(f"Found {len(qa_df)} matching records for VQA evaluation from the test list.")

    all_predicted_qa_strings = []
    all_ground_truth_qa_strings = []
    metrics_calculator = RRGMetrics()

    all_images_avg_scores = {}

    for index, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="VQA Evaluation"):
        img_id = str(row['IMG_ID']).zfill(4)
        
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(args.image_folder_path, f"{img_id}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            print(f"⚠️ Warning: Image for ID {img_id} not found. Skipping.")
            continue

        enhanced_report = ""
        try:
            image = Image.open(image_path).convert('RGB').resize((args.img_size, args.img_size))
            image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                seg_output, _ = seg_model(image_tensor, mode='seg')
                binary_mask = (torch.sigmoid(seg_output).squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
                location_info = analyze_segmentation_mask(binary_mask)
        
                _, logits = class_model(image_tensor, mode='class')
                probs = F.softmax(logits, dim=1)
                predicted_class_idx = torch.argmax(logits, dim=1).item()
                predicted_class_label = class_labels[predicted_class_idx]
        
                initial_report = predict_single_image_rrg(
                    rrg_model, image_path, device, (args.img_size, args.img_size), rrg_tokenizer, 
                    max_length=100, class_label_idx=predicted_class_idx
                )
                if not initial_report: initial_report = "Report could not be generated."
                
                enhanced_report = (
                    f"{initial_report} "
                    f"Diagnostic analysis suggests the presence of {predicted_class_label}."
                    f"{f' Based on the segmentation mask, {location_info}.' if predicted_class_label != 'Normal' else ''}"
                )
        
                predicted_answers = get_vqa_answers_from_llm(
                    llm_model, llm_tokenizer, enhanced_report, FIXED_QUESTIONS, icl_examples, device
                )
        except Exception as e:
            print(f"An error occurred while processing image {img_id}: {e}")
            continue


        ground_truth_answers = [str(row.get(q, "")) for q in FIXED_QUESTIONS]

        current_image_scores = {}
        for question, pred_ans, gt_ans in zip(FIXED_QUESTIONS, predicted_answers, ground_truth_answers):
            pred_qa_string = f"{question.strip()} {str(pred_ans).strip()}"
            gt_qa_string = f"{question.strip()} {str(gt_ans).strip()}"

            pair_scores = metrics_calculator.compute_all_metrics(
                references=[gt_qa_string], 
                predictions=[pred_qa_string],
                tokenizer=rrg_tokenizer
            )
            
            for metric, score in pair_scores.items():
                if metric not in current_image_scores:
                    current_image_scores[metric] = []
                current_image_scores[metric].append(score)

        if current_image_scores:
            avg_scores_for_image = {
                metric: np.mean(scores) for metric, scores in current_image_scores.items()
            }
            
            for metric, avg_score in avg_scores_for_image.items():
                if metric not in all_images_avg_scores:
                    all_images_avg_scores[metric] = []
                all_images_avg_scores[metric].append(avg_score)
    
    print("\n--- Final VQA Evaluation Scores ---")
    if not all_images_avg_scores:
        print("No results were generated. Cannot calculate metrics.")
        return

    final_scores = {
        metric: np.mean(scores) for metric, scores in all_images_avg_scores.items()
    }

    num_samples = len(list(all_images_avg_scores.values())[0])
    print(f"Scores are averaged over {num_samples} samples.\n")
    for metric, score in final_scores.items():
        print(f"{metric.upper():<10}: {score:.4f}")

def run_vqa_mode(args, device):
    print(f"Running VQA ICL Evaluation Mode: {args.mode}")
    if 'test' in args.mode:
        run_vqa_icl_evaluation(args, device)


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

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs('checkpoints', exist_ok=True)
    
    mode_map = {
        'seg': run_seg_mode, 'class': run_class_mode, 'rrg': run_rrg_mode,
        'vqa': run_vqa_mode, 'full': run_predict_full_mode,
        '3task': run_3_task_prediction_mode
    }
    
    run_function = None
    for prefix, func in mode_map.items():
        if args.mode.startswith(prefix) or args.mode.endswith(prefix):
            run_function = func
            break
            
    if run_function:
        run_function(args, device)
    else:
        print(f"Error: Mode '{args.mode}' is not valid.")

if __name__ == "__main__":
    main()
