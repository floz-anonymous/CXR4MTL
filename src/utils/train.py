import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from transformers import BertTokenizer

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from losses import CombinedLoss, iou_score, dice_score, RRGMetrics
from data_utils import get_class_info
from ..models import UnifiedMultiTaskModel


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