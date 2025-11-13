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
