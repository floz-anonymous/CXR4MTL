import os
import cv2
import torch
import numpy as np
from PIL import Image

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
