import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk

from collections import Counter, defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class RRGMetrics:
    """
    Class to compute BLEU, ROUGE, METEOR, and CIDEr scores for radiology report generation.
    """
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
    
    def clean_text(self, text, tokenizer):
        """Clean text by removing special tokens and extra whitespace"""
        for special_token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token, 
                              tokenizer.unk_token, '[CLS]', '[SEP]', '[PAD]', '[UNK]']:
            if special_token:
                text = text.replace(special_token, '')
        
        text = ' '.join(text.lower().split())
        return text
        
    def compute_bleu(self, references, predictions, tokenizer=None):
        """
        Compute BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
        """
        bleu_scores = {'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': []}
        
        for ref, pred in zip(references, predictions):
            if tokenizer:
                ref = self.clean_text(ref, tokenizer)
                pred = self.clean_text(pred, tokenizer)
            
            if not ref.strip() or not pred.strip():
                continue
                
            ref_tokens = [ref.split()]
            pred_tokens = pred.split()
            
            try:
                bleu1 = sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoothing_function)
                bleu2 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing_function)
                bleu3 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoothing_function)
                bleu4 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing_function)
                
                bleu_scores['bleu1'].append(bleu1)
                bleu_scores['bleu2'].append(bleu2)
                bleu_scores['bleu3'].append(bleu3)
                bleu_scores['bleu4'].append(bleu4)
            except:
                continue
        
        return {k: np.mean(v) if v else 0.0 for k, v in bleu_scores.items()}
    
    def compute_rouge(self, references, predictions, tokenizer=None):
        """
        Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
        """
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for ref, pred in zip(references, predictions):
            if tokenizer:
                ref = self.clean_text(ref, tokenizer)
                pred = self.clean_text(pred, tokenizer)
            
            if not ref.strip() or not pred.strip():
                continue
                
            try:
                scores = self.rouge_scorer.score(ref, pred)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            except:
                continue
        
        return {k: np.mean(v) if v else 0.0 for k, v in rouge_scores.items()}
    
    def compute_meteor(self, references, predictions, tokenizer=None):
        """
        Compute METEOR scores
        """
        meteor_scores = []
        
        for ref, pred in zip(references, predictions):
            if tokenizer:
                ref = self.clean_text(ref, tokenizer)
                pred = self.clean_text(pred, tokenizer)
            
            if not ref.strip() or not pred.strip():
                continue
                
            try:
                ref_tokens = ref.split()
                pred_tokens = pred.split()
                
                score = meteor_score([ref_tokens], pred_tokens)
                meteor_scores.append(score)
            except:
                continue
        
        return np.mean(meteor_scores) if meteor_scores else 0.0
    
    def get_ngrams(self, text, n):
        """Extract n-grams from text"""
        tokens = text.split()
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def compute_document_frequency(self, references):
        """Compute document frequency for n-grams across all references"""
        df_counts = defaultdict(int)
        
        for ref in references:
            if isinstance(ref, str):
                ref_text = ref
            else:
                ref_text = ' '.join(ref) if isinstance(ref, list) else str(ref)
            
            unique_ngrams = set()
            for n in range(1, 5):
                ngrams = self.get_ngrams(ref_text.lower(), n)
                unique_ngrams.update(ngrams)
            
            for ngram in unique_ngrams:
                df_counts[ngram] += 1
        
        return df_counts
    
    def compute_cider_score(self, reference, prediction, df_counts, total_docs, sigma=6.0):
        """Compute CIDEr score for a single prediction-reference pair"""
        if not reference.strip() or not prediction.strip():
            return 0.0
        
        ref_text = reference.lower()
        pred_text = prediction.lower()
        
        cider_scores = []
        
        for n in range(1, 5):
            ref_ngrams = self.get_ngrams(ref_text, n)
            pred_ngrams = self.get_ngrams(pred_text, n)
            
            if not ref_ngrams or not pred_ngrams:
                cider_scores.append(0.0)
                continue
            
            ref_counts = Counter(ref_ngrams)
            pred_counts = Counter(pred_ngrams)
            
            ref_len = len(ref_ngrams)
            pred_len = len(pred_ngrams)
            
            ref_tfidf = {}
            pred_tfidf = {}
            
            for ngram, count in ref_counts.items():
                tf = count / ref_len 
                df = df_counts.get(ngram, 0)
                idf = math.log(max(1.0, total_docs / (df + 1.0))) 
                ref_tfidf[ngram] = tf * idf
            
            for ngram, count in pred_counts.items():
                tf = count / pred_len 
                df = df_counts.get(ngram, 0)
                idf = math.log(max(1.0, total_docs / (df + 1.0))) 
                pred_tfidf[ngram] = tf * idf
            
            ref_norm = math.sqrt(sum(v ** 2 for v in ref_tfidf.values()))
            pred_norm = math.sqrt(sum(v ** 2 for v in pred_tfidf.values()))
            
            if ref_norm == 0 or pred_norm == 0:
                similarity = 0.0
            else:
                dot_product = sum(
                    ref_tfidf.get(ngram, 0) * pred_tfidf.get(ngram, 0)
                    for ngram in set(ref_tfidf.keys()) | set(pred_tfidf.keys())
                )
                similarity = dot_product / (ref_norm * pred_norm)
            
            len_diff = abs(len(ref_text.split()) - len(pred_text.split()))
            penalty = math.exp(-(len_diff ** 2) / (2 * sigma ** 2))
            
            cider_scores.append(similarity * penalty)
        
        return 10.0 * np.mean(cider_scores)
    
    def compute_cider(self, references, predictions, tokenizer=None):
        """
        Compute CIDEr scores
        """
        if not references or not predictions:
            return 0.0
        
        cleaned_refs = []
        cleaned_preds = []
        
        for ref, pred in zip(references, predictions):
            if tokenizer:
                ref = self.clean_text(ref, tokenizer)
                pred = self.clean_text(pred, tokenizer)
            else:
                ref = ' '.join(ref.lower().strip().split())
                pred = ' '.join(pred.lower().strip().split())
            
            if ref.strip() and pred.strip():
                cleaned_refs.append(ref)
                cleaned_preds.append(pred)
        
        if not cleaned_refs:
            print("Warning: No valid reference-prediction pairs found for CIDEr computation")
            return 0.0
        
        df_counts = self.compute_document_frequency(cleaned_refs)
        total_docs = len(cleaned_refs)
        
        if not df_counts:
            print("Warning: No n-grams found in references for CIDEr computation")
            return 0.0
        
        cider_scores = []
        for ref, pred in zip(cleaned_refs, cleaned_preds):
            try:
                score = self.compute_cider_score(ref, pred, df_counts, total_docs, sigma=6.0)
                if not math.isnan(score) and not math.isinf(score):
                    cider_scores.append(score)
            except Exception as e:
                print(f"Warning: Error computing CIDEr for pair: {e}")
                continue
        
        if not cider_scores:
            print("Warning: No valid CIDEr scores computed")
            return 0.0
        
        return np.mean(cider_scores)
    
    def compute_all_metrics(self, references, predictions, tokenizer=None):
        """
        Compute all metrics (BLEU, ROUGE, METEOR, CIDEr)
        """
        results = {}
        
        bleu_scores = self.compute_bleu(references, predictions, tokenizer)
        results.update(bleu_scores)
        
        rouge_scores = self.compute_rouge(references, predictions, tokenizer)
        results.update(rouge_scores)
        
        meteor_score = self.compute_meteor(references, predictions, tokenizer)
        results['meteor'] = meteor_score
        
        cider_score = self.compute_cider(references, predictions, tokenizer)
        results['cider'] = cider_score
        
        return results

def iou_score(y_true, y_pred, threshold=0.5):
    """Calculates Intersection over Union (IoU) score."""
    y_pred = (y_pred > threshold).float()
    y_true = (y_true > threshold).float()
    intersection = torch.logical_and(y_true, y_pred).sum()
    union = torch.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.item()

def dice_score(y_true, y_pred, threshold=0.5):
    """Calculates the Dice score for segmentation."""
    y_pred = (y_pred > threshold).float()
    y_true = (y_true > threshold).float()
    
    intersection = torch.sum(y_true * y_pred)
    
    dice = (2. * intersection + 1e-8) / (torch.sum(y_true) + torch.sum(y_pred) + 1e-8)
    return dice.item()

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation, which is particularly effective for highly
    imbalanced segmentation masks, such as the lungs in a chest X-ray.
    """
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    """
    A combined loss function for segmentation, blending BCE with Dice loss.
    This provides a good balance between pixel-level accuracy (BCE) and
    structural accuracy (Dice).
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        return self.dice_weight * dice + self.bce_weight * bce

class FocalLoss(nn.Module):
    """
    Focal Loss for classification, designed to address class imbalance.
    It down-weights the loss assigned to well-classified examples,
    focusing on hard, misclassified examples.
    """
    def __init__(self, focal_alpha=1, focal_gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.focal_alpha * (1-pt)**self.focal_gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss