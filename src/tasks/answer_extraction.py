from ..models import UnifiedMultiTaskModel
from ..utils.data_utils import preprocess_and_load_rrg_data, RRGDataset, get_class_info
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from ..utils.losses import CombinedLoss
from ..utils.train import train_model
from ..utils.test import test_model
from ..utils.prediction import predict_single_image_rrg, predict_single_image_class
from ..utils.vqa_utils import run_vqa_icl_evaluation

def run_vqa_mode(args, device):
    print(f"Running VQA ICL Evaluation Mode: {args.mode}")
    if 'test' in args.mode:
        run_vqa_icl_evaluation(args, device)
