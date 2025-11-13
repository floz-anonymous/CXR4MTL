from ..models import UnifiedMultiTaskModel
from ..utils.data_utils import preprocess_and_load_data_seg, LungSegmentationDataset
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from ..utils.losses import CombinedLoss
from ..utils.train import train_model
from ..utils.test import test_model
from ..utils.prediction import predict_single_image_seg

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
