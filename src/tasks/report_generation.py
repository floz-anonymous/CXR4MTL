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
