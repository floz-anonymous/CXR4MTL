from ..models import UnifiedMultiTaskModel
from ..utils.data_utils import preprocess_and_load_data, LungClassificationDataset, get_class_info
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from ..utils.losses import CombinedLoss
from ..utils.train import train_model
from ..utils.test import test_model
from ..utils.prediction import predict_single_image_class

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
