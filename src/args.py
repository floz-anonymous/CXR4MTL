import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Unified Multi-Task Chest X-ray Analysis Framework")
    parser.add_argument('--mode', type=str, required=True,
                        choices=[
                            'train_seg', 'test_seg', 'pred_seg',
                            'train_class', 'test_class', 'pred_class',
                            'train_rrg', 'test_rrg', 'pred_rrg',
                            'train_vqa', 'test_vqa', 'pred_vqa',
                            'pred_full', 'pred_with_qa', '3task_pred'
                        ],
                        help="Execution mode.")
    parser.add_argument('--seg_image_dir', type=str, default='data/segmentation/images',
                        help="Directory containing segmentation images.")
    parser.add_argument('--seg_mask_dir', type=str, default='data/segmentation/masks',
                        help="Directory containing segmentation masks.")
    parser.add_argument('--seg_checkpoint_path', type=str, default='checkpoints/segmentation_best.pth',
                        help="Path to save/load segmentation checkpoint.")
    parser.add_argument('--seg_encoder_checkpoint_path', type=str, default='checkpoints/seg_encoder.pth',
                        help="Path for saving segmentation encoder.")
    parser.add_argument('--csv_file_path_class', type=str, default='data/classification/labels.xlsx',
                        help="Classification labels file.")
    parser.add_argument('--image_folder_path_class', type=str, default='data/classification/images',
                        help="Directory of classification images.")
    parser.add_argument('--class_checkpoint_path', type=str, default='checkpoints/classification_best.pth',
                        help="Path to save/load classification model.")
    parser.add_argument('--class_encoder_checkpoint_path', type=str, default='checkpoints/class_encoder.pth',
                        help="Path for saving classification encoder.")
    parser.add_argument('--csv_file_path', type=str, default='data/rrg/reports.xlsx',
                        help="Radiology report dataset path for RRG.")
    parser.add_argument('--image_folder_path', type=str, default='data/rrg/images',
                        help="Image directory for RRG task.")
    parser.add_argument('--rrg_checkpoint_path', type=str, default='checkpoints/rrg_best.pth',
                        help="Path to save/load RRG checkpoint.")
    parser.add_argument('--csv_file_path_vqa', type=str, default='data/vqa/vqa.xlsx',
                        help="Ground truth file for VQA.")
    parser.add_argument('--examples_path', type=str, default='data/vqa/examples_20shot.json',
                        help="Few-shot examples file for VQA.")
    parser.add_argument('--image_to_predict_path', type=str, default='sample.jpg',
                        help="Image for prediction modes.")
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dice_weight', type=float, default=0.7)
    parser.add_argument('--bce_weight', type=float, default=0.3)
    parser.add_argument('--class_weight_gamma', type=float, default=2.0)
    parser.add_argument('--vocab_size', type=int, default=30522)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--use_labels_for_rrg', action='store_false')
    return parser
