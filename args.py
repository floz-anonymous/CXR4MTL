import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Unified Multi-Modal Framework for Chest X-ray Analysis")

    parser.add_argument('--mode', type=str, required=True,
                        choices=['train_seg', 'test_seg', 'pred_seg',
                                 'train_class', 'test_class', 'pred_class',
                                 'train_rrg', 'test_rrg', 'pred_rrg',
                                 'train_vqa', 'test_vqa', 'pred_vqa', 'pred_full', 'pred_with_qa', '3task_pred'],
                        help="Mode to run the model in: train/test/predict for segmentation, classification, RRG, QA, or a full prediction pipeline.")

    parser.add_argument('--csv_file_path', type=str, default='/home/cvsingh007/Neerupam/Model/final_2.xlsx',
                        help="Path to the main CSV file containing image IDs and reports for RRG.")
    parser.add_argument('--image_folder_path', type=str, default='/home/cvsingh007/Neerupam/Original_Images',
                        help="Base directory containing the X-ray images for RRG.")
    parser.add_argument('--seg_image_dir', type=str, default='/home/cvsingh007/Neerupam/Original_Images',
                        help="Directory containing the images for segmentation.")
    parser.add_argument('--seg_mask_dir', type=str, default='/home/cvsingh007/Neerupam/Training', 
                        help="Directory containing the masks for segmentation.")
    parser.add_argument('--reports_path', type=str, default='/home/cvsingh007/Neerupam/Model/QA_maira2_based.xlsx',
                        help="Path to the CSV/Excel file containing the reports for RRG.")
    parser.add_argument('--csv_file_path_class', type=str, default='/home/cvsingh007/Neerupam/Model/addwithpleff.xlsx', 
                        help="Path to the CSV/Excel file for classification task.")
    parser.add_argument('--image_folder_path_class', type=str, default='/home/cvsingh007/Neerupam/balanced_dataset', 
                        help="Base directory containing images for the classification task.")

    parser.add_argument('--img_size', type=int, default=256,
                        help="The desired size for input images (square).")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility.")

    parser.add_argument('--epochs', type=int, default=25,
                        help="Number of epochs for training.")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for training and evaluation.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="Weight decay for the optimizer.")
    parser.add_argument('--patience', type=int, default=5,
                        help="Patience for ReduceLROnPlateau scheduler and early stopping.")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout rate for the models.")

    parser.add_argument('--dice_weight', type=float, default=0.7,
                        help="Weight for Dice loss in segmentation.")
    parser.add_argument('--bce_weight', type=float, default=0.3,
                        help="Weight for BCE loss in segmentation.")
    parser.add_argument('--class_weight_gamma', type=float, default=2.0,
                        help="Gamma for Focal Loss in classification.")

    parser.add_argument('--vocab_size', type=int, default=30522,
                        help="Vocabulary size for the RRG model (BERT uncased).")
    parser.add_argument('--embed_dim', type=int, default=768,
                        help="Embedding dimension for the RRG decoder.")
    parser.add_argument('--num_layers', type=int, default=4,
                        help="Number of layers in the RRG decoder.")
    parser.add_argument('--max_len', type=int, default=100,
                        help="Maximum length of the generated report.")
    parser.add_argument('--use_labels_for_rrg', action='store_false',
                        help="If set, use ground truth classification labels to condition the report generation model during testing.")

    parser.add_argument('--examples_path', type=str, default='/home/cvsingh007/Neerupam/Model/examples_5shot.json',
                        help="Examples for in-context learning.")
    parser.add_argument('--csv_file_path_vqa', type=str, default='/home/cvsingh007/Neerupam/Model/QA_maira2_based.xlsx',
                        help="Path to the VQA ground truth CSV file for evaluation.")

    parser.add_argument('--seg_checkpoint_path', type=str, default='checkpoints/seg_final.pth',
                        help="Path to save/load the best segmentation model checkpoint.")
    
    parser.add_argument('--class_checkpoint_path', type=str, default='checkpoints/class_final.pth',
                        help="Path to save/load the best classification model checkpoint.")
    
    parser.add_argument('--rrg_checkpoint_path', type=str, default='checkpoints/rrg_final.pth',
                        help="Path to save/load the best RRG model checkpoint.")
    
    parser.add_argument('--seg_encoder_checkpoint_path', type=str, default='checkpoints/seg_encoder_checkpoint.pth',
                        help="Path to save/load the encoder checkpoint after segmentation training.")
    parser.add_argument('--class_encoder_checkpoint_path', type=str, default='checkpoints/class_encoder_checkpoint.pth',
                        help="Path to save/load the encoder checkpoint after classification training.")

    parser.add_argument('--image_to_predict_path', type=str, default='/home/cvsingh007/Neerupam/Model/1111.jpg',
                        help="Path to a single image for individual prediction modes.")
    parser.add_argument('--image_folder', type=str, default='/home/cvsingh007/Neerupam/Model/Test_Report',
                        help="Path to a folder of images for the full prediction pipeline.")
    parser.add_argument('--output_path', type=str, default='/home/cvsingh007/Neerupam/Model/res',
                        help="Path to the folder where predictions (masks, csv) will be saved.")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print("Arguments parsed successfully:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
