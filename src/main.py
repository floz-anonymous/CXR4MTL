import torch
import os
from args import get_args
from utils.common import set_seed
from tasks.segmentation import run_seg_mode
from tasks.classification import run_class_mode
from tasks.report_generation import run_rrg_mode
from tasks.answer_extraction import run_vqa_mode

def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoints", exist_ok=True)

    modes = {
        "seg": run_seg_mode,
        "class": run_class_mode,
        "rrg": run_rrg_mode,
        "vqa": run_vqa_mode
    }

    for prefix, fn in modes.items():
        if args.mode.startswith(prefix):
            fn(args, device)
            return

    print(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()
