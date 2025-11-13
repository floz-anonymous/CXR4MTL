import torch
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ..models import UnifiedMultiTaskModel
from data_utils import get_class_info
from transformers import BertTokenizer, AutoTokenizer, BertModel, AutoModelForCausalLM
import pandas as pd
import numpy as np
import json
from losses import RRGMetrics
from tqdm import tqdm
import os
from PIL import Image
from prediction import predict_single_image_rrg, analyze_segmentation_mask


def build_icl_prompt_multi_q(examples, report, questions):
    prompt = (
        "You are an expert radiology assistant. Your task is to analyze the given REPORT "
        "and provide a concise answer for each question in the numbered list. "
        "Provide ONLY the answers, with each answer on a new line.\n\n"
    )
    for i, ex in enumerate(examples):
        prompt += f"--- EXAMPLE {i+1} ---\n"
        prompt += f"REPORT:\n{ex['report']}\n\n"
        prompt += "QUESTIONS & ANSWERS:\n"
        qa_block = ""
        for num, qa in enumerate(ex["qas"], 1):
            qa_block += f"{num}. {qa['q']}\n{qa['a']}\n"
        prompt += qa_block + "\n"
    prompt += f"--- TASK ---\n"
    prompt += f"REPORT:\n{report}\n\n"
    prompt += "QUESTIONS & ANSWERS:\n"
    for i, q in enumerate(questions, 1):
        prompt += f"{i}. {q}\n"
    return prompt

def generate_text_icl(model, tokenizer, prompt, device, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    input_ids_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = outputs[0][input_ids_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

def parse_multi_answers(generated_text, num_qs, original_questions):
    question_set = {q.strip().lower() for q in original_questions}
    lines = [line.strip() for line in generated_text.strip().split('\n') if line.strip()]
    answers = []
    for line in lines:
        if re.match(r"^---", line): continue
        cleaned_line = re.sub(r'^\d+[\.\)]?\s*', '', line).strip()
        if cleaned_line.lower() in question_set: continue
        cleaned_line = re.sub(r'^(answer|solution|response)\s*[:\-]?\s*', '', cleaned_line, flags=re.IGNORECASE).strip()
        if cleaned_line:
            answers.append(cleaned_line)
    final_answers = answers[:num_qs]
    while len(final_answers) < num_qs:
        final_answers.append("")
    return final_answers

def get_vqa_answers_from_llm(llm_model, llm_tokenizer, report_text, questions, icl_examples, device):
    if not report_text:
        print("⚠️ Warning: Report text is empty. Skipping VQA.")
        return [""] * len(questions)
        
    prompt = build_icl_prompt_multi_q(icl_examples, report_text, questions)
    output_text = generate_text_icl(llm_model, llm_tokenizer, prompt, device)
    predicted_answers = parse_multi_answers(output_text, len(questions), questions)
    
    return predicted_answers

def get_text_answer_from_deepseek(model, processor, report_text, questions):
    prompt = (
        "You are a radiology assistant. Based on the following report, answer each question briefly.\n\n"
        f"REPORT:\n{report_text}\n\n"
    )
    for i, q in enumerate(questions, 1):
        prompt += f"{i}. {q}\n"
    
    inputs = processor.tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.language_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    return processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_vqa_icl_evaluation(args, device):
    print(f"🚀 Starting VQA ICL Evaluation Mode on file: {args.csv_file_path_vqa}")

    TEST_IMAGE_IDS = [
        3815, 4515, 929, 1573, 1993, 2402, 4180, 5332, 1618, 3139, 5804, 3221, 5759, 4642, 478, 5655, 4043, 2223, 4675, 4034, 911, 2483,
        3358, 4906, 621, 5602, 6020, 5723, 2076, 2899, 97, 5942, 5552, 651, 4079, 5485, 956, 830, 291, 5229, 5205, 2493, 5117, 2457,
        2419, 724, 4572, 1306, 3252, 927, 4329, 2394, 1702, 2191, 3014, 5108, 2324, 694, 941, 2332, 5512, 859, 5647, 735, 2752, 4940,
        23, 3421, 378, 5030, 2227, 1264, 2360, 3664, 1195, 2896, 5629, 1495, 2408, 5433, 1849, 4659, 62, 3543, 2216, 4757, 5218, 1926,
        895, 6021, 5053, 3020, 4332, 3487, 1018, 1352, 4494, 563, 4678, 5305, 444, 4559, 3595, 2125, 3887, 484, 1252, 797, 1628, 1612,
        668, 5038, 809, 2327, 998, 1121, 5751, 2459, 4626, 169, 3461, 5439, 3300, 2115, 5840, 4289, 467, 3788, 516, 867, 3397, 1999,
        4453, 53, 2083, 4552, 2248, 3625, 5922, 1291, 3760, 4169, 4734, 3507, 4749, 2280, 872, 4987, 3341, 168, 5126, 326, 3449, 1813,
        663, 4344, 900, 491, 4990, 4879, 1587, 537, 1637, 5752, 814, 489, 1716, 26, 4162, 47, 942, 4013, 3195, 4041, 3218, 4450, 3627,
        6019, 5100, 3508, 5556, 5172, 1057, 3102, 3128, 1224, 2800, 4446, 356, 5097, 1416, 2645, 5109, 3463, 191, 2426, 635, 2695, 226,
        237, 6030, 4384, 5202, 2792, 995, 4287, 1508, 1742, 4833, 2286, 1877, 4475, 2044, 4830, 3728, 2082, 4664, 2307, 3492, 1157,
        2705, 1392, 3861, 2634, 4679, 5607, 4789, 2167, 5444, 4621, 5505, 1415, 158, 5096, 617, 5525, 3512, 240, 3902, 1867, 2690,
        221, 3708, 406, 5344, 5447, 1865, 2274, 4565, 450, 1258, 2904, 3391, 5182, 3824, 4667, 552, 1131, 1954, 526, 2995, 127, 950,
        5941, 5537, 4714, 5123, 1165, 1150, 3619, 2830, 1583, 4052, 4875, 2421, 793, 2005, 4474, 4367, 6013, 4347, 257, 4761, 3938,
        4256, 964, 5892, 2600, 1619, 4896, 4073, 1200, 2846, 2170, 4639, 140, 261, 3872, 2226, 2022, 1626, 5741, 838, 15, 2060, 1070,
        2368, 4969, 1697, 3737, 4560, 4571, 3753, 4596, 481, 1855, 1569, 3215, 332, 3624, 5526, 3481, 2244, 3890, 2622, 153, 3548, 2306,
        4775, 2879, 5706, 2858, 389, 5770, 1655, 2538, 3658, 5304, 3314, 3324, 2501, 1678, 316, 1952, 5909, 606, 3075, 5985, 4121,
        5974, 2303, 657, 1900, 4143, 4966, 28, 2497, 557, 973, 1796, 1769, 946, 123, 2592, 2607, 3430, 138, 3584, 292, 1938, 1120, 5355,
        778, 2438, 368, 4483, 3035, 301, 4878, 4031, 1068, 1535, 4827, 2373, 1842, 5378, 2909, 4136, 5226, 288, 1924, 3007, 235, 4350,
        4549, 3621, 2481, 835, 238, 3439, 3284, 2940, 5712, 4485, 936, 5021, 5844, 507, 5445, 3799, 2572, 1530, 4802, 4197, 5020, 3374,
        1343, 2161, 2176, 1767, 1358, 2963, 494, 5887, 5478, 3897, 2821, 352, 5290, 5699, 597, 5535, 2243, 3470, 836, 5385, 5249, 2386,
        3189, 1543, 5658, 1391, 4975, 2444, 2923, 5786, 2105, 3360, 5829, 2537, 3093, 5547, 4751, 3928, 1459, 5140, 2827, 358, 5705, 767,
        3216, 4039, 3054, 3877, 1408, 5523, 585, 2732, 1490, 2699, 3025, 5075, 4252, 142, 965, 3434, 439, 91, 1925, 3602, 1956, 2229,
        4547, 584, 4699, 2230, 776, 4886, 1594, 1832, 4301, 4091, 5737, 2043, 4605, 3782, 696, 3988, 1088, 1911, 3557, 250, 6011, 41,
        2510, 869, 5807, 1692, 3560, 2737, 4388, 1096, 1919, 1335, 464, 3756, 50, 4165, 4357, 5002, 89, 912, 4602, 2956, 1261, 2977,
        2154, 1589, 362, 1164, 5795, 3326, 463, 3725, 5371, 4144, 2876, 4522, 1833, 3479, 3525, 5994, 357, 5036, 4213, 2168, 5156, 218,
        1864, 883, 193, 5, 3297, 671, 2317, 4216, 1602, 5668, 1151, 4313, 1869, 223, 197, 1020, 4189, 5436, 4613, 2144, 2204, 3850, 2123,
        2351, 4433, 3806, 2160, 6010, 5187, 688, 4745, 2550, 2345, 4414, 3281, 5745, 4099, 39, 4154, 5800, 2277, 1454, 1485, 769, 5707,
        1763, 4314, 2668, 2447, 4258, 5081, 2625
    ]

    FIXED_QUESTIONS = [
        "Which lung is affected by this abnormality?",
        "What is the extent of infected region compared to total lung area?",
        "What is the severity of disease based on the % of lung involvement?",
        "What abnormalities are detected in this image?",
        "What disease or condition is identified in this image?",
        "What are the key findings from report?",
        "What additional observation about lung field, heart size or pleural space?",
        "What is final diagnostic decision?"
    ]
    print(f"Using {len(FIXED_QUESTIONS)} fixed questions for VQA evaluation.")

    print("\n--- Loading Models ---")
    rrg_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_classes, class_labels = get_class_info(args.csv_file_path)

    print("Loading Segmentation, Classification, and RRG models...")
    seg_model = UnifiedMultiTaskModel(1, 1, 1, 1, args).to(device)
    class_model = UnifiedMultiTaskModel(1, num_classes, 1, 1, args).to(device)
    rrg_model = UnifiedMultiTaskModel(1, num_classes, rrg_tokenizer.vocab_size, rrg_tokenizer.vocab_size, args).to(device)

    try:
        seg_model.load_state_dict(torch.load(args.seg_checkpoint_path, map_location=device))
        class_model.load_state_dict(torch.load(args.class_checkpoint_path, map_location=device))
        rrg_model.load_state_dict(torch.load(args.rrg_checkpoint_path, map_location=device))
    except FileNotFoundError as e:
        print(f"❌ FATAL ERROR: A required checkpoint was not found. Please ensure all model checkpoints exist. Error: {e}")
        return

    seg_model.eval()
    class_model.eval()
    rrg_model.eval()
    
    LLM_NAME = "Qwen/Qwen1.5-1.8B-Chat" 
    print(f"Loading LLM for VQA: {LLM_NAME}...")
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
        llm_model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME, torch_dtype="auto", device_map=device, trust_remote_code=True
        ).eval()
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load the LLM '{LLM_NAME}'. Check model name and internet connection. Error: {e}")
        return
    
    try:
        with open(args.examples_path, "r", encoding="utf-8") as f:
            icl_examples = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ FATAL ERROR: Could not load or parse ICL examples from '{args.examples_path}'. Error: {e}")
        return
    print("✅ All models and data loaded successfully.")

    try:
        full_qa_df = pd.read_excel(args.csv_file_path_vqa) if args.csv_file_path_vqa.endswith(('.xlsx', '.xls')) else pd.read_csv(args.csv_file_path_vqa)
        qa_df = full_qa_df[full_qa_df['IMG_ID'].isin(TEST_IMAGE_IDS)].reset_index(drop=True)

    except FileNotFoundError:
        print(f"❌ FATAL ERROR: VQA ground truth file not found at '{args.csv_file_path_vqa}'")
        return
    
    if len(qa_df) == 0:
        print("❌ FATAL ERROR: No matching IMG_IDs found in the QA file. Please check your test ID list.")
        return
        
    print(f"Found {len(qa_df)} matching records for VQA evaluation from the test list.")

    all_predicted_qa_strings = []
    all_ground_truth_qa_strings = []
    metrics_calculator = RRGMetrics()

    all_images_avg_scores = {}

    for index, row in tqdm(qa_df.iterrows(), total=len(qa_df), desc="VQA Evaluation"):
        img_id = str(row['IMG_ID']).zfill(4)
        
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = os.path.join(args.image_folder_path, f"{img_id}{ext}")
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if not image_path:
            print(f"⚠️ Warning: Image for ID {img_id} not found. Skipping.")
            continue

        enhanced_report = ""
        try:
            image = Image.open(image_path).convert('RGB').resize((args.img_size, args.img_size))
            image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                seg_output, _ = seg_model(image_tensor, mode='seg')
                binary_mask = (torch.sigmoid(seg_output).squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
                location_info = analyze_segmentation_mask(binary_mask)
        
                _, logits = class_model(image_tensor, mode='class')
                probs = F.softmax(logits, dim=1)
                predicted_class_idx = torch.argmax(logits, dim=1).item()
                predicted_class_label = class_labels[predicted_class_idx]
        
                initial_report = predict_single_image_rrg(
                    rrg_model, image_path, device, (args.img_size, args.img_size), rrg_tokenizer, 
                    max_length=100, class_label_idx=predicted_class_idx
                )
                if not initial_report: initial_report = "Report could not be generated."
                
                enhanced_report = (
                    f"{initial_report} "
                    f"Diagnostic analysis suggests the presence of {predicted_class_label}."
                    f"{f' Based on the segmentation mask, {location_info}.' if predicted_class_label != 'Normal' else ''}"
                )
        
                predicted_answers = get_vqa_answers_from_llm(
                    llm_model, llm_tokenizer, enhanced_report, FIXED_QUESTIONS, icl_examples, device
                )
        except Exception as e:
            print(f"An error occurred while processing image {img_id}: {e}")
            continue


        ground_truth_answers = [str(row.get(q, "")) for q in FIXED_QUESTIONS]

        current_image_scores = {}
        for question, pred_ans, gt_ans in zip(FIXED_QUESTIONS, predicted_answers, ground_truth_answers):
            pred_qa_string = f"{question.strip()} {str(pred_ans).strip()}"
            gt_qa_string = f"{question.strip()} {str(gt_ans).strip()}"

            pair_scores = metrics_calculator.compute_all_metrics(
                references=[gt_qa_string], 
                predictions=[pred_qa_string],
                tokenizer=rrg_tokenizer
            )
            
            for metric, score in pair_scores.items():
                if metric not in current_image_scores:
                    current_image_scores[metric] = []
                current_image_scores[metric].append(score)

        if current_image_scores:
            avg_scores_for_image = {
                metric: np.mean(scores) for metric, scores in current_image_scores.items()
            }
            
            for metric, avg_score in avg_scores_for_image.items():
                if metric not in all_images_avg_scores:
                    all_images_avg_scores[metric] = []
                all_images_avg_scores[metric].append(avg_score)
    
    print("\n--- Final VQA Evaluation Scores ---")
    if not all_images_avg_scores:
        print("No results were generated. Cannot calculate metrics.")
        return

    final_scores = {
        metric: np.mean(scores) for metric, scores in all_images_avg_scores.items()
    }

    num_samples = len(list(all_images_avg_scores.values())[0])
    print(f"Scores are averaged over {num_samples} samples.\n")
    for metric, score in final_scores.items():
        print(f"{metric.upper():<10}: {score:.4f}")
