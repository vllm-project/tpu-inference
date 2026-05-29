import os
import sys
import json
import torch
import argparse
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
from qwen_vl_utils import process_vision_info

# Re-use components from test_qwen
from test_qwen import load_model_and_components, process_model_input

class VisualPerturber:
    @staticmethod
    def blank_image(img):
        return Image.new("RGB", img.size, color=(128, 128, 128))
    
    @staticmethod
    def gaussian_blur(img, radius=15):
        return img.filter(ImageFilter.GaussianBlur(radius))
    
    @staticmethod
    def shuffle_patches(img, grid_size=(4, 4)):
        w, h = img.size
        pw, ph = w // grid_size[0], h // grid_size[1]
        patches = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                box = (i * pw, j * ph, (i + 1) * pw, (j + 1) * ph)
                patches.append(img.crop(box))
        np.random.shuffle(patches)
        shuffled_img = Image.new("RGB", (w, h))
        idx = 0
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                shuffled_img.paste(patches[idx], (i * pw, j * ph))
                idx += 1
        return shuffled_img

def generate_response(model, tokenizer_or_processor, model_inputs):
    NUM_BEAMS, TEMPERATURE, MAX_NEW_TOKENS, USE_CACHE = 1, 0.0, 512, True
    DO_SAMPLE = True if TEMPERATURE > 0 else False

    device = next(model.parameters()).device

    assistant_prompt, image_paths, images, text = model_inputs['assistant_prompt'], model_inputs['img_paths'], model_inputs['images'], model_inputs['question']

    with torch.no_grad():
        # CRITICAL BUG FIX: Pass PIL Image objects directly instead of original paths
        image_content = [{"type": "image", "image": img} for img in images]

        messages = [
            {"role": "assistant", "content": assistant_prompt},
            {"role": "user", "content": image_content + [{"type": "text", "text": text}]}
        ]

        input_text = tokenizer_or_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = tokenizer_or_processor(text=[input_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

        output_ids = model.generate(**inputs, num_beams=NUM_BEAMS, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS, use_cache=USE_CACHE, do_sample=DO_SAMPLE)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        response = tokenizer_or_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    torch.cuda.empty_cache()

    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_idx', type=int, required=True, help="0 to 7")
    parser.add_argument('--num_shards', type=int, default=8)
    parser.add_argument('--mode', type=str, required=True, choices=['standard', 'blind', 'shuffled', 'blurred', 'option_shuffled', 'attention_ablated'])
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument('--model_name', type=str, default='qwen2_5vl-7b')
    parser.add_argument('--dataset_path', type=str, default="/drive/SpatialScore/SpatialScore_benchmark/SpatialScore_benchmark.ndjson")
    parser.add_argument('--output_dir', type=str, default="/drive/SpatialScore/eval_results")
    args = parser.parse_args()

    # Load model on current GPU (pinned by CUDA_VISIBLE_DEVICES)
    model, processor = load_model_and_components(args.model_path, args.model_name)

    # Load full dataset
    with open(args.dataset_path, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Calculate shard slice
    total_items = len(data)
    shard_size = int(np.ceil(total_items / args.num_shards))
    start_idx = args.shard_idx * shard_size
    end_idx = min(start_idx + shard_size, total_items)
    shard_data = data[start_idx:end_idx]

    print(f"[Shard {args.shard_idx}] Processing items {start_idx} to {end_idx} (Total: {len(shard_data)})")

    # Configure target output directory
    mode_output_dir = os.path.join(args.output_dir, args.model_name, args.mode)
    os.makedirs(mode_output_dir, exist_ok=True)

    shard_results = []
    for idx, item in enumerate(tqdm(shard_data, desc=f"Shard {args.shard_idx}")):
        # Handle multi-choice option formatting and shuffling
        if item.get('question_type') == 'multi-choice' and 'options' in item:
            options = item.get('options', [])
            
            if args.mode == 'option_shuffled':
                # Shuffle options and adapt ground truth answer key
                gt_answer_text = options[ord(item['answer'].strip('()')) - 65] if 'answer' in item else ''
                shuffled_indices = np.random.permutation(len(options))
                options = [options[i] for i in shuffled_indices]
                new_gt_index = list(shuffled_indices).index(options.index(gt_answer_text)) if gt_answer_text in options else 0
                
                item['options'] = options
                item['answer'] = f"({chr(65 + new_gt_index)})"
            
            # ALWAYS format and append the options to the question text for multi-choice!
            question = item.get('question', '')
            option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
            formatted_options = []
            for i, option in enumerate(options):
                if i < len(option_letters):
                    formatted_options.append(f"({option_letters[i]}) {option}")
            
            if formatted_options:
                item['question'] = f"{question}\n" + "\n".join(formatted_options)

        # Process standard visual inputs
        model_inputs = process_model_input(item)

        # Apply visual perturbations if needed
        if args.mode == 'blind':
            model_inputs['images'] = [VisualPerturber.blank_image(img) for img in model_inputs['images']]
        elif args.mode == 'shuffled':
            model_inputs['images'] = [VisualPerturber.shuffle_patches(img) for img in model_inputs['images']]
        elif args.mode == 'blurred':
            model_inputs['images'] = [VisualPerturber.gaussian_blur(img, radius=15) for img in model_inputs['images']]

        # Generate model response
        response = generate_response(model, processor, model_inputs)

        result_entry = {
            "id": item.get('id', start_idx + idx),
            "source_dataset": item.get('source_dataset', 'unknown'),
            "category": item.get('category', 'unknown'),
            "task": item.get('task', 'unknown'),
            "sub_task": item.get('sub_task', 'unknown'),
            "question_type": item.get('question_type', 'unknown'),
            "question": item.get('question', ''),
            "options": item.get('options', []),
            "gt_answer": item.get('answer', ''),
            "pred_answer": response,
        }
        shard_results.append(result_entry)

    # Save shard outputs
    shard_file = os.path.join(mode_output_dir, f"shard_{args.shard_idx}.json")
    with open(shard_file, 'w') as f:
        json.dump(shard_results, f, indent=2)
    print(f"[Shard {args.shard_idx}] Completed! Results saved to {shard_file}")

if __name__ == '__main__':
    main()
