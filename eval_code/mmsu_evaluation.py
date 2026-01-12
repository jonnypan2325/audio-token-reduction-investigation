import logging

logging.getLogger("root").setLevel(logging.ERROR)
logging.getLogger("qwen_omni_utils").setLevel(logging.ERROR)
logging.getLogger("qwen").setLevel(logging.ERROR)

import json
import argparse
from collections import defaultdict

import torch
from typing import List
from tqdm import tqdm
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

def load_model_and_processor(model_path: str) -> tuple:
    """Load Qwen3 Omni model and processor."""
    if "Qwen2.5" in model_path:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
        return model, processor

    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    return model, processor


def inference(
    model,
    processor,
    audio_path: str,
    prompt: str,
) -> str:
    """Run inference on a single audio sample."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(
        **inputs,
        use_audio_in_video=True,
        return_audio=False,
        thinker_max_new_tokens=256,
        thinker_do_sample=False,
        max_new_tokens=512,
        do_sample=False,
    )

    text_output = processor.batch_decode(
        output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    generated_text = text_output[0].split("assistant\n")[-1].strip()
    print(generated_text)
    return generated_text


def inference_batch(
    model,
    processor,
    audio_paths: List[str],
    prompts: List[str],
) -> List[str]:
    """Run inference on a batch of audio samples. Returns list of generated_text."""
    assert len(audio_paths) == len(prompts)

    messages_list = []
    for audio_path, prompt in zip(audio_paths, prompts):
        messages_list.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]

    audios_list, images_list, videos_list = [], [], []
    for m in messages_list:
        audios, images, videos = process_mm_info(m, use_audio_in_video=True)
        audios_list.append(audios[0] if audios else None)
        images_list.append(images[0] if images else None)
        videos_list.append(videos[0] if videos else None)

    inputs = processor(
        text=texts,
        audio=audios_list,
        images=images_list if any(x is not None for x in images_list) else None,
        videos=videos_list if any(x is not None for x in videos_list) else None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            use_audio_in_video=True,
            return_audio=False,
            thinker_max_new_tokens=256,
            thinker_do_sample=False,
            max_new_tokens=512,
            do_sample=False,
        )

    text_output = processor.batch_decode(
        output[0] if isinstance(output, (list, tuple)) else output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    # print("RAW_DECODED_TAIL:", repr(text_output[0][-80:]))
    generated_texts = [t.split("assistant\n")[-1].strip() for t in text_output]
    return generated_texts

def format_mmsu_prompt(record: dict) -> str:
    """
    MMSU format: question + choices.
    It does not change the scoring logic (it takes A/B/C/D from response),
    so the model is prompted to output option letters as much as possible (but also supports "The answer is A." type).
    """
    q = record.get("question", "")
    a = record.get("choice_a", "")
    b = record.get("choice_b", "")
    c = record.get("choice_c", "")
    d = record.get("choice_d", "")
    prompt = (
        f"{q}\n"
        f"A. {a}\n"
        f"B. {b}\n"
        f"C. {c}\n"
        f"D. {d}\n"
        "Select one option from the provided choices."
    )
    return prompt


def get_audio_path_from_record(record: dict) -> str:
    """
    Adapt to various possible audio path fields in the record.
    """
    for k in ["audio", "audio_path", "audio_id", "wav", "path"]:
        if k in record and record[k]:
            return record[k]
    raise KeyError("No audio path field found in record. Expected one of: audio/audio_path/audio_id/wav/path")


def run_mmsu_inference_and_write_jsonl(
    model_path: str,
    input_jsonl: str,
    output_jsonl: str,
    batch_size: int = 1,
) -> None:
    print(f"Loading model: {model_path}")
    model, processor = load_model_and_processor(model_path)

    records = load_jsonl_data(input_jsonl)
    print(f"Total MMSU samples: {len(records)}")
    print(f"Running with batch_size = {batch_size}")

    batch_audio, batch_prompts, batch_indices = [], [], []

    def flush_batch():
        nonlocal batch_audio, batch_prompts, batch_indices
        if not batch_indices:
            return
        try:
            batch_responses = inference_batch(model, processor, batch_audio, batch_prompts) if batch_size > 1 else [
                inference(model, processor, batch_audio[0], batch_prompts[0])
            ]
        except Exception as e:
            print(f"Batch error: {e}")
            batch_responses = [str(e)] * len(batch_indices)

        for idx, resp in zip(batch_indices, batch_responses):
            records[idx]["response"] = resp

        batch_audio, batch_prompts, batch_indices = [], [], []

    for i, record in enumerate(tqdm(records, desc="Infer MMSU")):
        if i % 100 == 0 and i > 0:
            print('*'*20, f"Inferred {i} / {len(records)} samples", '*'*20)
        try:
            audio_path = get_audio_path_from_record(record)
        except Exception as e:
            # Write `missing audio path` to the response. It will be considered as format error/failure in evaluation.
            records[i]["response"] = "None"
            print(f"[{i}] missing audio path: {e}")
            continue

        prompt = format_mmsu_prompt(record)

        if batch_size <= 1:
            # 单条：直接推理，写回 response
            try:
                resp = inference(model, processor, audio_path, prompt)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                resp = "None"
            records[i]["response"] = resp
        else:
            batch_audio.append(audio_path)
            batch_prompts.append(prompt)
            batch_indices.append(i)
            if len(batch_indices) == batch_size:
                flush_batch()


    # flush last batch
    if batch_size > 1:
        flush_batch()

    # Write response to output jsonl
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved model responses to: {output_jsonl}")

def load_jsonl_data(jsonl_path):
    """Load data from the provided JSONL file."""
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {line.strip()}")
                print(e)
    return records

def calculate_accuracy_per_task_and_category(data):
    """Calculate accuracy for each unique task and category."""
    task_category_accuracy = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    task_average_accuracy = defaultdict(lambda: {"total_correct": 0, "total_count": 0})
    
    # Initialize counters
    total_correct = 0
    total_count = 0
    fail_num = 0 

    for record in data:
        task = record.get('category', '')
        category = record.get('sub-category', '')

        # Extract response
        response = record.get('response', '')
        # print(f'Response: {response}')
        try:
            predict = response.strip().replace('\n', '')
        except:
            print('Error prediction!')
            continue
        
        model_predict = None
        
        if predict != 'None' and predict:
            if predict[0] == 'A' or predict[0] == 'B' or predict[0] == 'C' or predict[0] == 'D':
                model_predict = predict[0]
            #This situation may occur when the answer given by gpt is "The answer is A."
            elif len(predict) > 1:
                if predict[-2] == 'A' or predict[-2] == 'B' or predict[-2] == 'C' or predict[-2] == 'D':
                    model_predict = predict[-2]
                else:
                    print(f'Wrong format response: {predict}')
                    continue
            else:
                print(f'Wrong format response: {predict}')
                continue
        
        # Get the correct answer
        answer_gt = record.get('answer_gt', '')
        choices = {
            'A': record.get('choice_a', ''),
            'B': record.get('choice_b', ''),
            'C': record.get('choice_c', ''),
            'D': record.get('choice_d', '')
        }
        
        # Check if the prediction matches the correct answer
        if model_predict:
            if model_predict == 'A' and choices['A'] == answer_gt:
                task_category_accuracy[task][category]["correct"] += 1
                total_correct += 1
            elif model_predict == 'B' and choices['B'] == answer_gt:
                task_category_accuracy[task][category]["correct"] += 1
                total_correct += 1
            elif model_predict == 'C' and choices['C'] == answer_gt:
                task_category_accuracy[task][category]["correct"] += 1
                total_correct += 1
            elif model_predict == 'D' and choices['D'] == answer_gt:
                task_category_accuracy[task][category]["correct"] += 1
                total_correct += 1
                
        # Increase the total count for the task and category
        task_category_accuracy[task][category]["total"] += 1
        total_count += 1

    # Calculate accuracy per task and category
    for task, categories in task_category_accuracy.items():
        total_correct_for_task = 0
        total_count_for_task = 0
        for category, counts in categories.items():
            total = counts["total"]
            correct = counts["correct"]
            accuracy = correct / total if total > 0 else 0
            task_category_accuracy[task][category] = accuracy
            
            # Calculate overall task accuracy
            total_correct_for_task += correct
            total_count_for_task += total
        
        # Calculate average accuracy for each task
        task_average_accuracy[task]["total_correct"] = total_correct_for_task
        task_average_accuracy[task]["total_count"] = total_count_for_task
        task_average_accuracy[task]["average_accuracy"] = total_correct_for_task / total_count_for_task if total_count_for_task > 0 else 0

    # Calculate overall accuracy
    overall_accuracy = total_correct / total_count if total_count > 0 else 0

    return task_category_accuracy, task_average_accuracy, overall_accuracy, total_count

def main():
    """Main function to load data, calculate accuracy, and print results."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process a JSONL file and calculate accuracy.")
    parser.add_argument('jsonl_path', type=str, help="Path to the input JSONL file")
    args = parser.parse_args()

    # Load data
    data = load_jsonl_data(args.jsonl_path)

    # Calculate accuracy
    task_category_accuracies, task_average_accuracies, overall_accuracy, total_count = calculate_accuracy_per_task_and_category(data)

    # Print accuracy for each category and sub-category
    for task, categories in task_category_accuracies.items():
        for category, accuracy in categories.items():
            print(f'Category: {task}, Sub-category: {category}, Accuracy: {accuracy:.4f}')

    # Print average accuracy for each category
    for task, accuracy_info in task_average_accuracies.items():
        average_accuracy = accuracy_info["average_accuracy"]
        print(f'Category: {task}, Average Accuracy: {average_accuracy:.4f}')

    # Print overall accuracy
    print(f'Overall Accuracy: {overall_accuracy:.4f}')
    print(f'Total count: {total_count}')

if __name__ == "__main__":
    import os
    print("CWD =", os.getcwd())


    # ======== Firstly, run inference and generate responses.jsonl. Then call `main` from the template to calculate accuracy. ========
    # python mmsu_inference_and_eval.py \
    #   --model_path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    #   --input_jsonl /path/to/mmsu.jsonl \
    #   --output_jsonl /path/to/mmsu_with_response.jsonl \
    #   --batch_size 8
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct") # "Qwen/Qwen2.5-Omni-7B"
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    
    print("Args:", args)

    run_mmsu_inference_and_write_jsonl(
        model_path=args.model_path,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        batch_size=args.batch_size,
    )

    data = load_jsonl_data(args.output_jsonl)
    task_category_accuracies, task_average_accuracies, overall_accuracy, total_count = calculate_accuracy_per_task_and_category(data)

    for task, categories in task_category_accuracies.items():
        for category, accuracy in categories.items():
            print(f'Category: {task}, Sub-category: {category}, Accuracy: {accuracy:.4f}')

    for task, accuracy_info in task_average_accuracies.items():
        average_accuracy = accuracy_info["average_accuracy"]
        print(f'Category: {task}, Average Accuracy: {average_accuracy:.4f}')

    print(f'Overall Accuracy: {overall_accuracy:.4f}')
    print(f'Total count: {total_count}')