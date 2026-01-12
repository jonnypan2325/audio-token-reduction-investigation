import json
import torch
from typing import Dict, List, Any
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import argparse
from pathlib import Path
from tqdm import tqdm
import re

def load_model_and_processor(model_path: str) -> tuple:
    """Load Qwen3 Omni model and processor."""
    if "Qwen/Qwen2.5-Omni-7B" in model_path:
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
    # import pdb; pdb.set_trace()
    text_output = processor.batch_decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # Extract only the generated part (remove the input prompt)
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

    # Build messages for each sample
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

    # Apply chat template per sample (batch text)
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]

    # process_mm_info per sample (can't easily vectorize because it returns per-message mm blobs)
    audios_list, images_list, videos_list = [], [], []
    for m in messages_list:
        audios, images, videos = process_mm_info(m, use_audio_in_video=True)
        audios_list.append(audios[0] if audios else None)
        images_list.append(images[0] if images else None)
        videos_list.append(videos[0] if videos else None)

    # Processor supports batched text/audio
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
            thinker_max_new_tokens=256,  # ✅ 保持和你原 inference 一致
            thinker_do_sample=False,
            max_new_tokens=512,          # ✅ 保持一致
            do_sample=False,             # ✅ 保持一致
        )

    text_output = processor.batch_decode(
        output[0] if isinstance(output, (list, tuple)) else output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    generated_texts = [t.split("assistant\n")[-1].strip() for t in text_output]
    # 你原 inference 会 print，这里也保持“行为一致”可选：
    # for gt in generated_texts: print(gt)

    return generated_texts


def format_multiple_choice_prompt(question: str, choices: List[str]) -> str:
    """Format the question and choices for the model."""
    prompt = f"{question}Select one option from the provided choices.{choices}"
    return prompt


def string_match(answer, prediction, choices):
    # Function to normalize and tokenize text
    def tokenize(text):
        # Convert to lowercase and find all word tokens
        return set(re.findall(r"\b\w+\b", text.lower()))

    # Tokenize prediction and answer
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)

    if not prediction_tokens:
        return False

    # Tokenize incorrect choices and exclude tokens present in the answer
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)

    # Condition 1: All tokens of the answer are in the prediction
    cond1 = answer_tokens.issubset(prediction_tokens)

    # Condition 2: Prediction does not contain any tokens from incorrect choices (excluding shared words)
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)

    return cond1 and cond2

def evaluate_mmau_subset(
    model_path: str,
    data_file: str,
    output_file: str,
    batch_size: int = 1,
) -> None:
    """Evaluate the model on MMAU test-mini subset."""
    print(f"Running with batch_size = {batch_size}")

    
    # Load model and processor
    print("Loading model and processor...")
    model, processor = load_model_and_processor(model_path)
    
    # Load test data
    print(f"Loading test data from {data_file}...")
    with open(data_file, 'r') as f:
        test_data = json.load(f)
    
    print(f"Total samples: {len(test_data)}")
    
    # Process each sample
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    # for sample in tqdm(test_data, desc="Evaluating samples"):
    #     sample_id = sample['id']
    #     audio_path = sample['audio_id']
    #     question = sample['question']
    #     choices = sample['choices']
    #     ground_truth = sample['answer']
        
    #     # Format the prompt
    #     prompt = format_multiple_choice_prompt(question, choices)

    #     # audio_path=str(ROOT_DIR / audio_path)
        
    #     try:
    #         # Get model prediction
    #         model_response = inference(model, processor, audio_path, prompt)
    #         model_prediction = model_response
            
    #         # Check if prediction is correct
    #         is_correct = string_match(model_response, ground_truth, choices)
    #         if is_correct:
    #             correct_predictions += 1
    #         total_predictions += 1
            
    #         # Create result entry
    #         result_sample = sample.copy()
    #         result_sample['model_prediction'] = model_prediction
    #         result_sample['model_response'] = model_response
    #         result_sample['is_correct'] = is_correct
            
    #         results.append(result_sample)
            
    #         print(f"Sample {sample_id}: {'✓' if is_correct else '✗'}, {(correct_predictions/total_predictions):.4f} ({correct_predictions}/{total_predictions})")
    #         print(f"  Question: {question[:100]}...")
    #         print(f"  Ground truth: {ground_truth}")
    #         print(f"  Model prediction: {model_prediction}")
    #         print(f"  Model response: {model_response[:100]}...")
    #         print()
            
    #     except Exception as e:
    #         print(f"Error processing sample {sample_id}: {e}")
    #         # Add failed sample with error info
    #         result_sample = sample.copy()
    #         result_sample['model_prediction'] = "ERROR"
    #         result_sample['model_response'] = str(e)
    #         result_sample['is_correct'] = False
    #         results.append(result_sample)
    
    
    batch_audio_paths = []
    batch_prompts = []
    batch_samples = []

    for sample in tqdm(test_data, desc="Evaluating samples"):
        sample_id = sample['id']
        audio_path = sample['audio_id']
        question = sample['question']
        choices = sample['choices']
        ground_truth = sample['answer']

        prompt = format_multiple_choice_prompt(question, choices)

        if batch_size <= 1:
            # 单条模式：保持你原本行为
            try:
                model_response = inference(model, processor, audio_path, prompt)
                model_prediction = model_response

                is_correct = string_match(model_response, ground_truth, choices)
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1

                result_sample = sample.copy()
                result_sample['model_prediction'] = model_prediction
                result_sample['model_response'] = model_response
                result_sample['is_correct'] = is_correct
                results.append(result_sample)

                print(f"Sample {sample_id}: {'✓' if is_correct else '✗'}, {(correct_predictions/total_predictions):.4f} ({correct_predictions}/{total_predictions})")
                print(f"  Question: {question[:100]}...")
                print(f"  Ground truth: {ground_truth}")
                print(f"  Model prediction: {model_prediction}")
                print(f"  Model response: {model_response[:100]}...")
                print()

            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                result_sample = sample.copy()
                result_sample['model_prediction'] = "ERROR"
                result_sample['model_response'] = str(e)
                result_sample['is_correct'] = False
                results.append(result_sample)

        else:
            # 批量模式：先收集
            batch_samples.append(sample)
            batch_audio_paths.append(audio_path)
            batch_prompts.append(prompt)

            if len(batch_samples) == batch_size:
                try:
                    batch_responses = inference_batch(model, processor, batch_audio_paths, batch_prompts)
                except Exception as e:
                    print(f"Batch error: {e}")
                    batch_responses = [str(e)] * len(batch_samples)

                # 逐条复用你原来的 string_match & logging
                for s, model_response in zip(batch_samples, batch_responses):
                    sid = s['id']
                    q = s['question']
                    ch = s['choices']
                    gt = s['answer']

                    model_prediction = model_response
                    is_correct = string_match(model_response, gt, ch)
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1

                    result_sample = s.copy()
                    result_sample['model_prediction'] = model_prediction
                    result_sample['model_response'] = model_response
                    result_sample['is_correct'] = is_correct
                    results.append(result_sample)

                    print(f"Sample {sid}: {'✓' if is_correct else '✗'}, {(correct_predictions/total_predictions):.4f} ({correct_predictions}/{total_predictions})")
                    print(f"  Question: {q[:100]}...")
                    print(f"  Ground truth: {gt}")
                    print(f"  Model prediction: {model_prediction}")
                    print(f"  Model response: {model_response[:100]}...")
                    print()

                batch_samples.clear()
                batch_audio_paths.clear()
                batch_prompts.clear()

    # 处理尾巴（不足 batch_size 的最后一批）
    if batch_size > 1 and batch_samples:
        try:
            batch_responses = inference_batch(model, processor, batch_audio_paths, batch_prompts)
        except Exception as e:
            print(f"Final batch error: {e}")
            batch_responses = [str(e)] * len(batch_samples)

        for s, model_response in zip(batch_samples, batch_responses):
            sid = s['id']
            q = s['question']
            ch = s['choices']
            gt = s['answer']

            model_prediction = model_response
            is_correct = string_match(model_response, gt, ch)
            if is_correct:
                correct_predictions += 1
            total_predictions += 1

            result_sample = s.copy()
            result_sample['model_prediction'] = model_prediction
            result_sample['model_response'] = model_response
            result_sample['is_correct'] = is_correct
            results.append(result_sample)

            print(f"Sample {sid}: {'✓' if is_correct else '✗'}, {(correct_predictions/total_predictions):.4f} ({correct_predictions}/{total_predictions})")
            print(f"  Question: {q[:100]}...")
            print(f"  Ground truth: {gt}")
            print(f"  Model prediction: {model_prediction}")
            print(f"  Model response: {model_response[:100]}...")
            print()


    # Calculate accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    # Add evaluation metadata
    evaluation_results = {
        "metadata": {
            "model_path": model_path,
            "total_samples": len(test_data),
            "processed_samples": total_predictions,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
        },
        "results": results
    }
    
    # Save results
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation completed!")
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 Omni on MMAU test-mini subset")
    parser.add_argument(
        "--model_path", 
        type=str, 
        # default="/dllab/user/zhumuzhi/weights/Qwen3-Omni-30B-A3B-Instruct/",
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    )
    parser.add_argument(
        "--data_file", 
        type=str, 
        default="mmau-test-mini.json",
        help="Path to the MMAU test-mini JSON file"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="mmau_evaluation_results_av_reorder_fix_mini.json",
        help="Path to save the evaluation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference. 1 = single-sample inference, >1 = batch inference."
    )
    
    args = parser.parse_args()
    
    evaluate_mmau_subset(
        model_path=args.model_path,
        data_file=args.data_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()