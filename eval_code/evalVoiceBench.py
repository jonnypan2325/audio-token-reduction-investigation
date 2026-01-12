import logging
logging.getLogger("root").setLevel(logging.ERROR)
logging.getLogger("qwen_omni_utils").setLevel(logging.ERROR)
logging.getLogger("qwen").setLevel(logging.ERROR)

from datasets import load_dataset, Audio
import json
from tqdm import tqdm
from loguru import logger

import torch
from typing import Dict, List, Any
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
import argparse
from pathlib import Path
from tqdm import tqdm
import re

import numpy as np

def decode_audio_decoder(audio_dec):
    """
    Decode HF datasets torchcodec AudioDecoder into (float32 np.ndarray, sr).
    Tries multiple APIs to be compatible across versions.
    """
    # common API 1
    if hasattr(audio_dec, "get_all_samples"):
        s = audio_dec.get_all_samples()
        if isinstance(s, tuple) and len(s) == 2:
            arr, sr = s
            return np.asarray(arr, dtype=np.float32), int(sr)
        if hasattr(s, "data") and hasattr(s, "sample_rate"):
            return np.asarray(s.data, dtype=np.float32), int(s.sample_rate)

    # common API 2
    if hasattr(audio_dec, "decode"):
        s = audio_dec.decode()
        if isinstance(s, dict) and "array" in s and "sampling_rate" in s:
            return np.asarray(s["array"], dtype=np.float32), int(s["sampling_rate"])
        if hasattr(s, "data") and hasattr(s, "sample_rate"):
            return np.asarray(s.data, dtype=np.float32), int(s.sample_rate)

    raise TypeError(f"Unsupported AudioDecoder type: {type(audio_dec)}")


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

def ensure_mono_1d(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono 1D float32 waveform for Qwen.
    Accepts shapes:
      - (T,)
      - (1, T) / (T, 1)
      - (C, T) or (T, C) where C=2 etc.
    Returns:
      - (T,) float32
    """
    x = np.asarray(audio, dtype=np.float32)

    if x.ndim == 1:
        return x

    if x.ndim == 2:
        # squeeze singleton dims
        if 1 in x.shape:
            x = np.squeeze(x)
            if x.ndim == 1:
                return x

        # handle multi-channel
        # heuristic: if one dim is small (<=8) treat as channels
        if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
            # (C, T)
            return x.mean(axis=0)
        else:
            # (T, C)
            return x.mean(axis=1)

    # fallback
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError(f"Expected mono 1D audio after squeeze/merge, got shape {x.shape}")
    return x

def inference(
    model,
    processor,
    audio,
    prompt: str,
) -> str:
    """
    Run inference on a single audio sample.
    audio can be a np.ndarray or a path to audio file.
    """
    # print(f"audio shape/type: {type(audio)}, {audio.shape if isinstance(audio, np.ndarray) else 'N/A'}")
    # print(f"audio.ndim: {audio.ndim if isinstance(audio, np.ndarray) else 'N/A'}")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # print(f"Input prompt: {text}")
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
    # print(f"inputs: {inputs}")
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
    text_output = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # Extract only the generated part (remove the input prompt)
    generated_text = text_output.split("assistant\n")[-1].strip()
    # print(generated_text)
    return generated_text

# def inference_batch(
#     model,
#     processor,
#     audios_np: List[np.ndarray],
#     prompts: List[str],
# ) -> List[str]:
#     """
#     Run inference on a batch of audio samples.
#     audios_np: list of np.ndarray (each is float32 waveform)
#     prompts: list of prompt strings
#     Returns: list of generated_text (len = batch)
#     """
#     assert len(audios_np) == len(prompts)

#     # 1) build messages per sample
#     messages_list = []
#     for audio_arr, prompt in zip(audios_np, prompts):
#         messages_list.append(
#             [
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "audio", "audio": audio_arr},
#                         {"type": "text", "text": prompt},
#                     ],
#                 }
#             ]
#         )

#     # 2) chat template per sample -> list[str]
#     texts = [
#         processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
#         for m in messages_list
#     ]

#     # 3) process multimodal info per sample, then collate
#     audios_batch, images_batch, videos_batch = [], [], []
#     # ---- 关键：拍平 audio 的一层 list ----
#     audios_flat, images_flat, videos_flat = [], [], []
#     for m in messages_list:
#         a, im, v = process_mm_info(m, use_audio_in_video=True)

#         # a 通常是 [waveform]，batch 时要变成 waveform
#         if isinstance(a, (list, tuple)) and len(a) == 1:
#             a = a[0]
#         audios_flat.append(a)

#         if isinstance(im, (list, tuple)) and len(im) == 1:
#             im = im[0]
#         images_flat.append(im)

#         if isinstance(v, (list, tuple)) and len(v) == 1:
#             v = v[0]
#         videos_flat.append(v)

#     # 4) processor batching
#     inputs = processor(
#         text=texts,
#         audio=audios_flat,
#         images=images_flat,
#         videos=videos_flat,
#         return_tensors="pt",
#         padding=True,
#         use_audio_in_video=True,
#     )
#     inputs = inputs.to(model.device).to(model.dtype)

#     # 5) generate (batched)
#     output = model.generate(
#         **inputs,
#         use_audio_in_video=True,
#         return_audio=False,
#         thinker_max_new_tokens=256,
#         thinker_do_sample=False,
#         max_new_tokens=512,
#         do_sample=False,
#     )

#     # 6) decode -> list[str]
#     decoded = processor.batch_decode(
#         output,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False,
#     )

#     # 7) keep only assistant part
#     responses = []
#     for s in decoded:
#         responses.append(s.split("assistant\n")[-1].strip())
#     return responses


def inference_batch(model, processor, audios_np, prompts):
    assert len(audios_np) == len(prompts)

    messages_list = []
    for audio_arr, prompt in zip(audios_np, prompts):
        messages_list.append([{
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_arr},
                {"type": "text", "text": prompt},
            ],
        }])

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
             for m in messages_list]

    audios_flat = []
    images_flat = []
    videos_flat = []

    for m in messages_list:
        a, im, v = process_mm_info(m, use_audio_in_video=True)

        # a usually: [waveform]
        if isinstance(a, (list, tuple)) and len(a) == 1:
            a = a[0]
        audios_flat.append(a)

        # im/v might be None, [], or [None] depending on utils/version
        images_flat.append(im)
        videos_flat.append(v)

    # --------- IMPORTANT: only pass images/videos if they are not all empty/None ----------
    kwargs = dict(
        text=texts,
        audio=audios_flat,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )

    def _is_empty_mm(x):
        # treat None, [], [None] as empty
        if x is None:
            return True
        if isinstance(x, (list, tuple)) and len(x) == 0:
            return True
        if isinstance(x, (list, tuple)) and len(x) == 1 and x[0] is None:
            return True
        return False

    if not all(_is_empty_mm(x) for x in images_flat):
        kwargs["images"] = images_flat

    if not all(_is_empty_mm(x) for x in videos_flat):
        kwargs["videos"] = videos_flat

    inputs = processor(**kwargs)
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

    decoded = processor.batch_decode(
        output,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return [s.split("assistant\n")[-1].strip() for s in decoded]



def evaluate(
    model_path: str,
    dataset,
    batch_size: int = 1,
) -> None:
    """Evaluate the model on MMAU test-mini subset."""
    print(f"Running with batch_size = {batch_size}")

    # Load model and processor
    print("Loading model and processor...")
    model, processor = load_model_and_processor(model_path)
    
    # Load test data
    print(f"Loading dataset: {dataset}...")
    print(f"Total samples: {dataset.num_rows}")
    
    test_data = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    
    # Process each sample
    # results = []
    # for sample in tqdm(test_data, desc="Evaluating samples", total=dataset.num_rows):
    #     tmp = {k: v for k, v in sample.items() if k != 'audio'}
    #     (audio_path, sr) = decode_audio_decoder(sample['audio'])

    #     prompt = tmp['prompt']

    #     if batch_size <= 1:
    #         model_response = inference(model, processor, audio_path, prompt)
    #         tmp['response'] = model_response
    #         results.append(tmp)

    # return results
    
    results = []

    batch_audios = []
    batch_prompts = []
    batch_tmps = []

    for sample in tqdm(test_data, desc="Evaluating samples", total=dataset.num_rows):
        tmp = {k: v for k, v in sample.items() if k != 'audio'}
        audio_arr, sr = decode_audio_decoder(sample['audio'])
        audio_arr = ensure_mono_1d(audio_arr)
        prompt = tmp['prompt']

        if batch_size <= 1:
            model_response = inference(model, processor, audio_arr, prompt)
            tmp['response'] = model_response
            results.append(tmp)
            continue

        # batch mode: collect
        batch_audios.append(audio_arr)
        batch_prompts.append(prompt)
        batch_tmps.append(tmp)

        # run when full
        if len(batch_tmps) == batch_size:
            try:
                batch_responses = inference_batch(model, processor, batch_audios, batch_prompts)
            except Exception as e:
                logger.exception(f"Batch inference error: {e}")
                batch_responses = [f"[ERROR] {repr(e)}"] * len(batch_tmps)

            for t, r in zip(batch_tmps, batch_responses):
                t['response'] = r
                results.append(t)

            batch_audios.clear()
            batch_prompts.clear()
            batch_tmps.clear()

    # flush tail
    if batch_size > 1 and batch_tmps:
        try:
            batch_responses = inference_batch(model, processor, batch_audios, batch_prompts)
        except Exception as e:
            logger.exception(f"Final batch inference error: {e}")
            batch_responses = [f"[ERROR] {repr(e)}"] * len(batch_tmps)

        for t, r in zip(batch_tmps, batch_responses):
            t['response'] = r
            results.append(t)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 Omni on MMAU test-mini subset")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="Qwen/Qwen2.5-Omni-7B", # "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default='alpacaeval'
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference. 1 = single-sample inference, >1 = batch inference."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file to save results. If not set, defaults to '<model>-<data>-<split>.jsonl'."
    )
    
    args = parser.parse_args()
    
    dataset = load_dataset('hlt-lab/voicebench', args.data, split=args.split)
    
    results = evaluate(
                    model_path=args.model_path,
                    dataset=dataset,
                    batch_size=args.batch_size,
                    )
    # output_file = f'{args.model_path.split("/")[-1]}-{args.data}-{args.split}.jsonl'
    with open(args.output_file, 'w') as f:
        for record in results:
            json_line = json.dumps(record)  # Convert dictionary to JSON string
            f.write(json_line + '\n')


if __name__ == "__main__":
    main()




#%%
# def main():
#     parser = ArgumentParser()
#     parser.add_argument('--model', type=str, default='qwen2', choices=list(model_cls_mapping.keys()))
#     parser.add_argument('--data', type=str, default='alpacaeval')
#     parser.add_argument('--split', type=str, default='test')
#     parser.add_argument('--modality', type=str, default='audio', choices=['audio', 'text', 'ttft'])
#     args = parser.parse_args()

#     # load data
#     data = load_dataset('hlt-lab/voicebench', args.data, split=args.split)
#     data = data.cast_column("audio", Audio(sampling_rate=16_000))

#     # load model
#     model = model_cls_mapping[args.model]()
#     # data = data.select([0,1,2,3,4,5])

#     if args.modality == 'ttft':
#         # avoid cold start
#         _ = model.generate_ttft(data[0]['audio'])

#     # inference
#     results = []
#     for item in tqdm(data, total=len(data)):
#         tmp = {k: v for k, v in item.items() if k != 'audio'}
#         if args.modality == 'text':
#             response = model.generate_text(item['prompt'])
#         elif args.modality == 'audio':
#             response = model.generate_audio(item['audio'])
#         elif args.modality == 'ttft':
#             response = model.generate_ttft(item['audio'])
#         else:
#             raise NotImplementedError
#         logger.info(item['prompt'])
#         logger.info(response)
#         logger.info('====================================')
#         tmp['response'] = response
#         results.append(tmp)

#     # save results
#     output_file = f'{args.model}-{args.data}-{args.split}-{args.modality}.jsonl'
#     with open(output_file, 'w') as f:
#         for record in results:
#             json_line = json.dumps(record)  # Convert dictionary to JSON string
#             f.write(json_line + '\n')


# if __name__ == '__main__':
#     main()
