#!/usr/bin/env python3
"""
Debug script to understand the audio encoding pipeline and find the correct formula.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import librosa
from pathlib import Path
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

def debug_audio_encoding():
    """Debug the audio encoding process step by step."""
    
    audio_path = Path("../data/audio_samples/sample_001.wav")
    
    print("="*80)
    print("DEBUGGING QWEN2.5-OMNI AUDIO ENCODING PIPELINE")
    print("="*80)
    print()
    
    # Load model
    print("Loading model...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        dtype=torch.float16,
        attn_implementation="eager",
        device_map="cuda:0"
    )
    model.eval()
    print("✓ Model loaded\n")
    
    # Load audio
    audio_array, sr = librosa.load(str(audio_path), sr=processor.feature_extractor.sampling_rate)
    duration = len(audio_array) / sr
    
    print(f"Audio Input:")
    print(f"  Duration: {duration:.3f} seconds")
    print(f"  Samples: {len(audio_array):,}")
    print(f"  Sample Rate: {sr} Hz")
    print()
    
    # Prepare conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": "Transcribe this audio."},
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(
        conversation, 
        add_generation_prompt=True, 
        tokenize=False, 
        add_system_prompt=True
    )
    
    # Process inputs - let's look at intermediate steps
    print("Processing audio through feature extractor...")
    audio_features = processor.feature_extractor(
        audio_array, 
        sampling_rate=sr, 
        return_tensors="pt"
    )
    
    print(f"Feature Extractor Output:")
    for key, value in audio_features.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    print()
    
    # Now process full inputs
    inputs = processor(
        text=[text_prompt], 
        audio=[audio_array], 
        return_tensors="pt", 
        sampling_rate=sr
    )
    
    print(f"Processor Output (combined):")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    print()
    
    # Move to device
    inputs = {k: v.to("cuda:0") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Get actual token IDs
    input_ids = inputs['input_ids'][0]
    audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.audio_token)
    audio_count = (input_ids == audio_token_id).sum().item()
    
    print(f"Token Analysis:")
    print(f"  Total tokens: {len(input_ids)}")
    print(f"  Audio tokens detected: {audio_count}")
    print(f"  Audio token ID: {audio_token_id}")
    print()
    
    # Try to understand the encoding formula
    if 'feature_attention_mask' in inputs:
        feature_mask = inputs['feature_attention_mask']
        mel_len = feature_mask.shape[-1]
        
        print(f"Feature Attention Mask:")
        print(f"  Shape: {feature_mask.shape}")
        print(f"  Length (mel_len): {mel_len}")
        print()
        
        # Apply formula
        formula_result = ((mel_len - 1) // 2 + 1 - 2) // 2 + 1
        print(f"Formula: ((mel_len - 1) // 2 + 1 - 2) // 2 + 1")
        print(f"  Result: {formula_result}")
        print(f"  Detected: {audio_count}")
        print(f"  Ratio: {formula_result / audio_count:.2f}:1")
        print()
        
        # Let's try to find the actual relationship
        print("Testing different formulas:")
        
        # Maybe it's based on the mel_len after some pooling?
        test_lens = [
            mel_len,
            mel_len // 2,
            mel_len // 4,
            mel_len // 8,
            mel_len // 16,
            mel_len // 32,
            mel_len // 64,
            mel_len // 98,  # 30000 / 305 ≈ 98
            mel_len // 100,
        ]
        
        for test_len in test_lens:
            result = ((test_len - 1) // 2 + 1 - 2) // 2 + 1
            print(f"  mel_len={test_len:6d} → formula={result:5d} → diff from actual: {result - audio_count:6d}")
        
        print()
        
        # Try reverse engineering: what mel_len gives us 305?
        target = audio_count
        # ((mel_len - 1) // 2 + 1 - 2) // 2 + 1 = target
        # ((mel_len - 1) // 2 - 1) // 2 + 1 = target
        # ((mel_len - 1) // 2 - 1) // 2 = target - 1
        # (mel_len - 1) // 2 - 1 = 2 * (target - 1)
        # (mel_len - 1) // 2 = 2 * (target - 1) + 1
        # mel_len - 1 = 2 * (2 * (target - 1) + 1)
        # mel_len = 2 * (2 * (target - 1) + 1) + 1
        
        reverse_mel_len = 2 * (2 * (target - 1) + 1) + 1
        print(f"Reverse engineering:")
        print(f"  To get {target} tokens, mel_len should be: {reverse_mel_len}")
        print(f"  Actual mel_len: {mel_len}")
        print(f"  Ratio: {mel_len / reverse_mel_len:.2f}:1")
        print()
    
    # Let's also check if there's any audio encoder in the model
    print("Model Architecture Info:")
    print(f"  Model type: {type(model).__name__}")
    if hasattr(model, 'audio_tower'):
        print(f"  Has audio_tower: Yes")
        print(f"    Type: {type(model.audio_tower).__name__}")
    if hasattr(model, 'audio_encoder'):
        print(f"  Has audio_encoder: Yes") 
        print(f"    Type: {type(model.audio_encoder).__name__}")
    
    # Try to trace through the model
    print("\n" + "="*80)
    print("ATTEMPTING TO TRACE AUDIO ENCODING...")
    print("="*80)
    print()
    
    with torch.no_grad():
        # Try to get intermediate representations
        if hasattr(model, 'get_audio_features'):
            print("Model has get_audio_features method - calling it...")
            try:
                audio_features_out = model.get_audio_features(inputs)
                print(f"  Audio features shape: {audio_features_out.shape}")
            except Exception as e:
                print(f"  Error: {e}")
        
        # Check model forward to see what happens
        print("\nChecking model attributes:")
        for attr in dir(model):
            if 'audio' in attr.lower() and not attr.startswith('_'):
                print(f"  {attr}")

if __name__ == '__main__':
    debug_audio_encoding()
