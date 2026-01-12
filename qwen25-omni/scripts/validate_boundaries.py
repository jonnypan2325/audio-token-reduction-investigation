#!/usr/bin/env python3
"""
Validation script for token boundary detection in Qwen2.5-Omni.

Tests boundary detection on samples with varying durations and validates
that the audio token count formula matches detected token counts.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import librosa
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

import sys
sys.path.append(str(Path(__file__).parent))
from token_utils import detect_token_boundaries, print_token_breakdown, validate_audio_token_count


def validate_sample(
    audio_path: Path,
    model,
    processor,
    verbose: bool = True
) -> dict:
    """
    Validate boundary detection for a single sample.
    
    Returns:
        Dictionary with validation results
    """
    # Load audio
    audio_array, sr = librosa.load(str(audio_path), sr=processor.feature_extractor.sampling_rate)
    duration = len(audio_array) / sr
    
    # Prepare inputs
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": "Transcribe this audio."},
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text_prompt], audio=[audio_array], return_tensors="pt", sampling_rate=sr)
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    
    # Detect boundaries
    feature_attention_mask = inputs.get('feature_attention_mask', inputs.get('attention_mask'))
    boundaries = detect_token_boundaries(
        inputs['input_ids'],
        feature_attention_mask,
        model,
        processor
    )
    
    # Validate
    is_valid, validation_message = validate_audio_token_count(boundaries)
    
    # Calculate mel spectrogram length for verification
    mel_length = len(audio_array) // 160  # 16kHz / 160 = 100fps
    audio_token_count_expected = ((mel_length - 1) // 2 + 1 - 2) // 2 + 1
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Sample: {audio_path.name}")
        print(f"{'='*80}")
        print(f"Duration: {duration:.2f}s")
        print(f"Audio samples: {len(audio_array)}")
        print(f"Mel frames (estimated): {mel_length}")
        print(f"Expected audio tokens (formula): {audio_token_count_expected}")
        
        # Print token breakdown
        print_token_breakdown(inputs['input_ids'], boundaries, processor, max_tokens=10)
    
    result = {
        'sample_id': audio_path.stem.replace('sample_', ''),
        'filename': audio_path.name,
        'duration': duration,
        'audio_samples': len(audio_array),
        'mel_frames_estimated': mel_length,
        'seq_len': boundaries['seq_len'],
        'system_tokens': boundaries['system_end'] - boundaries['system_start'],
        'audio_bos_pos': boundaries['audio_bos_pos'],
        'audio_start': boundaries['audio_start'],
        'audio_end': boundaries['audio_end'],
        'audio_eos_pos': boundaries['audio_eos_pos'],
        'audio_token_count_detected': boundaries['audio_token_count_detected'],
        'audio_token_count_formula': boundaries['audio_token_count_formula'],
        'audio_token_count_expected': audio_token_count_expected,
        'instruction_tokens': boundaries['instruction_end'] - boundaries['instruction_start'],
        'validation_passed': is_valid,
        'validation_message': validation_message
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate token boundary detection for Qwen2.5-Omni")
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing audio_samples')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                        help='Output directory for validation results')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Omni-7B',
                        help='Model name or path')
    parser.add_argument('--durations', type=float, nargs='+', default=[5.0, 7.0, 10.0, 13.0, 15.0],
                        help='Target durations to test (in seconds)')
    parser.add_argument('--use_saved', action='store_true',
                        help='Load from saved npz files if available')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    samples_dir = data_dir / 'audio_samples'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Qwen2.5-Omni Token Boundary Validation")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Samples directory: {samples_dir}")
    print(f"Target durations: {args.durations}")
    print("=" * 80)
    
    # Load model
    print("\nLoading model...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=torch.float16,
        attn_implementation="eager",
        device_map="cuda:0"
    )
    model.eval()
    print("✓ Model loaded")
    
    # Find samples matching target durations
    print("\nFinding samples with target durations...")
    sample_paths = sorted(samples_dir.glob("sample_*.wav"))
    
    selected_samples = []
    for target_duration in args.durations:
        best_match = None
        best_diff = float('inf')
        
        for sample_path in sample_paths:
            if sample_path in selected_samples:
                continue
            
            duration = librosa.get_duration(path=str(sample_path))
            diff = abs(duration - target_duration)
            
            if diff < best_diff:
                best_diff = diff
                best_match = sample_path
        
        if best_match:
            selected_samples.append(best_match)
            actual_duration = librosa.get_duration(path=str(best_match))
            print(f"  Target {target_duration}s → {best_match.name} ({actual_duration:.2f}s)")
    
    if not selected_samples:
        print("Error: No samples found")
        return
    
    # Validate each sample
    print(f"\nValidating {len(selected_samples)} samples...")
    results = []
    
    for sample_path in selected_samples:
        # Check for saved npz
        if args.use_saved:
            sample_id = sample_path.stem.replace('sample_', '')
            npz_path = output_dir / 'attention_data' / f'sample_{sample_id}.npz'
            
            if npz_path.exists():
                print(f"\nLoading from saved data: {npz_path}")
                data = np.load(npz_path, allow_pickle=True)
                boundaries = data['boundaries'][0].item()
                
                result = {
                    'sample_id': sample_id,
                    'filename': sample_path.name,
                    'duration': float(data['duration']),
                    'seq_len': int(data['seq_len']),
                    **{k: int(v) if isinstance(v, (int, np.integer)) else v 
                       for k, v in boundaries.items()}
                }
                results.append(result)
                continue
        
        # Run validation
        result = validate_sample(sample_path, model, processor, verbose=True)
        results.append(result)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = output_dir / 'boundary_validation.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Validation results saved: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total samples validated: {len(df)}")
    print(f"Validation passed: {df['validation_passed'].sum() if 'validation_passed' in df else 'N/A'}")
    print(f"Validation failed: {(~df['validation_passed']).sum() if 'validation_passed' in df else 'N/A'}")
    
    print("\nDuration statistics:")
    print(f"  Min: {df['duration'].min():.2f}s")
    print(f"  Max: {df['duration'].max():.2f}s")
    print(f"  Mean: {df['duration'].mean():.2f}s")
    
    print("\nSequence length statistics:")
    print(f"  Min: {df['seq_len'].min()}")
    print(f"  Max: {df['seq_len'].max()}")
    print(f"  Mean: {df['seq_len'].mean():.1f}")
    
    if 'audio_token_count_detected' in df:
        print("\nAudio token count statistics:")
        print(f"  Min: {df['audio_token_count_detected'].min()}")
        print(f"  Max: {df['audio_token_count_detected'].max()}")
        print(f"  Mean: {df['audio_token_count_detected'].mean():.1f}")
        
        if 'audio_token_count_formula' in df:
            formula_counts = df['audio_token_count_formula'].dropna()
            if len(formula_counts) > 0:
                mismatches = (df['audio_token_count_detected'] != df['audio_token_count_formula']).sum()
                print(f"\nFormula vs Detected mismatches: {mismatches}")
                if mismatches > 0:
                    print("  ⚠️  WARNING: Mismatches detected!")
    
    print("=" * 80)
    print(f"\nResults saved to: {csv_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
