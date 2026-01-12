#!/usr/bin/env python3
"""
Batch processing script for Qwen2.5-Omni attention analysis.

Processes 100 audio samples in batches of 10, extracting attention maps
and computing averaged attention patterns across all samples.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import librosa
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

import sys
sys.path.append(str(Path(__file__).parent))
from token_utils import detect_token_boundaries, normalize_boundaries, compute_boundary_statistics


def load_model_and_processor(model_name: str = "Qwen/Qwen2.5-Omni-7B"):
    """Load Qwen2.5-Omni model and processor."""
    print("Loading Qwen2.5-Omni model...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        attn_implementation="eager",  # Required for output_attentions
        device_map="cuda:0"
    )
    model.eval()
    print(f"✓ Model loaded on GPU")
    
    return model, processor


def process_single_sample(
    audio_path: Path,
    model,
    processor,
    instruction: str = "Transcribe this audio.",
    max_new_tokens: int = 50
) -> dict:
    """
    Process a single audio sample and extract attention maps.
    
    Returns:
        Dictionary with attention_maps, boundaries, input_ids, and metadata
    """
    # Load audio
    audio_array, sr = librosa.load(str(audio_path), sr=processor.feature_extractor.sampling_rate)
    duration = len(audio_array) / sr
    
    # Prepare conversation format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": instruction},
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    
    # Process inputs
    inputs = processor(text=[text_prompt], audio=[audio_array], return_tensors="pt", sampling_rate=sr)
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    
    # Get feature attention mask for boundary detection
    feature_attention_mask = inputs.get('feature_attention_mask', inputs.get('attention_mask'))
    
    # Generate with attention extraction
    with torch.no_grad():
        generate_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True
        )
    
    # Extract attention from first generated token
    attentions = generate_output.attentions  # (num_generated_tokens, num_layers)
    first_token_attentions = attentions[0]  # First generated token
    
    # Process attention matrices
    num_layers = len(first_token_attentions)
    attention_maps = []
    
    for layer_idx in range(num_layers):
        # Shape: [batch, num_heads, seq_len, seq_len]
        attn = first_token_attentions[layer_idx]
        # Average across heads
        avg_attn = torch.mean(attn, dim=1)[0].float().cpu().numpy()  # [seq_len, seq_len]
        attention_maps.append(avg_attn)
    
    # Stack into 3D array: (num_layers, seq_len, seq_len)
    attention_maps = np.stack(attention_maps, axis=0)
    
    # Detect token boundaries
    boundaries = detect_token_boundaries(
        inputs['input_ids'],
        feature_attention_mask,
        model,
        processor
    )
    
    result = {
        'attention_maps': attention_maps,
        'boundaries': boundaries,
        'input_ids': inputs['input_ids'][0].cpu().numpy(),
        'duration': duration,
        'seq_len': attention_maps.shape[1],
        'num_layers': num_layers
    }
    
    return result


def resize_attention_map(attention_map: np.ndarray, target_size: int = 1000) -> np.ndarray:
    """
    Resize attention map to target size using bilinear interpolation.
    
    Args:
        attention_map: Input attention map, shape (seq_len, seq_len)
        target_size: Target size for both dimensions
    
    Returns:
        Resized attention map, shape (target_size, target_size)
    """
    # Convert to torch tensor
    attn_tensor = torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Resize using bilinear interpolation
    resized = torch.nn.functional.interpolate(
        attn_tensor,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    )
    
    return resized.squeeze().numpy()


def process_batch(
    sample_paths: list,
    model,
    processor,
    output_dir: Path,
    batch_num: int,
    target_size: int = 1000
) -> tuple:
    """
    Process a batch of samples and accumulate attention maps.
    
    Returns:
        Tuple of (attention_sums, attention_squared_sums, boundary_list, num_processed)
    """
    attention_sums = None
    attention_squared_sums = None
    boundary_list = []
    
    for sample_path in tqdm(sample_paths, desc=f"Batch {batch_num}", leave=False):
        sample_id = sample_path.stem.replace('sample_', '')
        
        try:
            # Process sample
            result = process_single_sample(sample_path, model, processor)
            
            # Save raw attention data
            save_path = output_dir / 'attention_data' / f'sample_{sample_id}.npz'
            np.savez_compressed(
                save_path,
                attention_maps=result['attention_maps'],
                boundaries=np.array([result['boundaries']]),  # Wrap in array for saving
                input_ids=result['input_ids'],
                duration=result['duration'],
                seq_len=result['seq_len']
            )
            
            # Resize attention maps
            num_layers = result['num_layers']
            resized_attention = np.zeros((num_layers, target_size, target_size))
            
            for layer_idx in range(num_layers):
                resized_attention[layer_idx] = resize_attention_map(
                    result['attention_maps'][layer_idx],
                    target_size
                )
            
            # Accumulate for averaging
            if attention_sums is None:
                attention_sums = resized_attention.copy()
                attention_squared_sums = resized_attention ** 2
            else:
                attention_sums += resized_attention
                attention_squared_sums += resized_attention ** 2
            
            # Collect boundaries
            boundary_list.append(result['boundaries'])
            
        except Exception as e:
            print(f"\nError processing {sample_path.name}: {e}")
            continue
    
    return attention_sums, attention_squared_sums, boundary_list


def main():
    parser = argparse.ArgumentParser(description="Batch process audio samples for attention analysis")
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing audio_samples')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                        help='Directory for outputs')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Omni-7B',
                        help='Model name or path')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Number of samples per batch')
    parser.add_argument('--target_size', type=int, default=1000,
                        help='Target size for attention maps')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to process')
    parser.add_argument('--resume_batch', type=int, default=0,
                        help='Resume from specific batch number (0 = start fresh)')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = (script_dir / args.data_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    samples_dir = data_dir / 'audio_samples'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'attention_data').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Qwen2.5-Omni Batch Attention Analysis")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Samples directory: {samples_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Target attention size: {args.target_size}×{args.target_size}")
    print("=" * 80)
    
    # Load model
    model, processor = load_model_and_processor(args.model_name)
    num_layers = model.config.text_config.num_hidden_layers
    
    # Get sample paths
    sample_paths = sorted(samples_dir.glob("sample_*.wav"))[:args.max_samples]
    
    if len(sample_paths) == 0:
        print("Error: No samples found. Run download_samples.py first.")
        return
    
    print(f"\nFound {len(sample_paths)} samples")
    
    # Initialize accumulation arrays
    if args.resume_batch > 0:
        checkpoint_path = output_dir / 'attention_data' / f'checkpoint_batch{args.resume_batch-1}.npz'
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = np.load(checkpoint_path)
            attention_sums = checkpoint['attention_sums']
            attention_squared_sums = checkpoint['attention_squared_sums']
            boundary_list = checkpoint['boundary_list'].tolist()
            start_batch = args.resume_batch
        else:
            print(f"Warning: Checkpoint not found, starting fresh")
            attention_sums = None
            attention_squared_sums = None
            boundary_list = []
            start_batch = 0
    else:
        attention_sums = None
        attention_squared_sums = None
        boundary_list = []
        start_batch = 0
    
    # Process in batches
    num_batches = (len(sample_paths) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(start_batch, num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(sample_paths))
        batch_samples = sample_paths[start_idx:end_idx]
        
        print(f"\n[Batch {batch_idx + 1}/{num_batches}] Processing samples {start_idx + 1}-{end_idx}")
        
        batch_sums, batch_squared_sums, batch_boundaries = process_batch(
            batch_samples,
            model,
            processor,
            output_dir,
            batch_idx + 1,
            args.target_size
        )
        
        # Accumulate
        if attention_sums is None:
            attention_sums = batch_sums
            attention_squared_sums = batch_squared_sums
        else:
            if batch_sums is not None:
                attention_sums += batch_sums
                attention_squared_sums += batch_squared_sums
        
        boundary_list.extend(batch_boundaries)
        
        # Save checkpoint
        checkpoint_path = output_dir / 'attention_data' / f'checkpoint_batch{batch_idx}.npz'
        np.savez_compressed(
            checkpoint_path,
            attention_sums=attention_sums,
            attention_squared_sums=attention_squared_sums,
            boundary_list=np.array(boundary_list),
            num_samples=len(boundary_list)
        )
        print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    # Compute final statistics
    num_samples = len(boundary_list)
    print(f"\n{'='*80}")
    print(f"Computing final statistics from {num_samples} samples...")
    
    mean_attention = attention_sums / num_samples
    mean_squared = attention_squared_sums / num_samples
    std_attention = np.sqrt(mean_squared - mean_attention ** 2)
    
    # Compute boundary statistics
    boundary_stats = compute_boundary_statistics(boundary_list)
    
    # Save results
    results_path = output_dir / 'attention_data' / 'averaged_results.npz'
    np.savez_compressed(
        results_path,
        mean_attention=mean_attention,
        std_attention=std_attention,
        num_samples=num_samples,
        target_size=args.target_size
    )
    print(f"✓ Averaged results saved: {results_path}")
    
    stats_path = output_dir / 'boundary_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(boundary_stats, f, indent=2)
    print(f"✓ Boundary statistics saved: {stats_path}")
    
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total samples processed: {num_samples}")
    print(f"Attention maps shape: {mean_attention.shape}")
    print(f"Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Run visualize_from_saved.py to generate visualizations")
    print(f"  2. Check {stats_path} for boundary statistics")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
