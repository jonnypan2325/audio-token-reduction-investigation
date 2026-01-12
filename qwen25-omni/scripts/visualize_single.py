#!/usr/bin/env python3
"""
Single-sample visualization script for Qwen2.5-Omni attention analysis.

Analyzes a single audio sample, detects token boundaries, and visualizes
attention patterns across all layers with color-coded boundary markers.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import json
import argparse
from pathlib import Path
import librosa
from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

import sys
sys.path.append(str(Path(__file__).parent))
from token_utils import (
    detect_token_boundaries,
    print_token_breakdown,
    get_boundary_colors,
    validate_audio_token_count
)


def process_audio_sample(
    audio_path: Path,
    model,
    processor,
    instruction: str = "Transcribe this audio.",
    max_new_tokens: int = 50
) -> dict:
    """Process audio sample and extract attention maps."""
    # Load audio
    audio_array, sr = librosa.load(str(audio_path), sr=processor.feature_extractor.sampling_rate)
    duration = len(audio_array) / sr
    
    print(f"  Audio duration: {duration:.2f}s")
    print(f"  Sample rate: {sr}Hz")
    
    # Use default system prompt to avoid audio output issues
    # Prepare conversation format without custom system prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": instruction},
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False, add_system_prompt=True)
    print(f"  Prompt: {text_prompt[:100]}...")
    
    # Process inputs
    inputs = processor(text=[text_prompt], audio=[audio_array], return_tensors="pt", sampling_rate=sr)
    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
    
    print(f"  Input IDs shape: {inputs['input_ids'].shape}")
    
    # Get feature attention mask for boundary detection
    feature_attention_mask = inputs.get('feature_attention_mask', inputs.get('attention_mask'))
    
    # Generate with attention extraction
    print(f"  Generating (max_new_tokens={max_new_tokens})...")
    with torch.no_grad():
        generate_output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,  # Use greedy decoding
            return_audio=False  # Disable audio output generation
        )
    
    # Decode output
    generated_text = processor.batch_decode(generate_output.sequences, skip_special_tokens=True)[0]
    print(f"  Generated: {generated_text[:100]}...")
    
    # Debug: check what attentions are available
    print(f"  Debug: generate_output keys: {generate_output.keys()}")
    if hasattr(generate_output, 'decoder_attentions') and generate_output.decoder_attentions:
        print(f"  Using decoder_attentions")
        first_token_attentions = generate_output.decoder_attentions[0]
    elif hasattr(generate_output, 'attentions') and generate_output.attentions:
        print(f"  Using attentions, length: {len(generate_output.attentions)}")
        # generate_output.attentions is a tuple: (token_0_layers, token_1_layers, ...)
        # Each element is a tuple of layers for that generation step
        # We want the attention from first generated token
        first_token_attentions = generate_output.attentions[0]
        print(f"  First token attentions: {type(first_token_attentions)}, length: {len(first_token_attentions)}")
        print(f"  First layer attention shape: {first_token_attentions[0].shape}")
        # Check if all nan
        sample_attn = first_token_attentions[0]
        print(f"  Sample attention - contains_nan: {torch.isnan(sample_attn).any()}, all_nan: {torch.isnan(sample_attn).all()}")
        if not torch.isnan(sample_attn).all():
            print(f"  Sample attention - min: {sample_attn[~torch.isnan(sample_attn)].min().item()}")
            print(f"  Sample attention - max: {sample_attn[~torch.isnan(sample_attn)].max().item()}")
    else:
        raise ValueError(f"No attentions found in generate output. Available keys: {list(generate_output.keys())}")
    
    # Process attention matrices
    num_layers = len(first_token_attentions)
    print(f"  Number of attention layers: {num_layers}")
    print(f"  First layer attention shape: {first_token_attentions[0].shape}")
    attention_maps = []
    
    for layer_idx in range(num_layers):
        attn = first_token_attentions[layer_idx]
        avg_attn = torch.mean(attn.float(), dim=1)[0].cpu().numpy()
        # Replace NaN with zeros (can occur in final layers)
        if np.isnan(avg_attn).any():
            print(f"  WARNING: Layer {layer_idx} has {np.isnan(avg_attn).sum()} NaN values, replacing with 0")
            avg_attn = np.nan_to_num(avg_attn, nan=0.0)
        attention_maps.append(avg_attn)
    
    attention_maps = np.stack(attention_maps, axis=0)
    print(f"  Attention maps shape: {attention_maps.shape}, min: {attention_maps.min()}, max: {attention_maps.max()}")
    
    # Detect token boundaries
    print(f"\n  Detecting token boundaries...")
    boundaries = detect_token_boundaries(
        inputs['input_ids'],
        feature_attention_mask,
        model,
        processor
    )
    
    # Validate
    is_valid, message = validate_audio_token_count(boundaries)
    print(f"  Audio token validation: {message}")
    
    result = {
        'attention_maps': attention_maps,
        'boundaries': boundaries,
        'input_ids': inputs['input_ids'][0].cpu().numpy(),
        'generated_text': generated_text,
        'duration': duration,
        'seq_len': attention_maps.shape[1],
        'num_layers': num_layers
    }
    
    return result


def visualize_attention_maps(
    attention_maps: np.ndarray,
    boundaries: dict,
    output_path: Path,
    pooling_stride: int = 5,
    log_vmin: float = 0.0007
):
    """
    Visualize attention maps in 8x4 grid with color-coded boundaries.
    
    Args:
        attention_maps: Attention matrices, shape (num_layers, seq_len, seq_len)
        boundaries: Dictionary with boundary positions
        output_path: Path to save visualization
        pooling_stride: Stride for downsampling
        log_vmin: Minimum value for log scale
    """
    num_layers = attention_maps.shape[0]
    seq_len = attention_maps.shape[1]
    
    # Apply pooling for visualization
    pooled_attention = []
    for layer_idx in range(num_layers):
        attn = attention_maps[layer_idx]
        
        if seq_len > pooling_stride:
            attn_tensor = torch.from_numpy(attn).unsqueeze(0).unsqueeze(0)
            pooled = torch.nn.functional.avg_pool2d(
                attn_tensor,
                kernel_size=pooling_stride,
                stride=pooling_stride
            ).squeeze().numpy()
        else:
            pooled = attn
        
        pooled_attention.append(pooled)
    
    # Scale boundaries
    pooled_boundaries = {k: v // pooling_stride for k, v in boundaries.items() 
                        if isinstance(v, int) and k not in ['seq_len', 'audio_token_count_detected', 'audio_token_count_formula']}
    
    # Get colors
    colors = get_boundary_colors()
    
    # Create visualization
    fig, axes = plt.subplots(8, 4, figsize=(40, 72))
    axes = axes.flatten()
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        attn_map = pooled_attention[layer_idx]
        
        # Determine colorbar range
        vmax = max(attn_map.max(), log_vmin * 10)  # Ensure vmax > vmin
        
        # Create heatmap
        sns.heatmap(
            attn_map,
            cmap='viridis',
            norm=LogNorm(vmin=log_vmin, vmax=vmax),
            ax=ax,
            cbar=layer_idx == num_layers - 1,
            xticklabels=False,
            yticklabels=False
        )
        
        # Draw boundary lines with thicker, more visible styling
        max_pos = attn_map.shape[0]
        
        # Audio BOS
        if pooled_boundaries.get('audio_bos_pos', -1) >= 0 and pooled_boundaries['audio_bos_pos'] < max_pos:
            ax.axvline(pooled_boundaries['audio_bos_pos'], color=colors['audio_bos'], 
                      linewidth=3, alpha=1.0, linestyle='-', label='Audio BOS')
            ax.axhline(pooled_boundaries['audio_bos_pos'], color=colors['audio_bos'], 
                      linewidth=3, alpha=1.0, linestyle='-')
        
        # Audio region
        if pooled_boundaries.get('audio_start', -1) >= 0 and pooled_boundaries['audio_start'] < max_pos:
            ax.axvline(pooled_boundaries['audio_start'], color=colors['audio'], 
                      linewidth=3, alpha=1.0, label='Audio Start')
            ax.axhline(pooled_boundaries['audio_start'], color=colors['audio'], 
                      linewidth=3, alpha=1.0)
        
        if pooled_boundaries.get('audio_end', -1) >= 0 and pooled_boundaries['audio_end'] < max_pos:
            ax.axvline(pooled_boundaries['audio_end'], color=colors['audio'], 
                      linewidth=3, alpha=1.0, label='Audio End')
            ax.axhline(pooled_boundaries['audio_end'], color=colors['audio'], 
                      linewidth=3, alpha=1.0)
        
        # Audio EOS
        if pooled_boundaries.get('audio_eos_pos', -1) >= 0 and pooled_boundaries['audio_eos_pos'] < max_pos:
            ax.axvline(pooled_boundaries['audio_eos_pos'], color=colors['audio_eos'], 
                      linewidth=3, alpha=1.0, linestyle='-', label='Audio EOS')
            ax.axhline(pooled_boundaries['audio_eos_pos'], color=colors['audio_eos'], 
                      linewidth=3, alpha=1.0, linestyle='-')
        
        # Instruction start
        if pooled_boundaries.get('instruction_start', -1) >= 0 and pooled_boundaries['instruction_start'] < max_pos:
            ax.axvline(pooled_boundaries['instruction_start'], color=colors['instruction'], 
                      linewidth=3, alpha=1.0, linestyle='-', label='Instruction')
            ax.axhline(pooled_boundaries['instruction_start'], color=colors['instruction'], 
                      linewidth=3, alpha=1.0, linestyle='-')
        
        # Add text annotations for the first subplot to show regions
        if layer_idx == 0:
            # System tokens region
            system_mid = pooled_boundaries.get('audio_bos_pos', 0) / 2
            ax.text(system_mid, -max_pos * 0.08, 'SYSTEM\nTOKENS', 
                   ha='center', va='top', fontsize=14, fontweight='bold',
                   color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            # Audio tokens region
            audio_start = pooled_boundaries.get('audio_start', 0)
            audio_end = pooled_boundaries.get('audio_end', max_pos)
            audio_mid = (audio_start + audio_end) / 2
            ax.text(audio_mid, -max_pos * 0.08, 'AUDIO\nTOKENS', 
                   ha='center', va='top', fontsize=14, fontweight='bold',
                   color='lime', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            # Instruction tokens region
            instr_start = pooled_boundaries.get('instruction_start', 0)
            instr_mid = (instr_start + max_pos) / 2
            ax.text(instr_mid, -max_pos * 0.08, 'TEXT\nINSTRUCTION', 
                   ha='center', va='top', fontsize=14, fontweight='bold',
                   color='red', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # Set title
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=14, pad=10, fontweight='bold')
        
        # Set axis labels with boundary positions
        if layer_idx % 4 == 0:  # Left column
            ax.set_ylabel('Target Token Position', fontsize=12, fontweight='bold')
        
        if layer_idx >= 24:  # Bottom row
            ax.set_xlabel('Source Token Position', fontsize=12, fontweight='bold')
    
    # Add comprehensive legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['audio_bos'], label=f"Audio BOS (pos {boundaries.get('audio_bos_pos', 'N/A')})"),
        Patch(facecolor=colors['audio'], label=f"Audio Tokens ({boundaries.get('audio_start', 'N/A')}-{boundaries.get('audio_end', 'N/A')})"),
        Patch(facecolor=colors['audio_eos'], label=f"Audio EOS (pos {boundaries.get('audio_eos_pos', 'N/A')})"),
        Patch(facecolor=colors['instruction'], label=f"Instruction Start (pos {boundaries.get('instruction_start', 'N/A')})")
    ]
    fig.legend(handles=legend_elements, loc='upper center', fontsize=16, 
              bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=True, fancybox=True, shadow=True)
    fig.legend(handles=legend_elements, loc='upper center', fontsize=16, 
              bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=True, fancybox=True, shadow=True)
    
    # Add main title with token breakdown info
    title_text = f'Attention Maps Across 28 Layers - Qwen2.5-Omni-7B\n'
    title_text += f'Total: {seq_len} tokens | System: {boundaries.get("audio_bos_pos", 0)} | '
    title_text += f'Audio: {boundaries.get("audio_end", 0) - boundaries.get("audio_start", 0)} | '
    title_text += f'Instruction: {seq_len - boundaries.get("instruction_start", seq_len)}'
    plt.suptitle(title_text, fontsize=18, y=0.998, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize attention for single Qwen2.5-Omni audio sample")
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                        help='Output directory')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Omni-7B',
                        help='Model name or path')
    parser.add_argument('--instruction', type=str, default='Transcribe this audio.',
                        help='Instruction text')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                        help='Maximum tokens to generate')
    parser.add_argument('--pooling_stride', type=int, default=5,
                        help='Pooling stride for visualization')
    parser.add_argument('--save_data', action='store_true',
                        help='Save attention data as npz file')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    audio_path = Path(args.audio_path)
    output_dir = (script_dir / args.output_dir).resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'attention_data').mkdir(exist_ok=True)
    (output_dir / 'visualizations').mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Qwen2.5-Omni Single Sample Attention Visualization")
    print("=" * 80)
    print(f"Audio: {audio_path}")
    print(f"Model: {args.model_name}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    # Load model
    print("\n[1/4] Loading model...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=torch.float16,
        attn_implementation="eager",
        device_map="cuda:0"
    )
    model.eval()
    print("✓ Model loaded")
    
    # Process sample
    print("\n[2/4] Processing audio sample...")
    result = process_audio_sample(audio_path, model, processor, args.instruction, args.max_new_tokens)
    
    # Print token breakdown
    print("\n[3/4] Token analysis...")
    print_token_breakdown(
        torch.from_numpy(result['input_ids']),
        result['boundaries'],
        processor,
        max_tokens=20
    )
    
    # Save attention data
    if args.save_data:
        data_path = output_dir / 'attention_data' / 'single_sample.npz'
        np.savez_compressed(
            data_path,
            attention_maps=result['attention_maps'],
            boundaries=np.array([result['boundaries']]),
            input_ids=result['input_ids'],
            duration=result['duration'],
            seq_len=result['seq_len']
        )
        print(f"✓ Attention data saved: {data_path}")
    
    # Save boundaries JSON
    boundaries_path = output_dir / 'single_sample_boundaries.json'
    with open(boundaries_path, 'w') as f:
        json.dump({
            'boundaries': {k: int(v) if isinstance(v, (int, np.integer)) else v 
                          for k, v in result['boundaries'].items()},
            'generated_text': result['generated_text'],
            'duration': result['duration'],
            'audio_path': str(audio_path)
        }, f, indent=2)
    print(f"✓ Boundaries saved: {boundaries_path}")
    
    # Visualize
    print("\n[4/4] Creating visualization...")
    viz_path = output_dir / 'visualizations' / 'single_sample_attention.png'
    visualize_attention_maps(
        result['attention_maps'],
        result['boundaries'],
        viz_path,
        args.pooling_stride
    )
    
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
