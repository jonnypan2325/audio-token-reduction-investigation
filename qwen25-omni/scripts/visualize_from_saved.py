#!/usr/bin/env python3
"""
Visualization-only script for Qwen2.5-Omni attention analysis.

Generates visualizations from saved .npz attention data without re-running inference.
Supports both single-sample and averaged (batch) visualization modes.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import json
import argparse
from pathlib import Path
import torch

import sys
sys.path.append(str(Path(__file__).parent))
from token_utils import get_boundary_colors


def load_averaged_data(output_dir: Path) -> tuple:
    """Load averaged attention results from batch processing."""
    results_path = output_dir / 'attention_data' / 'averaged_results.npz'
    stats_path = output_dir / 'boundary_statistics.json'
    
    if not results_path.exists():
        raise FileNotFoundError(f"Averaged results not found: {results_path}")
    
    print(f"Loading averaged results from: {results_path}")
    data = np.load(results_path)
    
    mean_attention = data['mean_attention']
    std_attention = data['std_attention']
    num_samples = int(data['num_samples'])
    target_size = int(data['target_size'])
    
    print(f"  Samples: {num_samples}")
    print(f"  Attention shape: {mean_attention.shape}")
    print(f"  Target size: {target_size}")
    
    # Load boundary statistics
    boundary_stats = None
    if stats_path.exists():
        print(f"Loading boundary statistics from: {stats_path}")
        with open(stats_path, 'r') as f:
            boundary_stats = json.load(f)
    
    return mean_attention, std_attention, boundary_stats, num_samples


def load_single_sample_data(npz_path: Path) -> tuple:
    """Load single sample attention data."""
    print(f"Loading single sample data from: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    attention_maps = data['attention_maps']
    boundaries = data['boundaries'][0].item() if data['boundaries'].ndim > 0 else data['boundaries'].item()
    duration = float(data['duration'])
    
    print(f"  Duration: {duration:.2f}s")
    print(f"  Attention shape: {attention_maps.shape}")
    
    return attention_maps, boundaries, duration


def visualize_averaged_attention(
    mean_attention: np.ndarray,
    std_attention: np.ndarray,
    boundary_stats: dict,
    output_path: Path,
    num_samples: int,
    log_vmin: float = 0.0007,
    show_std: bool = True
):
    """
    Visualize averaged attention maps with mean boundaries and std regions.
    
    Args:
        mean_attention: Mean attention across samples, shape (num_layers, size, size)
        std_attention: Standard deviation of attention
        boundary_stats: Statistics for boundaries (mean, std)
        output_path: Path to save visualization
        num_samples: Number of samples averaged
        log_vmin: Minimum value for log scale
        show_std: Whether to show std deviation regions
    """
    num_layers = mean_attention.shape[0]
    colors = get_boundary_colors()
    
    # Create visualization
    fig, axes = plt.subplots(8, 4, figsize=(32, 64))
    axes = axes.flatten()
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        attn_map = mean_attention[layer_idx]
        
        # Create heatmap
        sns.heatmap(
            attn_map,
            cmap='viridis',
            norm=LogNorm(vmin=log_vmin, vmax=attn_map.max()),
            ax=ax,
            cbar=layer_idx == num_layers - 1,
            xticklabels=False,
            yticklabels=False
        )
        
        # Draw mean boundary lines with std shading
        if boundary_stats:
            max_pos = attn_map.shape[0]
            
            # Audio BOS
            if 'audio_bos_pos' in boundary_stats and boundary_stats['audio_bos_pos']['count'] > 0:
                mean_pos = boundary_stats['audio_bos_pos']['mean']
                std_pos = boundary_stats['audio_bos_pos']['std']
                if mean_pos >= 0 and mean_pos < max_pos:
                    ax.axvline(mean_pos, color=colors['audio_bos'], linewidth=1.5, 
                              alpha=0.9, linestyle='--', label='Audio BOS (mean)')
                    ax.axhline(mean_pos, color=colors['audio_bos'], linewidth=1.5, 
                              alpha=0.9, linestyle='--')
                    if show_std and std_pos > 0:
                        ax.axvspan(max(0, mean_pos - std_pos), min(max_pos, mean_pos + std_pos),
                                  color=colors['audio_bos'], alpha=0.1)
            
            # Audio region
            if 'audio_start' in boundary_stats and boundary_stats['audio_start']['count'] > 0:
                mean_pos = boundary_stats['audio_start']['mean']
                std_pos = boundary_stats['audio_start']['std']
                if mean_pos >= 0 and mean_pos < max_pos:
                    ax.axvline(mean_pos, color=colors['audio'], linewidth=2,
                              alpha=0.9, label='Audio Start (mean)')
                    ax.axhline(mean_pos, color=colors['audio'], linewidth=2, alpha=0.9)
                    if show_std and std_pos > 0:
                        ax.axvspan(max(0, mean_pos - std_pos), min(max_pos, mean_pos + std_pos),
                                  color=colors['audio'], alpha=0.1)
            
            if 'audio_end' in boundary_stats and boundary_stats['audio_end']['count'] > 0:
                mean_pos = boundary_stats['audio_end']['mean']
                std_pos = boundary_stats['audio_end']['std']
                if mean_pos >= 0 and mean_pos < max_pos:
                    ax.axvline(mean_pos, color=colors['audio'], linewidth=2,
                              alpha=0.9, label='Audio End (mean)')
                    ax.axhline(mean_pos, color=colors['audio'], linewidth=2, alpha=0.9)
                    if show_std and std_pos > 0:
                        ax.axvspan(max(0, mean_pos - std_pos), min(max_pos, mean_pos + std_pos),
                                  color=colors['audio'], alpha=0.1)
            
            # Audio EOS
            if 'audio_eos_pos' in boundary_stats and boundary_stats['audio_eos_pos']['count'] > 0:
                mean_pos = boundary_stats['audio_eos_pos']['mean']
                std_pos = boundary_stats['audio_eos_pos']['std']
                if mean_pos >= 0 and mean_pos < max_pos:
                    ax.axvline(mean_pos, color=colors['audio_eos'], linewidth=1.5,
                              alpha=0.9, linestyle='--', label='Audio EOS (mean)')
                    ax.axhline(mean_pos, color=colors['audio_eos'], linewidth=1.5,
                              alpha=0.9, linestyle='--')
                    if show_std and std_pos > 0:
                        ax.axvspan(max(0, mean_pos - std_pos), min(max_pos, mean_pos + std_pos),
                                  color=colors['audio_eos'], alpha=0.1)
            
            # Instruction
            if 'instruction_start' in boundary_stats and boundary_stats['instruction_start']['count'] > 0:
                mean_pos = boundary_stats['instruction_start']['mean']
                if mean_pos >= 0 and mean_pos < max_pos:
                    ax.axvline(mean_pos, color=colors['instruction'], linewidth=1.5,
                              alpha=0.7, linestyle=':', label='Instruction (mean)')
                    ax.axhline(mean_pos, color=colors['instruction'], linewidth=1.5,
                              alpha=0.7, linestyle=':')
        
        # Set title
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=12, pad=10)
        
        # Set axis labels
        if layer_idx % 4 == 0:  # Left column
            ax.set_ylabel('Normalized Position', fontsize=10)
        
        if layer_idx >= 28:  # Bottom row
            ax.set_xlabel('Normalized Position', fontsize=10)
    
    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        # Remove duplicates
        unique = dict(zip(labels, handles))
        fig.legend(unique.values(), unique.keys(), loc='upper right', 
                  fontsize=14, bbox_to_anchor=(0.98, 0.98))
    
    title = f'Averaged Attention Maps - Qwen2.5-Omni-7B\n({num_samples} samples)'
    plt.suptitle(title, fontsize=20, y=0.995, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    plt.close()


def visualize_single_sample(
    attention_maps: np.ndarray,
    boundaries: dict,
    output_path: Path,
    pooling_stride: int = 5,
    log_vmin: float = 0.0007
):
    """Visualize single sample attention (same as visualize_single.py)."""
    num_layers = attention_maps.shape[0]
    seq_len = attention_maps.shape[1]
    colors = get_boundary_colors()
    
    # Apply pooling
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
    
    # Create visualization
    fig, axes = plt.subplots(8, 4, figsize=(32, 64))
    axes = axes.flatten()
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        attn_map = pooled_attention[layer_idx]
        
        sns.heatmap(
            attn_map,
            cmap='viridis',
            norm=LogNorm(vmin=log_vmin, vmax=attn_map.max()),
            ax=ax,
            cbar=layer_idx == num_layers - 1,
            xticklabels=False,
            yticklabels=False
        )
        
        # Draw boundaries (similar to visualize_single.py)
        max_pos = attn_map.shape[0]
        
        if pooled_boundaries.get('audio_bos_pos', -1) >= 0 and pooled_boundaries['audio_bos_pos'] < max_pos:
            ax.axvline(pooled_boundaries['audio_bos_pos'], color=colors['audio_bos'], 
                      linewidth=1.5, alpha=0.8, linestyle='--')
            ax.axhline(pooled_boundaries['audio_bos_pos'], color=colors['audio_bos'], 
                      linewidth=1.5, alpha=0.8, linestyle='--')
        
        if pooled_boundaries.get('audio_start', -1) >= 0 and pooled_boundaries['audio_start'] < max_pos:
            ax.axvline(pooled_boundaries['audio_start'], color=colors['audio'], 
                      linewidth=2, alpha=0.9)
            ax.axhline(pooled_boundaries['audio_start'], color=colors['audio'], 
                      linewidth=2, alpha=0.9)
        
        if pooled_boundaries.get('audio_end', -1) >= 0 and pooled_boundaries['audio_end'] < max_pos:
            ax.axvline(pooled_boundaries['audio_end'], color=colors['audio'], 
                      linewidth=2, alpha=0.9)
            ax.axhline(pooled_boundaries['audio_end'], color=colors['audio'], 
                      linewidth=2, alpha=0.9)
        
        if pooled_boundaries.get('audio_eos_pos', -1) >= 0 and pooled_boundaries['audio_eos_pos'] < max_pos:
            ax.axvline(pooled_boundaries['audio_eos_pos'], color=colors['audio_eos'], 
                      linewidth=1.5, alpha=0.8, linestyle='--')
            ax.axhline(pooled_boundaries['audio_eos_pos'], color=colors['audio_eos'], 
                      linewidth=1.5, alpha=0.8, linestyle='--')
        
        if pooled_boundaries.get('instruction_start', -1) >= 0 and pooled_boundaries['instruction_start'] < max_pos:
            ax.axvline(pooled_boundaries['instruction_start'], color=colors['instruction'], 
                      linewidth=1.5, alpha=0.7, linestyle=':')
            ax.axhline(pooled_boundaries['instruction_start'], color=colors['instruction'], 
                      linewidth=1.5, alpha=0.7, linestyle=':')
        
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=12, pad=10)
        
        if layer_idx % 4 == 0:
            tick_positions = np.arange(0, attn_map.shape[0], max(1, attn_map.shape[0] // 5))
            tick_labels = [str(int(pos * pooling_stride)) for pos in tick_positions]
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=8)
            ax.set_ylabel('Token Position', fontsize=10)
        
        if layer_idx >= 28:
            tick_positions = np.arange(0, attn_map.shape[1], max(1, attn_map.shape[1] // 5))
            tick_labels = [str(int(pos * pooling_stride)) for pos in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=8, rotation=45)
            ax.set_xlabel('Token Position', fontsize=10)
    
    plt.suptitle('Attention Maps - Qwen2.5-Omni-7B', 
                 fontsize=20, y=0.995, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Qwen2.5-Omni attention from saved data")
    parser.add_argument('--mode', type=str, default='averaged',
                        choices=['averaged', 'single'],
                        help='Visualization mode')
    parser.add_argument('--output_dir', type=str, default='../outputs',
                        help='Output directory with saved data')
    parser.add_argument('--sample_file', type=str,
                        help='NPZ file for single sample mode (e.g., sample_001.npz)')
    parser.add_argument('--pooling_stride', type=int, default=5,
                        help='Pooling stride for single sample visualization')
    parser.add_argument('--log_vmin', type=float, default=0.0007,
                        help='Minimum value for log colormap scale')
    parser.add_argument('--hide_std', action='store_true',
                        help='Hide std deviation regions in averaged mode')
    parser.add_argument('--output_name', type=str,
                        help='Custom output filename (without extension)')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = (script_dir / args.output_dir).resolve()
    
    print("=" * 80)
    print("Qwen2.5-Omni Attention Visualization (from saved data)")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    if args.mode == 'averaged':
        # Load averaged data
        mean_attention, std_attention, boundary_stats, num_samples = load_averaged_data(output_dir)
        
        # Generate visualization
        output_name = args.output_name or 'averaged_attention_100samples'
        output_path = output_dir / 'visualizations' / f'{output_name}.png'
        
        print("\nGenerating averaged visualization...")
        visualize_averaged_attention(
            mean_attention,
            std_attention,
            boundary_stats,
            output_path,
            num_samples,
            args.log_vmin,
            show_std=not args.hide_std
        )
    
    elif args.mode == 'single':
        # Load single sample data
        if not args.sample_file:
            # Use default
            args.sample_file = 'single_sample.npz'
        
        npz_path = output_dir / 'attention_data' / args.sample_file
        
        if not npz_path.exists():
            print(f"Error: Sample file not found: {npz_path}")
            return
        
        attention_maps, boundaries, duration = load_single_sample_data(npz_path)
        
        # Generate visualization
        output_name = args.output_name or npz_path.stem
        output_path = output_dir / 'visualizations' / f'{output_name}_attention.png'
        
        print("\nGenerating single sample visualization...")
        visualize_single_sample(
            attention_maps,
            boundaries,
            output_path,
            args.pooling_stride,
            args.log_vmin
        )
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
