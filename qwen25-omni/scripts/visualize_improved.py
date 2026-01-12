#!/usr/bin/env python3
"""
Create improved visualization with clear labels from saved attention data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
import json
from pathlib import Path
import argparse


def load_data(npz_path, boundaries_json):
    """Load attention data and boundaries."""
    data = np.load(npz_path)
    with open(boundaries_json, 'r') as f:
        boundaries = json.load(f)
    
    return data['attention_maps'], boundaries


def visualize_with_labels(attention_maps, boundaries, output_path, pooling_stride=5, log_vmin=0.0007):
    """Create visualization with clear boundary labels."""
    num_layers, seq_len, _ = attention_maps.shape
    
    # Pool attention maps for visualization
    pooled_attention = []
    for layer_idx in range(num_layers):
        attn = attention_maps[layer_idx]
        if seq_len > pooling_stride:
            from scipy.ndimage import zoom
            scale = 1.0 / pooling_stride
            pooled = zoom(attn, (scale, scale), order=1)
        else:
            pooled = attn
        pooled_attention.append(pooled)
    
    # Scale boundaries
    pooled_boundaries = {k: int(v // pooling_stride) for k, v in boundaries.items() 
                        if isinstance(v, int) and k not in ['seq_len', 'audio_token_count_detected', 'audio_token_count_formula']}
    
    # Colors
    colors = {
        'audio_bos': 'cyan',
        'audio': 'lime',
        'audio_eos': 'yellow',
        'instruction': 'red'
    }
    
    # Create figure
    fig, axes = plt.subplots(7, 4, figsize=(32, 56))
    axes = axes.flatten()
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        attn_map = pooled_attention[layer_idx]
        max_pos = attn_map.shape[0]
        
        # Determine colorbar range
        vmax = max(attn_map.max(), log_vmin * 10)
        
        # Create heatmap
        sns.heatmap(
            attn_map,
            cmap='viridis',
            norm=LogNorm(vmin=log_vmin, vmax=vmax),
            ax=ax,
            cbar=(layer_idx == num_layers - 1),
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Attention Weight', 'shrink': 0.8} if layer_idx == num_layers - 1 else {}
        )
        
        # Draw VERY THICK, BRIGHT boundary lines with white outline for maximum visibility
        line_style = {'linewidth': 8, 'alpha': 1.0, 'linestyle': '-'}
        outline_style = {'linewidth': 10, 'alpha': 0.8, 'linestyle': '-', 'color': 'white'}
        
        # Audio BOS (cyan with white outline)
        if 'audio_bos_pos' in pooled_boundaries and pooled_boundaries['audio_bos_pos'] < max_pos:
            pos = pooled_boundaries['audio_bos_pos']
            ax.axvline(pos, **outline_style)
            ax.axhline(pos, **outline_style)
            ax.axvline(pos, color=colors['audio_bos'], **line_style)
            ax.axhline(pos, color=colors['audio_bos'], **line_style)
        
        # Audio start (lime green with white outline)
        if 'audio_start' in pooled_boundaries and pooled_boundaries['audio_start'] < max_pos:
            pos = pooled_boundaries['audio_start']
            ax.axvline(pos, **outline_style)
            ax.axhline(pos, **outline_style)
            ax.axvline(pos, color=colors['audio'], **line_style)
            ax.axhline(pos, color=colors['audio'], **line_style)
        
        # Audio end (lime green with white outline)
        if 'audio_end' in pooled_boundaries and pooled_boundaries['audio_end'] < max_pos:
            pos = pooled_boundaries['audio_end']
            ax.axvline(pos, **outline_style)
            ax.axhline(pos, **outline_style)
            ax.axvline(pos, color=colors['audio'], **line_style)
            ax.axhline(pos, color=colors['audio'], **line_style)
        
        # Audio EOS (yellow with white outline)
        if 'audio_eos_pos' in pooled_boundaries and pooled_boundaries['audio_eos_pos'] < max_pos:
            pos = pooled_boundaries['audio_eos_pos']
            ax.axvline(pos, **outline_style)
            ax.axhline(pos, **outline_style)
            ax.axvline(pos, color=colors['audio_eos'], **line_style)
            ax.axhline(pos, color=colors['audio_eos'], **line_style)
        
        # Instruction start (red with white outline)
        if 'instruction_start' in pooled_boundaries and pooled_boundaries['instruction_start'] < max_pos:
            pos = pooled_boundaries['instruction_start']
            ax.axvline(pos, **outline_style)
            ax.axhline(pos, **outline_style)
            ax.axvline(pos, color=colors['instruction'], **line_style)
            ax.axhline(pos, color=colors['instruction'], **line_style)
        
        # Add region labels on first subplot
        if layer_idx == 0:
            y_offset = -max_pos * 0.10
            fontsize = 18
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.8, edgecolor='white', linewidth=2)
            
            # System tokens
            if 'audio_bos_pos' in pooled_boundaries:
                mid = pooled_boundaries['audio_bos_pos'] / 2
                ax.text(mid, y_offset, 'SYSTEM\nTOKENS', 
                       ha='center', va='top', fontsize=fontsize, fontweight='bold',
                       color='white', bbox=bbox_props)
            
            # Audio tokens
            if 'audio_start' in pooled_boundaries and 'audio_end' in pooled_boundaries:
                mid = (pooled_boundaries['audio_start'] + pooled_boundaries['audio_end']) / 2
                count = boundaries.get('audio_end', 0) - boundaries.get('audio_start', 0)
                ax.text(mid, y_offset, f'AUDIO\nTOKENS\n({count})', 
                       ha='center', va='top', fontsize=fontsize, fontweight='bold',
                       color='lime', bbox=bbox_props)
            
            # Instruction tokens
            if 'instruction_start' in pooled_boundaries:
                mid = (pooled_boundaries['instruction_start'] + max_pos) / 2
                count = seq_len - boundaries.get('instruction_start', seq_len)
                ax.text(mid, y_offset, f'TEXT\nINSTRUCTION\n({count} tokens)', 
                       ha='center', va='top', fontsize=fontsize, fontweight='bold',
                       color='red', bbox=bbox_props)
        
        # Title
        ax.set_title(f'Layer {layer_idx + 1}', fontsize=16, pad=10, fontweight='bold')
        
        # Axis labels
        if layer_idx % 4 == 0:
            ax.set_ylabel('Target Token →', fontsize=14, fontweight='bold')
        if layer_idx >= 24:
            ax.set_xlabel('Source Token →', fontsize=14, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(num_layers, len(axes)):
        axes[idx].axis('off')
    
    # Create legend
    legend_elements = [
        Patch(facecolor=colors['audio_bos'], edgecolor='white', linewidth=2,
              label=f'Audio BOS (pos {boundaries.get("audio_bos_pos", "N/A")})'),
        Patch(facecolor=colors['audio'], edgecolor='white', linewidth=2,
              label=f'Audio Tokens (pos {boundaries.get("audio_start", "N/A")}-{boundaries.get("audio_end", "N/A")})'),
        Patch(facecolor=colors['audio_eos'], edgecolor='white', linewidth=2,
              label=f'Audio EOS (pos {boundaries.get("audio_eos_pos", "N/A")})'),
        Patch(facecolor=colors['instruction'], edgecolor='white', linewidth=2,
              label=f'Instruction (pos {boundaries.get("instruction_start", "N/A")}+)')
    ]
    
    fig.legend(handles=legend_elements, loc='upper center', fontsize=20, 
              bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=True, 
              fancybox=True, shadow=True, edgecolor='black', facecolor='white')
    
    # Title with breakdown
    title_text = 'Attention Maps Across 28 Layers - Qwen2.5-Omni-7B\n'
    sys_count = boundaries.get('audio_bos_pos', 0)
    audio_count = boundaries.get('audio_end', 0) - boundaries.get('audio_start', 0)
    instr_count = seq_len - boundaries.get('instruction_start', seq_len)
    title_text += f'Sequence: {seq_len} tokens = {sys_count} system + {audio_count} audio + {instr_count} instruction'
    
    plt.suptitle(title_text, fontsize=22, y=0.998, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.988])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Improved visualization saved: {output_path}")
    print(f"  Resolution: {fig.get_size_inches()[0]*300:.0f}x{fig.get_size_inches()[1]*300:.0f} pixels")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Create improved attention visualization with labels")
    parser.add_argument('--npz', type=str, 
                       default='../outputs/attention_data/single_sample.npz',
                       help='Path to NPZ file with attention data')
    parser.add_argument('--boundaries', type=str,
                       default='../outputs/single_sample_boundaries.json',
                       help='Path to boundaries JSON file')
    parser.add_argument('--output', type=str,
                       default='../outputs/visualizations/attention_labeled.png',
                       help='Output path for visualization')
    parser.add_argument('--pooling_stride', type=int, default=5,
                       help='Pooling stride for downsampling')
    
    args = parser.parse_args()
    
    npz_path = Path(args.npz)
    boundaries_json = Path(args.boundaries)
    output_path = Path(args.output)
    
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    if not boundaries_json.exists():
        raise FileNotFoundError(f"Boundaries JSON not found: {boundaries_json}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Creating Improved Attention Visualization")
    print("="*80)
    print(f"Input NPZ: {npz_path}")
    print(f"Boundaries: {boundaries_json}")
    print(f"Output: {output_path}")
    print("="*80)
    
    attention_maps, boundaries = load_data(npz_path, boundaries_json)
    print(f"\nLoaded attention maps: {attention_maps.shape}")
    print(f"Token boundaries: {boundaries}")
    
    visualize_with_labels(attention_maps, boundaries, output_path, args.pooling_stride)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
