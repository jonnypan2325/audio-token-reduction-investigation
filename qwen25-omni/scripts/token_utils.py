#!/usr/bin/env python3
"""
Token boundary detection utilities for Qwen2.5-Omni audio models.

This module provides functions to:
1. Detect token boundaries (system, audio_bos, audio, audio_eos, instruction)
2. Normalize boundaries for fixed-size attention maps
3. Validate audio token counts against encoder formula
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


def detect_token_boundaries(
    input_ids: torch.Tensor,
    feature_attention_mask: torch.Tensor,
    model,
    processor
) -> Dict[str, int]:
    """
    Detect token boundaries in Qwen2.5-Omni input sequence.
    
    Args:
        input_ids: Input token IDs, shape (batch_size, seq_len) or (seq_len,)
        feature_attention_mask: Audio feature attention mask for encoder
        model: Qwen2.5-Omni model instance
        processor: Qwen2.5-Omni processor instance
    
    Returns:
        Dictionary with boundary positions:
        {
            'system_start': int,
            'system_end': int,
            'audio_bos_pos': int,
            'audio_start': int,
            'audio_end': int,
            'audio_eos_pos': int,
            'instruction_start': int,
            'instruction_end': int,
            'seq_len': int
        }
    """
    # Handle batch dimension
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    
    input_ids_np = input_ids.cpu().numpy()
    seq_len = len(input_ids_np)
    
    # Get special token IDs from processor
    try:
        audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.audio_token)
        audio_bos_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.audio_bos_token)
        audio_eos_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.audio_eos_token)
    except AttributeError:
        # Fallback to known token IDs for Qwen2.5-Omni
        audio_token_id = 151646
        audio_bos_id = getattr(processor.tokenizer, 'audio_bos_token_id', None)
        audio_eos_id = getattr(processor.tokenizer, 'audio_eos_token_id', None)
    
    # Find audio BOS token
    audio_bos_positions = np.where(input_ids_np == audio_bos_id)[0] if audio_bos_id else np.array([])
    audio_bos_pos = int(audio_bos_positions[0]) if len(audio_bos_positions) > 0 else -1
    
    # Find audio EOS token
    audio_eos_positions = np.where(input_ids_np == audio_eos_id)[0] if audio_eos_id else np.array([])
    audio_eos_pos = int(audio_eos_positions[0]) if len(audio_eos_positions) > 0 else -1
    
    # Find audio token region (tokens with audio_token_id)
    audio_token_positions = np.where(input_ids_np == audio_token_id)[0]
    
    if len(audio_token_positions) > 0:
        audio_start = int(audio_token_positions[0])
        audio_end = int(audio_token_positions[-1]) + 1
    elif audio_bos_pos >= 0 and audio_eos_pos >= 0:
        # If audio tokens are already embedded, use BOS/EOS markers
        audio_start = audio_bos_pos + 1
        audio_end = audio_eos_pos
    else:
        # Fallback: estimate based on sequence structure
        audio_start = -1
        audio_end = -1
    
    # Calculate expected audio token count from encoder
    audio_token_count_formula = None
    if feature_attention_mask is not None and hasattr(model, 'audio_tower'):
        try:
            if feature_attention_mask.dim() == 2:
                audio_lengths = feature_attention_mask.sum(-1)
            else:
                audio_lengths = feature_attention_mask.sum()
                audio_lengths = audio_lengths.unsqueeze(0)
            
            _, audio_output_lengths = model.audio_tower._get_feat_extract_output_lengths(audio_lengths)
            audio_token_count_formula = int(audio_output_lengths[0].item())
        except Exception as e:
            print(f"Warning: Could not compute audio token count from encoder: {e}")
    
    # Determine system and instruction boundaries
    if audio_bos_pos >= 0:
        system_start = 0
        system_end = audio_bos_pos
    else:
        system_start = 0
        system_end = 0
    
    if audio_eos_pos >= 0:
        instruction_start = audio_eos_pos + 1
        instruction_end = seq_len
    else:
        instruction_start = audio_end if audio_end > 0 else seq_len
        instruction_end = seq_len
    
    boundaries = {
        'system_start': system_start,
        'system_end': system_end,
        'audio_bos_pos': audio_bos_pos,
        'audio_start': audio_start,
        'audio_end': audio_end,
        'audio_eos_pos': audio_eos_pos,
        'instruction_start': instruction_start,
        'instruction_end': instruction_end,
        'seq_len': seq_len,
        'audio_token_count_detected': audio_end - audio_start if audio_start >= 0 and audio_end > audio_start else 0,
        'audio_token_count_formula': audio_token_count_formula
    }
    
    return boundaries


def normalize_boundaries(
    boundaries: Dict[str, int],
    seq_len: int,
    target_size: int = 1000
) -> Dict[str, float]:
    """
    Normalize boundary positions to target size using proportional scaling.
    
    Args:
        boundaries: Dictionary with boundary positions
        seq_len: Original sequence length
        target_size: Target size for normalization (default: 1000)
    
    Returns:
        Dictionary with normalized boundary positions (float values)
    """
    if seq_len == 0:
        return {k: 0.0 for k in boundaries.keys() if k != 'seq_len'}
    
    normalized = {}
    for key, value in boundaries.items():
        if key in ['seq_len', 'audio_token_count_detected', 'audio_token_count_formula']:
            normalized[key] = value
        elif value >= 0:
            normalized[key] = (value / seq_len) * target_size
        else:
            normalized[key] = -1.0
    
    normalized['target_size'] = target_size
    return normalized


def validate_audio_token_count(boundaries: Dict[str, int]) -> Tuple[bool, str]:
    """
    Validate that detected audio token count matches formula-based count.
    
    Args:
        boundaries: Dictionary with boundary positions including counts
    
    Returns:
        Tuple of (is_valid, message)
    """
    detected = boundaries.get('audio_token_count_detected', 0)
    formula = boundaries.get('audio_token_count_formula')
    
    if formula is None:
        return True, "Formula count not available"
    
    if detected == formula:
        return True, f"Match: {detected} tokens"
    else:
        diff = abs(detected - formula)
        return False, f"Mismatch: detected={detected}, formula={formula}, diff={diff}"


def print_token_breakdown(
    input_ids: torch.Tensor,
    boundaries: Dict[str, int],
    processor,
    max_tokens: int = 20
) -> None:
    """
    Print a detailed breakdown of tokens by type.
    
    Args:
        input_ids: Input token IDs
        boundaries: Dictionary with boundary positions
        processor: Processor for decoding tokens
        max_tokens: Maximum tokens to show per region
    """
    if input_ids.dim() == 2:
        input_ids = input_ids[0]
    
    input_ids_np = input_ids.cpu().numpy()
    
    def print_region(name: str, start: int, end: int, color: str = ""):
        if start < 0 or end < 0 or start >= end:
            print(f"\n{name}: [Empty or not detected]")
            return
        
        region_ids = input_ids_np[start:end]
        count = len(region_ids)
        print(f"\n{name}: {count} tokens (positions {start}-{end-1})")
        
        # Show first few tokens
        show_count = min(max_tokens, count)
        for i in range(show_count):
            token_id = region_ids[i]
            try:
                token_str = processor.tokenizer.decode([token_id])
                print(f"  [{start+i}] ID={token_id:6d} '{token_str}'")
            except:
                print(f"  [{start+i}] ID={token_id:6d} [decode error]")
        
        if count > show_count:
            print(f"  ... ({count - show_count} more tokens)")
    
    print("=" * 80)
    print("TOKEN BREAKDOWN")
    print("=" * 80)
    print(f"Total sequence length: {boundaries['seq_len']}")
    
    print_region("SYSTEM TOKENS", boundaries['system_start'], boundaries['system_end'])
    
    if boundaries['audio_bos_pos'] >= 0:
        print_region("AUDIO BOS", boundaries['audio_bos_pos'], boundaries['audio_bos_pos'] + 1)
    
    print_region("AUDIO TOKENS", boundaries['audio_start'], boundaries['audio_end'])
    
    if boundaries['audio_eos_pos'] >= 0:
        print_region("AUDIO EOS", boundaries['audio_eos_pos'], boundaries['audio_eos_pos'] + 1)
    
    print_region("INSTRUCTION TOKENS", boundaries['instruction_start'], boundaries['instruction_end'])
    
    # Validation
    is_valid, message = validate_audio_token_count(boundaries)
    print(f"\nAudio Token Count Validation: {message}")
    if not is_valid:
        print("  ⚠️  WARNING: Mismatch detected!")
    
    print("=" * 80)


def get_boundary_colors() -> Dict[str, str]:
    """
    Get color scheme for boundary visualization.
    
    Returns:
        Dictionary mapping boundary types to colors
    """
    return {
        'system': '#1f77b4',      # Blue
        'audio_bos': '#00ffff',   # Cyan
        'audio': '#2ca02c',       # Green
        'audio_eos': '#ffff00',   # Yellow
        'instruction': '#d62728'  # Red
    }


def compute_boundary_statistics(
    boundary_list: list[Dict[str, int]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics (mean, std) for boundaries across multiple samples.
    
    Args:
        boundary_list: List of boundary dictionaries from multiple samples
    
    Returns:
        Dictionary with mean and std for each boundary type
    """
    if not boundary_list:
        return {}
    
    # Collect normalized boundaries
    all_normalized = []
    for boundaries in boundary_list:
        seq_len = boundaries.get('seq_len', 1)
        normalized = normalize_boundaries(boundaries, seq_len, target_size=1000)
        all_normalized.append(normalized)
    
    # Compute statistics
    stats = {}
    keys = ['system_start', 'system_end', 'audio_bos_pos', 'audio_start', 
            'audio_end', 'audio_eos_pos', 'instruction_start', 'instruction_end']
    
    for key in keys:
        values = [b[key] for b in all_normalized if b[key] >= 0]
        if values:
            stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': len(values)
            }
        else:
            stats[key] = {
                'mean': -1.0,
                'std': 0.0,
                'min': -1.0,
                'max': -1.0,
                'count': 0
            }
    
    return stats
