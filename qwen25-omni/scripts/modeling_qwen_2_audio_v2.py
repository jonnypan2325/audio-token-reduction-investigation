"""
FastV-style Audio Token Pruning for Qwen 2.5 Audio (Hook-based Implementation)
Implements attention-based pruning to reduce audio tokens during inference using forward hooks.
"""

import torch
import torch.nn as nn
from transformers import Qwen2_5OmniForConditionalGeneration
from typing import Optional, Tuple, Dict, Any


class Qwen2AudioForConditionalGenerationWithPruning(nn.Module):
    """
    Wrapper for Qwen2.5-Omni that implements FastV-style audio token pruning using hooks.
    
    Key features:
    - Calculates token importance using attention weights from last generated token
    - Applies "fake pruning" via masking (zeros out low-importance embeddings)
    - Uses forward hooks to modify hidden states between layers
    - Preserves model's original forward pass and caching behavior
    - Configurable pruning ratio and aggregation layer
    """
    
    def __init__(
        self,
        base_model: Qwen2_5OmniForConditionalGeneration,
        use_audio_pruning: bool = False,
        pruning_ratio: float = 0.5,
        aggregation_layer: int = 15,
        verbose: bool = False
    ):
        """
        Args:
            base_model: Pre-loaded Qwen2.5-Omni model
            use_audio_pruning: Enable/disable pruning
            pruning_ratio: Fraction of audio tokens to prune (0.5 = remove 50%, keep 50%)
            aggregation_layer: Layer index to start pruning (0-indexed, 15 for 28-layer model)
            verbose: Print debug information
        """
        super().__init__()
        self.model = base_model
        self.use_audio_pruning = use_audio_pruning
        self.pruning_ratio = pruning_ratio
        self.aggregation_layer = aggregation_layer
        self.verbose = verbose
        
        # Track audio token positions (set during forward pass)
        self.audio_token_start_idx = None
        self.audio_token_length = None
        self.current_pruning_mask = None
        self._pruning_mask_applied = False
        
        # Storage for attention weights
        self._attention_weights = None
        
        # Set up hooks if pruning is enabled
        self._hooks = []
        if self.use_audio_pruning:
            self._setup_pruning_hooks()
    
    def _setup_pruning_hooks(self):
        """
        Set up forward hooks to capture attention and apply pruning.
        """
        layers = self.model.thinker.model.layers
        
        # Hook for the aggregation layer - capture attention and compute mask
        def aggregation_hook(module, input, output):
            if not self.use_audio_pruning or self.audio_token_start_idx is None:
                return output
            
            hidden_states = output[0]
            
            # Check if we have attention weights in output
            if len(output) > 1 and output[1] is not None:
                attention_weights = output[1]  # [batch, num_heads, seq_len, seq_len]
                
                # Calculate importance scores
                importance_scores = self.calculate_audio_token_importance(
                    attention_weights,
                    self.audio_token_start_idx,
                    self.audio_token_length
                )
                
                # Create pruning mask
                keep_ratio = 1.0 - self.pruning_ratio
                self.current_pruning_mask = self.create_audio_pruning_mask(
                    importance_scores,
                    keep_ratio=keep_ratio
                )
                
                # Apply mask to hidden states
                audio_start = self.audio_token_start_idx
                audio_end = audio_start + self.audio_token_length
                hidden_states = hidden_states.clone()
                hidden_states[:, audio_start:audio_end, :] *= self.current_pruning_mask
                
                self._pruning_mask_applied = True
                
                if self.verbose:
                    print(f"  Applied pruning at layer {self.aggregation_layer}")
                
                # Return modified output
                return (hidden_states,) + output[1:]
            
            return output
        
        # Hook for layers after aggregation - continue applying mask
        def post_aggregation_hook(module, input, output):
            if not self.use_audio_pruning or not self._pruning_mask_applied:
                return output
            
            if self.current_pruning_mask is not None and self.audio_token_start_idx is not None:
                hidden_states = output[0].clone()
                audio_start = self.audio_token_start_idx
                audio_end = audio_start + self.audio_token_length
                hidden_states[:, audio_start:audio_end, :] *= self.current_pruning_mask
                return (hidden_states,) + output[1:]
            
            return output
        
        # Register hooks
        handle_agg = layers[self.aggregation_layer].register_forward_hook(aggregation_hook)
        self._hooks.append(handle_agg)
        
        # Register hooks for all layers after aggregation
        for idx in range(self.aggregation_layer + 1, len(layers)):
            handle = layers[idx].register_forward_hook(post_aggregation_hook)
            self._hooks.append(handle)
    
    def calculate_audio_token_importance(
        self, 
        attention_weights: torch.Tensor,
        audio_start_idx: int,
        audio_length: int
    ) -> torch.Tensor:
        """
        Calculate importance scores for audio tokens based on attention.
        
        Args:
            attention_weights: [batch, num_heads, seq_len, seq_len]
            audio_start_idx: Starting index of audio tokens in sequence
            audio_length: Number of audio tokens
            
        Returns:
            importance_scores: [batch, audio_length] - Average attention to each audio token
        """
        # Average across attention heads: [batch, seq_len, seq_len]
        attention_avg = attention_weights.mean(dim=1)
        
        # Get attention from last generated token to all audio tokens
        last_token_attention = attention_avg[:, -1, :]  # [batch, seq_len]
        
        # Extract attention to audio tokens only
        audio_attention = last_token_attention[:, audio_start_idx:audio_start_idx + audio_length]
        
        return audio_attention
    
    def create_audio_pruning_mask(
        self,
        importance_scores: torch.Tensor,
        keep_ratio: float
    ) -> torch.Tensor:
        """
        Create a binary mask for audio token pruning (fake pruning via masking).
        
        Args:
            importance_scores: [batch, audio_length]
            keep_ratio: Fraction of tokens to keep (e.g., 0.5 for 50%)
            
        Returns:
            mask: [batch, audio_length, 1] - Binary mask (1=keep, 0=prune)
        """
        batch_size, audio_length = importance_scores.shape
        num_keep = max(1, int(audio_length * keep_ratio))
        
        if self.verbose:
            print(f"  Pruning: keeping {num_keep}/{audio_length} audio tokens ({keep_ratio:.1%})")
        
        # Get indices of top-K important tokens
        _, top_indices = importance_scores.topk(num_keep, dim=1)  # [batch, num_keep]
        
        # Create binary mask
        mask = torch.zeros_like(importance_scores)  # [batch, audio_length]
        mask.scatter_(1, top_indices, 1.0)
        
        # Add dimension for broadcasting with embeddings [batch, audio_length, hidden_dim]
        mask = mask.unsqueeze(-1)  # [batch, audio_length, 1]
        
        return mask
    
    def set_audio_token_positions(self, start_idx: int, length: int):
        """
        Set the positions of audio tokens in the input sequence.
        This must be called before running inference with pruning.
        
        Args:
            start_idx: Starting index of audio tokens
            length: Number of audio tokens
        """
        self.audio_token_start_idx = start_idx
        self.audio_token_length = length
        self._pruning_mask_applied = False
        self.current_pruning_mask = None
        
        if self.verbose:
            print(f"Audio token positions: [{start_idx}:{start_idx + length}]")
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to base model with hooks active."""
        # Reset pruning state for each forward pass
        self._pruning_mask_applied = False
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Generation - delegates to base model with hooks active."""
        # Reset pruning state for generation
        self._pruning_mask_applied = False
        
        # Need to enable attention output for pruning
        if self.use_audio_pruning:
            kwargs['output_attentions'] = True
        
        return self.model.generate(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to the base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
    
    def __del__(self):
        """Clean up hooks when object is destroyed."""
        for hook in self._hooks:
            hook.remove()
