#!/usr/bin/env python3
"""
Visualize attention maps for Qwen2-Audio-7B-Instruct across all 32 layers.
Generates FastV-style heatmap visualization to analyze attention patterns on audio tokens.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import urllib.request
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import librosa
import soundfile as sf

# Configuration
MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"
AUDIO_URL = "https://www.openslr.org/resources/12/1272-128104-0000.flac"  # LibriSpeech sample
AUDIO_PATH = "test_audio.wav"
INSTRUCTION = "Transcribe this audio."
MAX_NEW_TOKENS = 50
POOLING_STRIDE = 5  # Reduced from 20 for better visualization of shorter sequences
LOG_VMIN = 0.0007

print("=" * 80)
print("Qwen2-Audio Attention Visualization")
print("=" * 80)

# Step 1: Download audio sample
print("\n[1/6] Downloading test audio sample...")
if not os.path.exists(AUDIO_PATH):
    try:
        # Download LibriSpeech sample
        temp_flac = "temp_audio.flac"
        urllib.request.urlretrieve(AUDIO_URL, temp_flac)
        
        # Load and convert to wav, ensure under 30 seconds
        audio, sr = librosa.load(temp_flac, sr=16000, duration=25)
        sf.write(AUDIO_PATH, audio, sr)
        os.remove(temp_flac)
        print(f"✓ Downloaded and saved audio: {AUDIO_PATH}")
        print(f"  Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}Hz")
    except Exception as e:
        print(f"✗ Failed to download from URL: {e}")
        print("  Using librosa example audio instead...")
        audio, sr = librosa.load(librosa.example('trumpet'), sr=16000, duration=25)
        sf.write(AUDIO_PATH, audio, sr)
        print(f"✓ Saved example audio: {AUDIO_PATH}")
else:
    print(f"✓ Audio file already exists: {AUDIO_PATH}")
    audio, sr = librosa.load(AUDIO_PATH, sr=16000)
    print(f"  Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}Hz")

# Step 2: Load model and processor
print("\n[2/6] Loading Qwen2-Audio-7B-Instruct model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    attn_implementation="eager"  # Required for output_attentions
)
model = model.to("cuda:0")
model.eval()
print(f"✓ Model loaded on GPU 0")
# Get number of layers from the language model part
num_layers = model.language_model.config.num_hidden_layers if hasattr(model, 'language_model') else 32
print(f"  Model layers: {num_layers}")

# Step 3: Prepare inputs and generate with attention output
print("\n[3/6] Running inference with attention extraction...")

# For Qwen2-Audio, we need to provide audio in conversation format with proper audio path
conversation = [
    {
        "role": "user", 
        "content": [
            {"type": "audio", "audio_url": AUDIO_PATH},
            {"type": "text", "text": INSTRUCTION},
        ]
    }
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
print(f"  Prompt: {text_prompt[:200]}...")

# Load audio as numpy array
audio_array, sr = librosa.load(AUDIO_PATH, sr=processor.feature_extractor.sampling_rate)

# Process with both text and audio
inputs = processor(text=[text_prompt], audio=[audio_array], return_tensors="pt", sampling_rate=sr)
inputs = inputs.to("cuda:0")

print(f"  Input IDs shape: {inputs['input_ids'].shape}")
print(f"  Input keys: {list(inputs.keys())}")
if 'audio_features' in inputs or 'input_features' in inputs:
    audio_key = 'audio_features' if 'audio_features' in inputs else 'input_features'
    print(f"  Audio features shape: {inputs[audio_key].shape}")
else:
    print(f"  WARNING: No audio features found in inputs!")
print(f"  Generating with max_new_tokens={MAX_NEW_TOKENS}...")

with torch.no_grad():
    generate_output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        output_attentions=True,
        return_dict_in_generate=True
    )

output_ids = generate_output.sequences
attentions = generate_output.attentions  # Tuple of tuples: (num_generated_tokens, num_layers)

print(f" Generation complete")
print(f"  Generated {len(attentions)} tokens")
print(f"  Output sequence length: {output_ids.shape[1]}")

# Decode output
generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
print(f"  Generated text: {generated_text[:200]}...")

# Step 4: Aggregate and process attention matrices
print("\n[4/6] Processing attention matrices across all layers...")

num_layers = len(attentions[0])  # Should be 32
print(f"  Number of layers: {num_layers}")

# Average attention across all generated tokens for each layer
layer_attentions = []
for layer_idx in range(num_layers):
    # Collect attention for this layer across all generation steps
    layer_attn_list = []
    for step_idx in range(len(attentions)):
        # attentions[step_idx][layer_idx] has shape: [batch, num_heads, seq_len, seq_len]
        attn = attentions[step_idx][layer_idx]
        layer_attn_list.append(attn)
    
    # Stack and average across generation steps
    # Each step has different seq_len (grows by 1), so we take the last step (largest)
    # Or average attention patterns for the prompt tokens
    # Following FastV: use attention from first generated token (most informative)
    first_token_attn = layer_attn_list[0]  # Shape: [batch, num_heads, seq_len, seq_len]
    
    # Average across attention heads
    avg_attn = torch.mean(first_token_attn, dim=1)[0].float()  # Shape: [seq_len, seq_len]
    
    # Apply average pooling for downsampling
    if avg_attn.shape[0] > POOLING_STRIDE:
        pooled_attn = torch.nn.functional.avg_pool2d(
            avg_attn.unsqueeze(0).unsqueeze(0),
            kernel_size=POOLING_STRIDE,
            stride=POOLING_STRIDE
        ).squeeze(0).squeeze(0)
    else:
        pooled_attn = avg_attn
    
    layer_attentions.append(pooled_attn.cpu().numpy())
    
    if layer_idx == 0:
        print(f"  Original attention shape: {avg_attn.shape}")
        print(f"  Pooled attention shape: {pooled_attn.shape}")

print(f" Processed {num_layers} layers")

# Identify token boundaries (before pooling)
print("\n[5/6] Identifying token boundaries...")
input_ids_np = inputs['input_ids'][0].cpu().numpy()
seq_len = len(input_ids_np)

# Find special tokens (this is model-specific, adjust as needed)
# For Qwen2-Audio, audio tokens are typically marked by special token IDs
# We'll need to analyze the input_ids to find boundaries

print(f"  Total input sequence length: {seq_len}")
print(f"  First 20 tokens: {input_ids_np[:20]}")
print(f"  Last 20 tokens: {input_ids_np[-20:]}")

# Simple heuristic: look for large blocks of repeated or sequential tokens (audio tokens)
# For visualization, we'll mark approximate boundaries
# TODO: Refine this based on actual token inspection

# For now, use simple estimation
# Typically: [system tokens] [audio tokens] [instruction tokens]
system_end = 10  # Approximate
audio_end = seq_len - 50  # Approximate (audio tokens are usually a large middle chunk)
instruction_end = seq_len

boundaries = {
    'system': 0,
    'audio_start': system_end,
    'audio_end': audio_end,
    'instruction_end': instruction_end
}

# Scale boundaries by pooling stride
pooled_boundaries = {k: v // POOLING_STRIDE for k, v in boundaries.items()}

print(f"  Estimated boundaries (original): {boundaries}")
print(f"  Pooled boundaries: {pooled_boundaries}")

# Step 5: Create visualization
print("\n[6/6] Creating 8x4 grid visualization...")

fig, axes = plt.subplots(8, 4, figsize=(32, 64))
axes = axes.flatten()

for layer_idx in range(num_layers):
    ax = axes[layer_idx]
    attn_map = layer_attentions[layer_idx]
    
    # Create heatmap
    sns.heatmap(
        attn_map,
        cmap='viridis',
        norm=LogNorm(vmin=LOG_VMIN, vmax=attn_map.max()),
        ax=ax,
        cbar=layer_idx == num_layers - 1,  # Show colorbar only on last plot
        xticklabels=False,
        yticklabels=False
    )
    
    # Draw boundary lines
    if pooled_boundaries['audio_start'] < attn_map.shape[0]:
        ax.axvline(pooled_boundaries['audio_start'], color='white', linewidth=1, alpha=0.7)
        ax.axhline(pooled_boundaries['audio_start'], color='white', linewidth=1, alpha=0.7)
    
    if pooled_boundaries['audio_end'] < attn_map.shape[0]:
        ax.axvline(pooled_boundaries['audio_end'], color='white', linewidth=1, alpha=0.7)
        ax.axhline(pooled_boundaries['audio_end'], color='white', linewidth=1, alpha=0.7)
    
    # Set title
    ax.set_title(f'Layer {layer_idx + 1}', fontsize=12, pad=10)
    
    # Set axis labels (show original token positions)
    if layer_idx % 4 == 0:  # Left column
        tick_positions = np.arange(0, attn_map.shape[0], max(1, attn_map.shape[0] // 5))
        tick_labels = [str(int(pos * POOLING_STRIDE)) for pos in tick_positions]
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_ylabel('Token Position', fontsize=10)
    
    if layer_idx >= 28:  # Bottom row
        tick_positions = np.arange(0, attn_map.shape[1], max(1, attn_map.shape[1] // 5))
        tick_labels = [str(int(pos * POOLING_STRIDE)) for pos in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=45)
        ax.set_xlabel('Token Position', fontsize=10)

plt.suptitle('Attention Maps Across 32 Layers - Qwen2-Audio-7B-Instruct', 
             fontsize=20, y=0.995, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.995])

# Save figure
output_path = "attention_maps_qwen2_audio.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f" Visualization saved: {output_path}")

print("\n" + "=" * 80)
print("Visualization complete!")
print("=" * 80)
