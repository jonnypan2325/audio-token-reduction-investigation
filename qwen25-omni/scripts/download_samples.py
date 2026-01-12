#!/usr/bin/env python3
"""
Download speech-only audio samples from LibriSpeech for Qwen2.5-Omni analysis.

Downloads 100 samples with durations between 5-15 seconds from LibriSpeech
dev-clean and test-clean splits.
"""

import os
import urllib.request
import tarfile
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse


def download_and_extract_librispeech(split: str, output_dir: Path) -> Path:
    """
    Download and extract LibriSpeech split.
    
    Args:
        split: LibriSpeech split name (e.g., 'dev-clean', 'test-clean')
        output_dir: Directory to extract files
    
    Returns:
        Path to extracted files
    """
    base_url = "https://www.openslr.org/resources/12"
    split_url = f"{base_url}/{split}.tar.gz"
    tar_path = output_dir / f"{split}.tar.gz"
    
    print(f"Downloading {split}...")
    
    if not tar_path.exists():
        # Download with progress bar
        def progress_hook(block_num, block_size, total_size):
            if hasattr(progress_hook, 'pbar'):
                progress_hook.pbar.update(block_size)
            else:
                progress_hook.pbar = tqdm(total=total_size, unit='B', unit_scale=True)
        
        try:
            urllib.request.urlretrieve(split_url, tar_path, reporthook=progress_hook)
            if hasattr(progress_hook, 'pbar'):
                progress_hook.pbar.close()
        except Exception as e:
            print(f"Error downloading {split}: {e}")
            if tar_path.exists():
                tar_path.unlink()
            raise
    
    print(f"Extracting {split}...")
    extract_dir = output_dir / "LibriSpeech"
    
    if not extract_dir.exists():
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(output_dir)
    
    return extract_dir / split.replace('-', '-')


def find_audio_files(librispeech_dir: Path, min_duration: float = 5.0, max_duration: float = 15.0) -> list:
    """
    Find all audio files in LibriSpeech directory within duration range.
    
    Args:
        librispeech_dir: Path to LibriSpeech split directory
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
    
    Returns:
        List of tuples (file_path, duration, speaker_id, chapter_id)
    """
    audio_files = []
    
    print(f"Scanning for audio files ({min_duration}-{max_duration}s)...")
    
    # LibriSpeech structure: speaker_id/chapter_id/*.flac
    flac_files = list(librispeech_dir.rglob("*.flac"))
    
    for flac_path in tqdm(flac_files, desc="Checking durations"):
        try:
            duration = librosa.get_duration(path=str(flac_path))
            
            if min_duration <= duration <= max_duration:
                # Extract speaker and chapter from path
                parts = flac_path.parts
                speaker_id = parts[-3]
                chapter_id = parts[-2]
                
                audio_files.append((flac_path, duration, speaker_id, chapter_id))
        except Exception as e:
            print(f"Error processing {flac_path}: {e}")
            continue
    
    print(f"Found {len(audio_files)} files matching duration criteria")
    return audio_files


def process_and_save_samples(
    audio_files: list,
    output_dir: Path,
    num_samples: int = 100,
    target_sr: int = 16000
) -> pd.DataFrame:
    """
    Process audio files and save as WAV samples.
    
    Args:
        audio_files: List of (file_path, duration, speaker_id, chapter_id)
        output_dir: Directory to save processed samples
        num_samples: Number of samples to process
        target_sr: Target sample rate
    
    Returns:
        DataFrame with sample metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select samples (try to diversify speakers)
    selected_files = []
    seen_speakers = set()
    
    # First pass: one sample per speaker
    for file_info in audio_files:
        speaker_id = file_info[2]
        if speaker_id not in seen_speakers:
            selected_files.append(file_info)
            seen_speakers.add(speaker_id)
        if len(selected_files) >= num_samples:
            break
    
    # Second pass: fill remaining slots
    if len(selected_files) < num_samples:
        for file_info in audio_files:
            if file_info not in selected_files:
                selected_files.append(file_info)
            if len(selected_files) >= num_samples:
                break
    
    metadata = []
    
    print(f"\nProcessing {len(selected_files)} samples...")
    
    for idx, (flac_path, duration, speaker_id, chapter_id) in enumerate(tqdm(selected_files, desc="Processing")):
        sample_id = f"{idx + 1:03d}"
        output_path = output_dir / f"sample_{sample_id}.wav"
        
        try:
            # Load and convert to target sample rate
            audio, sr = librosa.load(str(flac_path), sr=target_sr)
            
            # Save as WAV
            sf.write(output_path, audio, target_sr)
            
            # Record metadata
            metadata.append({
                'sample_id': sample_id,
                'filename': output_path.name,
                'duration': duration,
                'sample_rate': target_sr,
                'speaker_id': speaker_id,
                'chapter_id': chapter_id,
                'source_path': str(flac_path.relative_to(flac_path.parents[3]))
            })
        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            continue
    
    return pd.DataFrame(metadata)


def main():
    parser = argparse.ArgumentParser(description="Download LibriSpeech samples for Qwen2.5-Omni analysis")
    parser.add_argument('--output_dir', type=str, default='../data',
                        help='Output directory for samples and metadata')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to download')
    parser.add_argument('--min_duration', type=float, default=5.0,
                        help='Minimum audio duration in seconds')
    parser.add_argument('--max_duration', type=float, default=15.0,
                        help='Maximum audio duration in seconds')
    parser.add_argument('--split', type=str, default='dev-clean',
                        choices=['dev-clean', 'test-clean'],
                        help='LibriSpeech split to use')
    parser.add_argument('--keep_archive', action='store_true',
                        help='Keep downloaded tar.gz archive')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = (script_dir / args.output_dir).resolve()
    download_dir = output_dir / 'downloads'
    samples_dir = output_dir / 'audio_samples'
    
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("LibriSpeech Sample Downloader for Qwen2.5-Omni")
    print("=" * 80)
    print(f"Split: {args.split}")
    print(f"Duration range: {args.min_duration}-{args.max_duration}s")
    print(f"Target samples: {args.num_samples}")
    print(f"Output directory: {samples_dir}")
    print("=" * 80)
    
    # Download and extract LibriSpeech
    librispeech_dir = download_and_extract_librispeech(args.split, download_dir)
    
    # Find suitable audio files
    audio_files = find_audio_files(librispeech_dir, args.min_duration, args.max_duration)
    
    if len(audio_files) == 0:
        print("Error: No audio files found matching criteria")
        return
    
    if len(audio_files) < args.num_samples:
        print(f"Warning: Only found {len(audio_files)} files, requested {args.num_samples}")
    
    # Process and save samples
    metadata_df = process_and_save_samples(audio_files, samples_dir, args.num_samples)
    
    # Save metadata
    metadata_path = output_dir / 'metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\nMetadata saved to: {metadata_path}")
    
    # Cleanup
    if not args.keep_archive:
        tar_path = download_dir / f"{args.split}.tar.gz"
        if tar_path.exists():
            print("Removing tar.gz archive...")
            tar_path.unlink()
    
    # Print summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Total samples processed: {len(metadata_df)}")
    print(f"Duration range: {metadata_df['duration'].min():.2f}s - {metadata_df['duration'].max():.2f}s")
    print(f"Mean duration: {metadata_df['duration'].mean():.2f}s")
    print(f"Unique speakers: {metadata_df['speaker_id'].nunique()}")
    print(f"Samples directory: {samples_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
