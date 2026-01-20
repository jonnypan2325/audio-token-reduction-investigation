"""
Compare MMAU Evaluation: Baseline vs. FastV Pruning
Runs evaluation twice (with and without pruning) and saves results for analysis.
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse


def run_evaluation(
    model_path: str,
    data_file: str,
    output_file: str,
    batch_size: int = 1,
    use_pruning: bool = False,
    pruning_ratio: float = 0.5,
    aggregation_layer: int = 15,
) -> dict:
    """
    Run a single evaluation with specified configuration.
    
    Returns:
        Dictionary with evaluation results metadata
    """
    # Build command
    script_path = Path(__file__).parent.parent / "MMAU" / "evalMMAU_wenjun.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--model_path", model_path,
        "--data_file", data_file,
        "--output_file", output_file,
        "--batch_size", str(batch_size),
    ]
    
    if use_pruning:
        cmd.extend([
            "--use_pruning",
            "--pruning_ratio", str(pruning_ratio),
            "--aggregation_layer", str(aggregation_layer),
        ])
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    # Run evaluation
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        # Load results to extract metadata
        with open(output_file, 'r') as f:
            eval_results = json.load(f)
        
        return eval_results['metadata']
    
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def compare_pruning(
    model_path: str = "Qwen/Qwen2.5-Omni-7B",
    data_file: str = "mmau-test-mini.json",
    batch_size: int = 1,
    pruning_ratio: float = 0.5,
    aggregation_layer: int = 15,
    output_dir: str = "pruning_comparison_results",
):
    """
    Run comparison between baseline and pruned models.
    
    Args:
        model_path: Path to model or HuggingFace model ID
        data_file: Path to MMAU test data
        batch_size: Batch size for inference
        pruning_ratio: Fraction of tokens to prune
        aggregation_layer: Layer to start pruning
        output_dir: Directory to save results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Timestamp for this comparison run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full path to data file (relative to MMAU directory)
    mmau_dir = Path(__file__).parent.parent / "MMAU"
    data_file_path = str(mmau_dir / data_file)
    
    print(f"\n{'#'*80}")
    print(f"# MMAU Pruning Comparison")
    print(f"# Timestamp: {timestamp}")
    print(f"# Model: {model_path}")
    print(f"# Data: {data_file}")
    print(f"# Pruning: ratio={pruning_ratio}, layer={aggregation_layer}")
    print(f"{'#'*80}\n")
    
    results_summary = {
        "timestamp": timestamp,
        "model_path": model_path,
        "data_file": data_file,
        "batch_size": batch_size,
        "pruning_config": {
            "ratio": pruning_ratio,
            "aggregation_layer": aggregation_layer,
        },
        "results": {}
    }
    
    # Run baseline evaluation
    print(f"\n[1/2] Running BASELINE evaluation (no pruning)...")
    baseline_output = output_path / f"mmau_baseline_{timestamp}.json"
    baseline_metadata = run_evaluation(
        model_path=model_path,
        data_file=data_file_path,
        output_file=str(baseline_output),
        batch_size=batch_size,
        use_pruning=False,
    )
    
    if baseline_metadata:
        print(f"✓ Baseline completed: {baseline_metadata['accuracy']:.4f} accuracy")
        results_summary["results"]["baseline"] = {
            "output_file": str(baseline_output),
            "accuracy": baseline_metadata["accuracy"],
            "correct": baseline_metadata["correct_predictions"],
            "total": baseline_metadata["processed_samples"],
        }
    else:
        print("✗ Baseline evaluation failed!")
        return
    
    # Run pruned evaluation
    print(f"\n[2/2] Running PRUNED evaluation (ratio={pruning_ratio})...")
    pruned_output = output_path / f"mmau_pruned_{pruning_ratio}_{timestamp}.json"
    pruned_metadata = run_evaluation(
        model_path=model_path,
        data_file=data_file_path,
        output_file=str(pruned_output),
        batch_size=batch_size,
        use_pruning=True,
        pruning_ratio=pruning_ratio,
        aggregation_layer=aggregation_layer,
    )
    
    if pruned_metadata:
        print(f"✓ Pruned completed: {pruned_metadata['accuracy']:.4f} accuracy")
        results_summary["results"]["pruned"] = {
            "output_file": str(pruned_output),
            "accuracy": pruned_metadata["accuracy"],
            "correct": pruned_metadata["correct_predictions"],
            "total": pruned_metadata["processed_samples"],
        }
    else:
        print("✗ Pruned evaluation failed!")
        return
    
    # Calculate comparison metrics
    baseline_acc = baseline_metadata["accuracy"]
    pruned_acc = pruned_metadata["accuracy"]
    accuracy_drop = baseline_acc - pruned_acc
    accuracy_drop_pct = (accuracy_drop / baseline_acc * 100) if baseline_acc > 0 else 0
    
    results_summary["comparison"] = {
        "baseline_accuracy": baseline_acc,
        "pruned_accuracy": pruned_acc,
        "accuracy_drop": accuracy_drop,
        "accuracy_drop_percentage": accuracy_drop_pct,
        "tokens_pruned_percentage": pruning_ratio * 100,
    }
    
    # Save summary
    summary_file = output_path / f"comparison_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline Accuracy:  {baseline_acc:.4f} ({baseline_metadata['correct_predictions']}/{baseline_metadata['processed_samples']})")
    print(f"Pruned Accuracy:    {pruned_acc:.4f} ({pruned_metadata['correct_predictions']}/{pruned_metadata['processed_samples']})")
    print(f"Accuracy Drop:      {accuracy_drop:.4f} ({accuracy_drop_pct:.2f}%)")
    print(f"Tokens Pruned:      {pruning_ratio * 100:.0f}%")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_path}")
    print(f"  - Baseline: {baseline_output.name}")
    print(f"  - Pruned:   {pruned_output.name}")
    print(f"  - Summary:  {summary_file.name}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare MMAU evaluation with and without FastV-style pruning"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-Omni-7B",
        help="Model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="mmau-test-mini.json",
        help="MMAU test data file (relative to MMAU directory)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.5,
        help="Fraction of audio tokens to prune (0.0-1.0)"
    )
    parser.add_argument(
        "--aggregation_layer",
        type=int,
        default=15,
        help="Layer index to start pruning (0-27 for 28-layer model)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pruning_comparison_results",
        help="Directory to save comparison results"
    )
    
    args = parser.parse_args()
    
    compare_pruning(
        model_path=args.model_path,
        data_file=args.data_file,
        batch_size=args.batch_size,
        pruning_ratio=args.pruning_ratio,
        aggregation_layer=args.aggregation_layer,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
