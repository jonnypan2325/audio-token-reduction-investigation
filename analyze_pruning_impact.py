"""
Analyze Pruning Impact
Compares baseline and pruned MMAU evaluation results to assess accuracy vs. efficiency trade-offs.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from collections import defaultdict


def load_results(file_path: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_per_sample(baseline_results: Dict, pruned_results: Dict) -> pd.DataFrame:
    """
    Analyze per-sample differences between baseline and pruned.
    
    Returns:
        DataFrame with per-sample comparison
    """
    baseline_samples = {s['id']: s for s in baseline_results['results']}
    pruned_samples = {s['id']: s for s in pruned_results['results']}
    
    comparisons = []
    
    for sample_id in baseline_samples:
        if sample_id not in pruned_samples:
            continue
        
        baseline_sample = baseline_samples[sample_id]
        pruned_sample = pruned_samples[sample_id]
        
        # Check if predictions changed
        baseline_correct = baseline_sample.get('is_correct', False)
        pruned_correct = pruned_sample.get('is_correct', False)
        
        # Categorize the change
        if baseline_correct and pruned_correct:
            status = "both_correct"
        elif not baseline_correct and not pruned_correct:
            status = "both_incorrect"
        elif baseline_correct and not pruned_correct:
            status = "degraded"  # Pruning caused error
        else:
            status = "improved"  # Pruning fixed error (rare)
        
        comparisons.append({
            'id': sample_id,
            'question': baseline_sample.get('question', '')[:100],
            'baseline_correct': baseline_correct,
            'pruned_correct': pruned_correct,
            'status': status,
            'baseline_prediction': baseline_sample.get('model_prediction', ''),
            'pruned_prediction': pruned_sample.get('model_prediction', ''),
            'ground_truth': baseline_sample.get('answer', ''),
        })
    
    return pd.DataFrame(comparisons)


def analyze_by_category(df: pd.DataFrame, baseline_results: Dict) -> pd.DataFrame:
    """
    Analyze accuracy by category/task type if available.
    
    Returns:
        DataFrame with category-level statistics
    """
    # Try to extract categories from the baseline results
    category_stats = defaultdict(lambda: {'total': 0, 'baseline_correct': 0, 'pruned_correct': 0})
    
    baseline_samples = {s['id']: s for s in baseline_results['results']}
    
    for _, row in df.iterrows():
        sample_id = row['id']
        sample = baseline_samples.get(sample_id, {})
        
        # Try to extract category (MMAU has task types like 'speech', 'music', 'sound')
        category = sample.get('task', sample.get('category', 'unknown'))
        
        category_stats[category]['total'] += 1
        if row['baseline_correct']:
            category_stats[category]['baseline_correct'] += 1
        if row['pruned_correct']:
            category_stats[category]['pruned_correct'] += 1
    
    # Convert to DataFrame
    category_data = []
    for category, stats in category_stats.items():
        baseline_acc = stats['baseline_correct'] / stats['total'] if stats['total'] > 0 else 0
        pruned_acc = stats['pruned_correct'] / stats['total'] if stats['total'] > 0 else 0
        acc_drop = baseline_acc - pruned_acc
        
        category_data.append({
            'category': category,
            'total_samples': stats['total'],
            'baseline_accuracy': baseline_acc,
            'pruned_accuracy': pruned_acc,
            'accuracy_drop': acc_drop,
            'accuracy_drop_pct': (acc_drop / baseline_acc * 100) if baseline_acc > 0 else 0,
        })
    
    return pd.DataFrame(category_data).sort_values('total_samples', ascending=False)


def print_analysis(
    baseline_results: Dict,
    pruned_results: Dict,
    output_dir: str = None,
    save_csv: bool = True,
):
    """
    Print comprehensive analysis of pruning impact.
    
    Args:
        baseline_results: Baseline evaluation results
        pruned_results: Pruned evaluation results
        output_dir: Directory to save CSV files
        save_csv: Whether to save detailed CSV files
    """
    baseline_meta = baseline_results['metadata']
    pruned_meta = pruned_results['metadata']
    
    # Overall statistics
    print(f"\n{'='*80}")
    print(f"PRUNING IMPACT ANALYSIS")
    print(f"{'='*80}\n")
    
    print(f"Model: {baseline_meta['model_path']}")
    print(f"Total Samples: {baseline_meta['total_samples']}")
    print(f"Pruning Configuration:")
    print(f"  - Ratio: {pruned_meta.get('pruning_ratio', 0.5):.1%}")
    print(f"  - Aggregation Layer: {pruned_meta.get('aggregation_layer', 15)}")
    print()
    
    # Accuracy comparison
    baseline_acc = baseline_meta['accuracy']
    pruned_acc = pruned_meta['accuracy']
    acc_drop = baseline_acc - pruned_acc
    acc_drop_pct = (acc_drop / baseline_acc * 100) if baseline_acc > 0 else 0
    
    print(f"{'Metric':<30} {'Baseline':<15} {'Pruned':<15} {'Change':<15}")
    print(f"{'-'*80}")
    print(f"{'Accuracy':<30} {baseline_acc:<15.4f} {pruned_acc:<15.4f} {acc_drop:+.4f} ({acc_drop_pct:+.2f}%)")
    print(f"{'Correct Predictions':<30} {baseline_meta['correct_predictions']:<15} {pruned_meta['correct_predictions']:<15} {pruned_meta['correct_predictions'] - baseline_meta['correct_predictions']:+}")
    print()
    
    # Per-sample analysis
    df_samples = analyze_per_sample(baseline_results, pruned_results)
    
    print(f"Per-Sample Analysis:")
    print(f"{'-'*80}")
    status_counts = df_samples['status'].value_counts()
    for status, count in status_counts.items():
        pct = count / len(df_samples) * 100
        print(f"  {status:<20} {count:>5} ({pct:>5.1f}%)")
    print()
    
    # Category analysis (if available)
    df_categories = analyze_by_category(df_samples, baseline_results)
    
    if len(df_categories) > 1:  # If we have actual categories (not just 'unknown')
        print(f"Category-Level Analysis:")
        print(f"{'-'*80}")
        print(df_categories.to_string(index=False))
        print()
    
    # Degraded samples (errors caused by pruning)
    degraded_samples = df_samples[df_samples['status'] == 'degraded']
    
    if len(degraded_samples) > 0:
        print(f"Degraded Samples (Correct → Incorrect due to Pruning): {len(degraded_samples)}")
        print(f"{'-'*80}")
        for idx, row in degraded_samples.head(10).iterrows():
            print(f"\nSample ID: {row['id']}")
            print(f"  Question: {row['question']}")
            print(f"  Ground Truth: {row['ground_truth']}")
            print(f"  Baseline Prediction: {row['baseline_prediction']}")
            print(f"  Pruned Prediction: {row['pruned_prediction']}")
        
        if len(degraded_samples) > 10:
            print(f"\n  ... and {len(degraded_samples) - 10} more")
        print()
    
    # Improved samples (rare but interesting)
    improved_samples = df_samples[df_samples['status'] == 'improved']
    
    if len(improved_samples) > 0:
        print(f"Improved Samples (Incorrect → Correct due to Pruning): {len(improved_samples)}")
        print(f"{'-'*80}")
        for idx, row in improved_samples.head(5).iterrows():
            print(f"\nSample ID: {row['id']}")
            print(f"  Question: {row['question']}")
            print(f"  Ground Truth: {row['ground_truth']}")
            print(f"  Baseline Prediction: {row['baseline_prediction']}")
            print(f"  Pruned Prediction: {row['pruned_prediction']}")
        print()
    
    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Pruning {pruned_meta.get('pruning_ratio', 0.5):.0%} of audio tokens at layer {pruned_meta.get('aggregation_layer', 15)} resulted in:")
    print(f"  - Accuracy: {baseline_acc:.4f} → {pruned_acc:.4f} ({acc_drop_pct:+.2f}%)")
    print(f"  - Token Reduction: {pruned_meta.get('pruning_ratio', 0.5):.0%}")
    print(f"  - Trade-off: {acc_drop_pct:.2f}% accuracy drop for {pruned_meta.get('pruning_ratio', 0.5):.0%} token reduction")
    print(f"{'='*80}\n")
    
    # Save CSV files
    if save_csv and output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save per-sample comparison
        samples_csv = output_path / "per_sample_comparison.csv"
        df_samples.to_csv(samples_csv, index=False)
        print(f"✓ Saved per-sample comparison to: {samples_csv}")
        
        # Save category analysis
        if len(df_categories) > 1:
            categories_csv = output_path / "category_analysis.csv"
            df_categories.to_csv(categories_csv, index=False)
            print(f"✓ Saved category analysis to: {categories_csv}")
        
        # Save degraded samples
        if len(degraded_samples) > 0:
            degraded_csv = output_path / "degraded_samples.csv"
            degraded_samples.to_csv(degraded_csv, index=False)
            print(f"✓ Saved degraded samples to: {degraded_csv}")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pruning impact on MMAU evaluation results"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Path to baseline evaluation results JSON"
    )
    parser.add_argument(
        "--pruned",
        type=str,
        required=True,
        help="Path to pruned evaluation results JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pruning_analysis",
        help="Directory to save analysis CSV files"
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Don't save CSV files, only print to console"
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading baseline results from: {args.baseline}")
    baseline_results = load_results(args.baseline)
    
    print(f"Loading pruned results from: {args.pruned}")
    pruned_results = load_results(args.pruned)
    
    # Run analysis
    print_analysis(
        baseline_results=baseline_results,
        pruned_results=pruned_results,
        output_dir=args.output_dir,
        save_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()
