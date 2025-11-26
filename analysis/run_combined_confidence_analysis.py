#!/usr/bin/env python3
"""
Runner script for combined confidence analysis across GPT-4, Claude, and Gemini.

This script:
1. Checks for existing perturbation results for all models
2. Optionally runs GPT-4 perturbation analysis if needed
3. Combines and analyzes verbalized confidence scores
4. Generates comprehensive tables and visualizations
"""

import os
import sys
import warnings
import argparse
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add analysis directory to path
sys.path.append('analysis')

def check_data_availability():
    """Check which model data files are available."""
    results_dir = Path('results')

    available = {
        'GPT-4': results_dir / 'gpt4_perturbation_results.xlsx',
        'Claude Opus 4': results_dir / 'claude_opus_batch_perturbation_results.xlsx',
        'Gemini 2.0': results_dir / 'gemini_perturbation_results.xlsx'
    }

    status = {}
    for model, path in available.items():
        status[model] = path.exists()

    return status, available

def run_gpt4_analysis():
    """Run GPT-4 perturbation analysis if needed."""
    from perturb_prompts_gpt import main as run_gpt_analysis

    print("\n" + "="*60)
    print("Running GPT-4 Perturbation Analysis")
    print("="*60)

    try:
        df = run_gpt_analysis()
        print(f"[OK] GPT-4 analysis complete: {len(df)} perturbations processed")
        return True
    except Exception as e:
        print(f"[ERROR] Error running GPT-4 analysis: {str(e)}")
        print("  Note: You may need to set OPENAI_API_KEY in your .env file")
        return False

def run_combined_analysis(models_to_analyze):
    """Run combined analysis for available models."""
    from combine_model_confidence_analysis import ModelConfidenceAnalyzer

    print("\n" + "="*60)
    print("Running Combined Confidence Analysis")
    print("="*60)

    # Create analyzer
    analyzer = ModelConfidenceAnalyzer()

    # Update models list based on what's available (extract just the filenames)
    analyzer.models = {model: path.name for model, path in models_to_analyze.items()}

    # Run full analysis pipeline
    print(f"\nAnalyzing {len(models_to_analyze)} models:")
    for model, path in models_to_analyze.items():
        print(f"  - {model}: {path}")

    # Load and analyze data
    analyzer.load_data()
    analyzer.combine_confidence_scores()

    # Print summary to console
    analyzer.print_summary()

    # Save all results
    analyzer.save_results()

    return analyzer

def main():
    """Main function to coordinate the analysis."""
    parser = argparse.ArgumentParser(description='Run combined confidence analysis for LLM models')
    parser.add_argument('--skip-gpt4', action='store_true',
                       help='Skip GPT-4 analysis even if data is missing')
    parser.add_argument('--force-gpt4', action='store_true',
                       help='Force re-run GPT-4 analysis even if data exists')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("LLM VERBALIZED CONFIDENCE ANALYSIS")
    print("="*60)

    # Check data availability
    status, available_paths = check_data_availability()

    print("\nData Availability Check:")
    print("-"*40)
    for model, exists in status.items():
        symbol = "[OK]" if exists else "[X]"
        print(f"  {symbol} {model}: {'Available' if exists else 'Not found'}")

    # Determine which models to analyze
    models_to_analyze = {}

    # Handle GPT-4
    if status['GPT-4']:
        if args.force_gpt4:
            print("\n[!] Forcing GPT-4 re-analysis...")
            if run_gpt4_analysis():
                models_to_analyze['GPT-4'] = available_paths['GPT-4']
        else:
            models_to_analyze['GPT-4'] = available_paths['GPT-4']
    else:
        if not args.skip_gpt4:
            print("\n[!] GPT-4 data not found. Running analysis...")
            print("  (Use --skip-gpt4 to skip this step)")
            if run_gpt4_analysis():
                models_to_analyze['GPT-4'] = available_paths['GPT-4']
        else:
            print("\n[!] Skipping GPT-4 analysis as requested")

    # Add other models if available
    if status['Claude Opus 4']:
        models_to_analyze['Claude Opus 4'] = available_paths['Claude Opus 4']
    else:
        print("\n[!] Claude Opus 4 data not found")
        print("  Run perturb_prompts_claude_batch.py to generate it")

    if status['Gemini 2.0']:
        models_to_analyze['Gemini 2.0'] = available_paths['Gemini 2.0']
    else:
        print("\n[!] Gemini 2.0 data not found")
        print("  Run perturb_prompts_gemini_batch.py to generate it")

    # Check if we have enough data to proceed
    if len(models_to_analyze) < 2:
        print("\n[ERROR] Need at least 2 models for comparison analysis")
        print("  Please generate perturbation data for more models")
        return 1

    # Run combined analysis
    analyzer = run_combined_analysis(models_to_analyze)

    # Print final summary
    print("\n" + "="*60)
    print("[COMPLETE] ANALYSIS COMPLETE")
    print("="*60)
    print("\nOutput Files Generated:")
    print("-"*40)

    output_dir = Path('results/combined_analysis')
    if output_dir.exists():
        files = [
            ('Combined Scores', 'combined_confidence_scores.csv'),
            ('Summary Statistics', 'summary_statistics.csv'),
            ('Per-Prompt Stats', 'per_prompt_statistics.csv'),
            ('LaTeX Tables', 'confidence_tables.tex'),
            ('Visualization', 'confidence_comparison.png')
        ]

        for desc, filename in files:
            filepath = output_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                if size > 1024*1024:
                    size_str = f"{size/1024/1024:.1f} MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size} bytes"
                print(f"  [OK] {desc}: {filename} ({size_str})")
            else:
                print(f"  [X] {desc}: {filename} (not found)")

    print("\nKey Insights:")
    print("-"*40)

    # Get summary statistics for insights
    import pandas as pd
    summary_file = output_dir / 'summary_statistics.csv'
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        model_stats = summary_df[summary_df['Model'] != 'Overall (All Models)']

        if not model_stats.empty:
            # Model with highest confidence
            highest_idx = model_stats['Mean'].idxmax()
            highest_model = model_stats.loc[highest_idx, 'Model']
            highest_mean = model_stats.loc[highest_idx, 'Mean']

            # Model with lowest confidence
            lowest_idx = model_stats['Mean'].idxmin()
            lowest_model = model_stats.loc[lowest_idx, 'Model']
            lowest_mean = model_stats.loc[lowest_idx, 'Mean']

            # Model with highest variability
            highest_std_idx = model_stats['Std Dev'].idxmax()
            highest_std_model = model_stats.loc[highest_std_idx, 'Model']
            highest_std = model_stats.loc[highest_std_idx, 'Std Dev']

            print(f"  * Most confident: {highest_model} (mean={highest_mean:.1f})")
            print(f"  * Least confident: {lowest_model} (mean={lowest_mean:.1f})")
            print(f"  * Most variable: {highest_std_model} (std={highest_std:.1f})")

            # Confidence level distributions
            print("\n  Confidence Level Distributions:")
            for _, row in model_stats.iterrows():
                model = row['Model']
                high_conf = row['% High (67-100)']
                print(f"    - {model}: {high_conf:.1f}% high confidence responses")

    print("\nLaTeX Integration:")
    print("-"*40)
    print("  Include in your paper with:")
    print("  \\input{results/combined_analysis/confidence_tables.tex}")

    print("\n[SUCCESS] Analysis complete! Check results/combined_analysis/ for all outputs.")

    return 0

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    sys.exit(main())