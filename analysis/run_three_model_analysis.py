#!/usr/bin/env python3
"""
Run combined analysis for GPT-4.1, Claude Opus 4, and Gemini 2.0
using existing perturbation results.
"""

import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import pandas as pd
import numpy as np
import os
import sys
sys.path.append('analysis')

from combine_model_confidence_analysis import ModelConfidenceAnalyzer

def load_gpt41_data():
    """Load GPT-4.1 data from combined_results.xlsx"""
    print("Loading GPT-4.1 data from results/combined_results.xlsx...")

    # Load the combined results file
    df = pd.read_excel('results/combined_results.xlsx')

    # Filter for GPT-4.1 model
    gpt41_df = df[df['Model'] == 'gpt-4.1-2025-04-14'].copy()

    # Rename columns to match expected format
    gpt41_df['Model_Name'] = 'GPT-4.1'
    gpt41_df['Confidence'] = gpt41_df['Confidence Value']

    print(f"  Loaded {len(gpt41_df)} GPT-4.1 records")

    # Save to a separate file for consistency with other models
    output_file = 'results/gpt41_perturbation_results.xlsx'
    gpt41_df.to_excel(output_file, index=False)
    print(f"  Saved GPT-4.1 data to {output_file}")

    return gpt41_df

def main():
    """Run the combined analysis for all three models."""

    print("\n" + "="*70)
    print("THREE-MODEL VERBALIZED CONFIDENCE ANALYSIS")
    print("GPT-4.1, Claude Opus 4, and Gemini 2.0")
    print("="*70)

    # First, extract and prepare GPT-4.1 data
    gpt41_df = load_gpt41_data()

    # Create analyzer with all three models
    print("\nInitializing analyzer with three models...")
    analyzer = ModelConfidenceAnalyzer()

    # Update models to include all three
    analyzer.models = {
        'GPT-4.1': 'gpt41_perturbation_results.xlsx',
        'Claude Opus 4': 'claude_opus_batch_perturbation_results.xlsx',
        'Gemini 2.0': 'gemini_perturbation_results.xlsx'
    }

    # Load data for all models
    print("\nLoading perturbation data for all models...")
    data = analyzer.load_data()

    if len(data) < 3:
        print(f"\nWarning: Only {len(data)} models loaded successfully.")
        print("Proceeding with available data...")

    # Combine confidence scores
    print("\nCombining confidence scores across models...")
    combined_df = analyzer.combine_confidence_scores()
    print(f"  Combined {len(combined_df)} unique perturbations")

    # Generate and display summary
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    analyzer.print_summary()

    # Save all results
    print("\nSaving results to results/combined_analysis/")
    analyzer.save_results()

    # Print final summary
    print("\n" + "="*70)
    print("FILES GENERATED")
    print("="*70)

    output_files = [
        ('Combined Scores', 'results/combined_analysis/combined_confidence_scores.csv'),
        ('Summary Statistics', 'results/combined_analysis/summary_statistics.csv'),
        ('Per-Prompt Stats', 'results/combined_analysis/per_prompt_statistics.csv'),
        ('LaTeX Tables', 'results/combined_analysis/confidence_tables.tex'),
        ('Visualization', 'results/combined_analysis/confidence_comparison.png')
    ]

    for desc, path in output_files:
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 1024*1024:
                size_str = f"{size/1024/1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} bytes"
            print(f"  [OK] {desc}: {path} ({size_str})")
        else:
            print(f"  [X] {desc}: {path} (not found)")

    # Load and display key statistics
    print("\n" + "="*70)
    print("KEY STATISTICS COMPARISON")
    print("="*70)

    summary_df = pd.read_csv('results/combined_analysis/summary_statistics.csv')
    model_stats = summary_df[summary_df['Model'] != 'Overall (All Models)']

    print("\nMean Confidence Scores:")
    for _, row in model_stats.iterrows():
        print(f"  {row['Model']}: {row['Mean']:.2f} +/- {row['Std Dev']:.2f}")

    print("\nHigh Confidence Responses (67-100):")
    for _, row in model_stats.iterrows():
        print(f"  {row['Model']}: {row['% High (67-100)']:.1f}%")

    print("\nMedian Confidence:")
    for _, row in model_stats.iterrows():
        print(f"  {row['Model']}: {row['Median']:.1f}")

    # Calculate and display correlations between models
    print("\n" + "="*70)
    print("MODEL CORRELATIONS")
    print("="*70)

    if len(combined_df) > 0:
        confidence_cols = [col for col in combined_df.columns if '_Confidence' in col and 'Mean' not in col and 'Std' not in col and 'Max' not in col and 'Min' not in col and 'Range' not in col]

        if len(confidence_cols) >= 2:
            corr_matrix = combined_df[confidence_cols].corr()

            print("\nPairwise Correlations (Pearson r):")
            models = ['GPT-4.1', 'Claude Opus 4', 'Gemini 2.0']
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i < j:
                        col1 = f'{model1}_Confidence'
                        col2 = f'{model2}_Confidence'
                        if col1 in corr_matrix.columns and col2 in corr_matrix.columns:
                            r = corr_matrix.loc[col1, col2]
                            if not pd.isna(r):
                                print(f"  {model1} vs {model2}: r = {r:.3f}")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nAll results saved to results/combined_analysis/")
    print("LaTeX tables ready for inclusion in your paper.")

    return 0

if __name__ == "__main__":
    sys.exit(main())