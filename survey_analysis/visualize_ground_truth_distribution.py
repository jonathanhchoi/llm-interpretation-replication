"""
Visualize the random distribution of ground truth values from human survey data.
This demonstrates what the random baseline distribution (N(0.619, 0.167)) looks like.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def load_human_survey_data():
    """Load and combine human survey data from both parts."""
    # Use a dictionary to collect all responses for each question
    question_responses = {}

    # Load part 1 survey results (skip the first header row)
    part1_survey_path = os.path.join(DATA_DIR, 'word_meaning_survey_results.csv')
    if os.path.exists(part1_survey_path):
        survey_df_part1 = pd.read_csv(part1_survey_path, skiprows=1)

        # Process all question columns from part 1
        for col in survey_df_part1.columns:
            if 'Left = No, Right = Yes' in col:
                # Extract question from column name
                parts = col.split(' - ')
                if len(parts) >= 2:
                    question = parts[-1].strip()
                    if question.endswith('?'):
                        # Get numeric values only (skip metadata rows)
                        values = pd.to_numeric(survey_df_part1[col], errors='coerce')
                        valid_values = values.dropna()
                        if len(valid_values) > 0:
                            # Convert 0-100 scale to 0-1 scale and store
                            if question not in question_responses:
                                question_responses[question] = []
                            question_responses[question].extend((valid_values / 100.0).tolist())

    # Load part 2 survey results (skip the first header row)
    part2_survey_path = os.path.join(DATA_DIR, 'word_meaning_survey_results_part_2.csv')
    if os.path.exists(part2_survey_path):
        survey_df_part2 = pd.read_csv(part2_survey_path, skiprows=1)

        # Process all question columns from part 2
        for col in survey_df_part2.columns:
            if 'Left = No, Right = Yes' in col:
                # Extract question from column name
                parts = col.split(' - ')
                if len(parts) >= 2:
                    question = parts[-1].strip()
                    if question.endswith('?'):
                        # Get numeric values only (skip metadata rows)
                        values = pd.to_numeric(survey_df_part2[col], errors='coerce')
                        valid_values = values.dropna()
                        if len(valid_values) > 0:
                            # Convert 0-100 scale to 0-1 scale and store
                            if question not in question_responses:
                                question_responses[question] = []
                            question_responses[question].extend((valid_values / 100.0).tolist())

    # Calculate means for each question
    all_human_values = []
    for question, responses in question_responses.items():
        if responses:
            all_human_values.append(np.mean(responses))

    return np.array(all_human_values)

def create_ground_truth_visualization(human_values, save_path='ground_truth_distribution.png'):
    """Create visualization showing the distribution of human ground truth values."""

    # Calculate statistics
    mean_val = np.mean(human_values)
    std_val = np.std(human_values)

    # Convert to percentage scale for display
    human_values_pct = human_values * 100
    mean_pct = mean_val * 100
    std_pct = std_val * 100

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Histogram of actual human values
    n, bins, patches = ax1.hist(human_values_pct, bins=30, density=True,
                                alpha=0.7, color='#2ca02c', edgecolor='black',
                                label='Actual Human Responses')

    # Overlay fitted normal distribution
    x = np.linspace(0, 100, 200)
    fitted_normal = stats.norm.pdf(x, mean_pct, std_pct)
    ax1.plot(x, fitted_normal, 'r-', linewidth=2,
            label=f'Fitted Normal\nN({mean_pct:.1f}, {std_pct:.1f})')

    # Add vertical lines for mean and std deviations
    ax1.axvline(mean_pct, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
               label=f'Mean: {mean_pct:.1f}%')
    ax1.axvline(mean_pct - std_pct, color='orange', linestyle=':', linewidth=1.5, alpha=0.6)
    ax1.axvline(mean_pct + std_pct, color='orange', linestyle=':', linewidth=1.5, alpha=0.6,
               label=f'±1 SD: {std_pct:.1f}%')

    ax1.set_xlabel('Percentage "Yes" Responses (%)', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Distribution of Human Ground Truth Values', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 100)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Right panel: Random baseline samples vs actual distribution
    np.random.seed(42)  # For reproducibility
    random_samples = np.random.normal(mean_pct, std_pct, 10000)
    random_samples = np.clip(random_samples, 0, 100)  # Clip to valid range

    # Plot both distributions for comparison
    ax2.hist(human_values_pct, bins=30, density=True, alpha=0.5,
            color='#2ca02c', edgecolor='black', label='Actual Human Data')
    ax2.hist(random_samples, bins=30, density=True, alpha=0.5,
            color='#17becf', edgecolor='black', label='Random Baseline\n(Sampled)')

    # Add theoretical normal curve
    theoretical_normal = stats.norm.pdf(x, mean_pct, std_pct)
    ax2.plot(x, theoretical_normal, 'r-', linewidth=2, alpha=0.8,
            label=f'Theoretical N({mean_pct:.1f}, {std_pct:.1f})')

    ax2.axvline(mean_pct, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Percentage "Yes" Responses (%)', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Random Baseline Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # Add overall title
    plt.suptitle('Ground Truth Distribution Analysis for Random Baseline',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return mean_val, std_val

def create_simplified_visualization(human_values, save_path='ground_truth_distribution_simple.png'):
    """Create a simplified single-panel visualization matching paper style."""

    # Calculate statistics
    mean_val = np.mean(human_values)
    std_val = np.std(human_values)

    # Convert to percentage scale for display
    human_values_pct = human_values * 100
    mean_pct = mean_val * 100
    std_pct = std_val * 100

    # Create single panel figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram of actual human values
    n, bins, patches = ax.hist(human_values_pct, bins=30, density=True,
                               alpha=0.7, color='#1f77b4', edgecolor='black')

    # Create smoothed empirical distribution using LOESS
    # Get histogram centers and heights for smoothing
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Apply LOESS smoothing to the histogram data
    # Use a fraction of 0.3 for the smoothing (adjust as needed)
    smoothed = lowess(n, bin_centers, frac=0.3, return_sorted=True)

    # Plot the smoothed curve
    ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2.5,
           label='Smoothed empirical distribution')

    # Add vertical lines for mean
    ax.axvline(mean_pct, color='red', linestyle='--', linewidth=2, alpha=0.8,
              label=f'Mean = {mean_pct:.1f}%')

    ax.set_xlabel('Percentage of "Yes" Responses (%)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    # Remove title
    ax.set_xlim(0, 100)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return mean_val, std_val

def main():
    """Main execution function."""
    print("="*60)
    print("GROUND TRUTH DISTRIBUTION VISUALIZATION")
    print("="*60)

    # Load human survey data
    print("\nLoading human survey data...")
    human_values = load_human_survey_data()

    if len(human_values) == 0:
        print("ERROR: No human survey data found!")
        return

    print(f"Loaded {len(human_values)} human ground truth values")

    # Calculate and display statistics
    mean_val = np.mean(human_values)
    std_val = np.std(human_values)
    print(f"\nGround Truth Statistics:")
    print(f"  Mean: {mean_val:.3f} ({mean_val*100:.1f}%)")
    print(f"  Std:  {std_val:.3f} ({std_val*100:.1f}%)")
    print(f"  Min:  {np.min(human_values):.3f} ({np.min(human_values)*100:.1f}%)")
    print(f"  Max:  {np.max(human_values):.3f} ({np.max(human_values)*100:.1f}%)")

    # Create visualizations
    print("\nCreating visualizations...")

    # Create detailed two-panel visualization
    mean1, std1 = create_ground_truth_visualization(human_values,
                                                    'ground_truth_distribution.png')
    print("  Saved detailed visualization to ground_truth_distribution.png")

    # Create simplified single-panel visualization (better for paper)
    mean2, std2 = create_simplified_visualization(human_values,
                                                  'ground_truth_distribution_simple.png')
    print("  Saved simplified visualization to ground_truth_distribution_simple.png")

    # Save statistics to JSON for reference
    stats_dict = {
        'n_questions': len(human_values),
        'mean': float(mean_val),
        'std': float(std_val),
        'mean_pct': float(mean_val * 100),
        'std_pct': float(std_val * 100),
        'min': float(np.min(human_values)),
        'max': float(np.max(human_values)),
        'median': float(np.median(human_values)),
        'description': 'Human ground truth distribution statistics for random baseline'
    }

    with open('ground_truth_statistics.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print("\n  Saved statistics to ground_truth_statistics.json")

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print("\nThe random baseline is drawn from:")
    print(f"  N({mean_val*100:.1f}, {std_val*100:.1f}) in percentage scale")
    print(f"  N({mean_val:.3f}, {std_val:.3f}) in 0-1 scale")
    print("\nThis distribution represents the empirical distribution of human")
    print("ground truth values across all survey questions.")

if __name__ == "__main__":
    main()