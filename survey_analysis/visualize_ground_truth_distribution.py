"""
Visualize the distribution of ground-truth values from human survey data.

The script also reports per-question dispersion in human responses. It uses the
same cleaned survey data used in the consolidated survey analysis and excludes
attention-check questions from all outputs.
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from survey_analysis_consolidated import load_cleaned_question_responses

# Define data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_human_survey_data(return_details=False):
    """Load cleaned human survey data from both survey waves."""
    survey_files = [
        os.path.join(DATA_DIR, 'word_meaning_survey_results.csv'),
        os.path.join(DATA_DIR, 'word_meaning_survey_results_part_2.csv'),
    ]
    question_responses, exclusion_stats = load_cleaned_question_responses(survey_files)

    rows = []
    for question, raw_responses in question_responses.items():
        responses = pd.Series(raw_responses, dtype=float).dropna()
        if responses.empty:
            continue

        response_id = f"Q{len(rows) + 1}"
        rows.append({
            'id': response_id,
            'question': question,
            'n': int(len(responses)),
            'mean': float(responses.mean()),
            'sd': float(responses.std(ddof=1)),
            'median': float(responses.median()),
            'iqr': float(responses.quantile(0.75) - responses.quantile(0.25)),
            'pct_0_25': float((responses <= 25).mean() * 100),
            'pct_26_50': float(((responses > 25) & (responses <= 50)).mean() * 100),
            'pct_51_75': float(((responses > 50) & (responses <= 75)).mean() * 100),
            'pct_76_100': float((responses > 75).mean() * 100),
        })

    distribution_df = pd.DataFrame(rows)
    human_values = distribution_df['mean'].to_numpy() / 100.0

    if return_details:
        return human_values, distribution_df, exclusion_stats
    return human_values

def latex_escape(value):
    """Escape a string for LaTeX table output."""
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    text = ''.join(replacements.get(char, char) for char in str(value))
    return text.replace('“', '``').replace('”', "''").replace('’', "'")

def save_distribution_table(distribution_df, csv_path, tex_path):
    """Save per-question human response distributions as CSV and LaTeX."""
    distribution_df.to_csv(csv_path, index=False)

    lines = [
        r'\begin{scriptsize}',
        r'\setlength{\tabcolsep}{3pt}',
        r'\begin{longtable}{lp{0.30\textwidth}rrrrrrr}',
        r"\caption{Distribution of human survey responses by question. Responses are on a 0--100 scale, where 0 indicates ``No, definitely not'' and 100 indicates ``Yes, definitely.'' The four rightmost columns report the percentage of responses in each range.}",
        r'\label{tab:human_response_distribution}\\',
        r'\hline',
        r'\textbf{ID} & \textbf{Question} & \textbf{N} & \textbf{Mean} & \textbf{SD} & \textbf{0--25} & \textbf{26--50} & \textbf{51--75} & \textbf{76--100} \\',
        r'\hline',
        r'\endfirsthead',
        r'\caption[]{Distribution of human survey responses by question (continued).}\\',
        r'\hline',
        r'\textbf{ID} & \textbf{Question} & \textbf{N} & \textbf{Mean} & \textbf{SD} & \textbf{0--25} & \textbf{26--50} & \textbf{51--75} & \textbf{76--100} \\',
        r'\hline',
        r'\endhead',
    ]

    for _, row in distribution_df.iterrows():
        lines.append(
            f"{latex_escape(row['id'])} & {latex_escape(row['question'])} & "
            f"{int(row['n'])} & {row['mean']:.1f} & {row['sd']:.1f} & "
            f"{row['pct_0_25']:.0f} & {row['pct_26_50']:.0f} & "
            f"{row['pct_51_75']:.0f} & {row['pct_76_100']:.0f} \\\\"
        )

    lines.extend([
        r'\hline',
        r'\end{longtable}',
        r'\end{scriptsize}',
    ])

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

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
    human_values, distribution_df, exclusion_stats = load_human_survey_data(return_details=True)

    if len(human_values) == 0:
        print("ERROR: No human survey data found!")
        return

    print(f"Loaded {len(human_values)} human ground truth values")
    print(f"Final cleaned respondent count: {exclusion_stats['final_count']}")

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
                                                    os.path.join(OUTPUT_DIR, 'ground_truth_distribution.png'))
    print("  Saved detailed visualization to ground_truth_distribution.png")

    # Create simplified single-panel visualization (better for paper)
    mean2, std2 = create_simplified_visualization(human_values,
                                                  os.path.join(OUTPUT_DIR, 'ground_truth_distribution_simple.png'))
    print("  Saved simplified visualization to ground_truth_distribution_simple.png")

    # Save per-question distribution table
    save_distribution_table(
        distribution_df,
        os.path.join(OUTPUT_DIR, 'human_response_distribution.csv'),
        os.path.join(OUTPUT_DIR, 'human_response_distribution_table.tex')
    )
    print("  Saved per-question distribution CSV to human_response_distribution.csv")
    print("  Saved per-question distribution table to human_response_distribution_table.tex")

    high_dispersion = int((distribution_df['sd'] >= 30).sum())
    bimodal_proxy = int((
        (distribution_df['pct_0_25'] >= 25) &
        (distribution_df['pct_76_100'] >= 25) &
        ((distribution_df['pct_26_50'] + distribution_df['pct_51_75']) <= 50)
    ).sum())

    # Save statistics to JSON for reference
    stats_dict = {
        'n_questions': len(human_values),
        'n_cleaned_respondents': int(exclusion_stats['final_count']),
        'mean': float(mean_val),
        'std': float(std_val),
        'mean_pct': float(mean_val * 100),
        'std_pct': float(std_val * 100),
        'min': float(np.min(human_values)),
        'max': float(np.max(human_values)),
        'median': float(np.median(human_values)),
        'mean_question_sd_pct': float(distribution_df['sd'].mean()),
        'median_question_sd_pct': float(distribution_df['sd'].median()),
        'questions_with_sd_at_least_30': high_dispersion,
        'bimodal_proxy_questions': bimodal_proxy,
        'description': 'Human ground-truth distribution used for the Normal baseline'
    }

    statistics_path = os.path.join(OUTPUT_DIR, 'ground_truth_statistics.json')
    with open(statistics_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2)
    print("\n  Saved statistics to ground_truth_statistics.json")

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print("\nThe Normal baseline is parameterized from:")
    print(f"  N({mean_val*100:.1f}, {std_val*100:.1f}) in percentage scale")
    print(f"  N({mean_val:.3f}, {std_val:.3f}) in 0-1 scale")
    print("\nThis distribution represents the empirical distribution of human")
    print("ground truth values across all survey questions.")
    print("\nPer-question response dispersion:")
    print(f"  Mean question-level SD: {distribution_df['sd'].mean():.1f} points")
    print(f"  Median question-level SD: {distribution_df['sd'].median():.1f} points")
    print(f"  Questions with SD >= 30 points: {high_dispersion}")
    print(f"  Questions meeting bimodality proxy: {bimodal_proxy}")

if __name__ == "__main__":
    main()
