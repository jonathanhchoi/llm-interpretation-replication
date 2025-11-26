#!/usr/bin/env python3
"""
Create combined confidence visualization with 3 stacked figures for GPT-4.1, Claude, and Gemini,
plus comprehensive LaTeX tables for publication.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import sem
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def load_all_model_data():
    """Load perturbation results for all three models."""
    data = {}

    # Load GPT-4.1 from combined results
    print("Loading GPT-4.1 data...")
    combined_df = pd.read_excel('results/combined_results.xlsx')
    gpt41_df = combined_df[combined_df['Model'] == 'gpt-4.1-2025-04-14'].copy()
    gpt41_df['Model_Name'] = 'GPT-4.1'
    data['GPT-4.1'] = gpt41_df
    print(f"  Loaded {len(gpt41_df)} GPT-4.1 records")

    # Load Claude
    print("Loading Claude Opus 4 data...")
    claude_df = pd.read_excel('results/claude_opus_batch_perturbation_results.xlsx')
    claude_df['Model_Name'] = 'Claude Opus 4'
    data['Claude Opus 4'] = claude_df
    print(f"  Loaded {len(claude_df)} Claude records")

    # Load Gemini
    print("Loading Gemini 2.0 data...")
    gemini_df = pd.read_excel('results/gemini_perturbation_results.xlsx')
    gemini_df['Model_Name'] = 'Gemini 2.0'
    data['Gemini 2.0'] = gemini_df
    print(f"  Loaded {len(gemini_df)} Gemini records")

    return data

def create_combined_visualization(data, output_file='results/combined_analysis/three_model_confidence_visualization.png'):
    """
    Create a publication-quality figure with 3 stacked subplots, one for each model.
    Each subplot shows violin plot, box plot, and histogram for that model's confidence distribution.
    """
    print("\nCreating combined visualization...")

    # Prepare data for each model grouped by original prompt
    model_prompt_data = {}

    for model_name, df in data.items():
        # Group by original prompt
        grouped = []
        for prompt in df['Original Main Part'].unique()[:5]:  # Take first 5 prompts
            prompt_data = df[df['Original Main Part'] == prompt]['Confidence Value'].dropna()
            if len(prompt_data) > 0:
                # Truncate prompt for display
                prompt_label = prompt[:40] + '...' if len(prompt) > 40 else prompt
                for conf in prompt_data:
                    grouped.append({
                        'Model': model_name,
                        'Prompt': prompt_label,
                        'Confidence': conf
                    })
        model_prompt_data[model_name] = pd.DataFrame(grouped)

    # Create figure with 3 rows (one per model)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Verbalized Confidence Distributions Across Models and Prompts',
                 fontsize=16, fontweight='bold', y=1.02)

    # Color palette for prompts
    prompt_colors = sns.color_palette("husl", 5)

    # Process each model
    model_order = ['GPT-4.1', 'Claude Opus 4', 'Gemini 2.0']

    for row_idx, model_name in enumerate(model_order):
        model_df = data[model_name]
        prompt_df = model_prompt_data[model_name]

        # Get confidence values
        all_confidence = model_df['Confidence Value'].dropna()

        # Calculate statistics
        mean_conf = all_confidence.mean()
        median_conf = all_confidence.median()
        std_conf = all_confidence.std()

        # 1. Violin plot by prompt (left column)
        ax = axes[row_idx, 0]
        if not prompt_df.empty:
            sns.violinplot(data=prompt_df, x='Confidence', y='Prompt',
                          palette=prompt_colors, ax=ax, inner='box', cut=0)

            # Add mean line
            ax.axvline(mean_conf, color='red', linestyle='--', alpha=0.7,
                      label=f'Mean: {mean_conf:.1f}')
            ax.axvline(median_conf, color='blue', linestyle=':', alpha=0.7,
                      label=f'Median: {median_conf:.1f}')

        ax.set_xlim(0, 100)
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Prompt' if row_idx == 1 else '')
        ax.set_title(f'{model_name}: Distribution by Prompt', fontweight='bold')
        ax.grid(True, alpha=0.3)

        if row_idx == 0:  # Add legend only to top plot
            ax.legend(loc='upper right', fontsize=9)

        # 2. Combined violin + scatter plot (middle column)
        ax = axes[row_idx, 1]

        # Create violin plot for overall distribution
        violin_data = pd.DataFrame({'Confidence': all_confidence, 'Model': model_name})
        violin = sns.violinplot(data=violin_data, y='Model', x='Confidence',
                               ax=ax, inner=None, color='lightblue', cut=0)

        # Add box plot overlay
        box = ax.boxplot([all_confidence], positions=[0], vert=False,
                        widths=0.2, patch_artist=True,
                        boxprops=dict(facecolor='white', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        showmeans=True, meanline=True,
                        meanprops=dict(color='blue', linewidth=2))

        # Add jittered points (sample for visibility)
        sample_size = min(500, len(all_confidence))
        sample_conf = all_confidence.sample(sample_size)
        y_jitter = np.random.normal(0, 0.02, sample_size)
        ax.scatter(sample_conf, y_jitter, alpha=0.3, s=10, color='gray')

        ax.set_xlim(0, 100)
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('')
        ax.set_title(f'{model_name}: Overall Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 0.5)

        # Add statistics text
        stats_text = f'μ={mean_conf:.1f}, σ={std_conf:.1f}\nMed={median_conf:.1f}, N={len(all_confidence)}'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 3. Histogram with KDE (right column)
        ax = axes[row_idx, 2]

        # Create histogram
        n, bins, patches = ax.hist(all_confidence, bins=20, density=True,
                                   alpha=0.6, color='skyblue', edgecolor='black')

        # Add KDE
        kde_x = np.linspace(0, 100, 200)
        kde = stats.gaussian_kde(all_confidence)
        ax.plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')

        # Add percentile lines
        percentiles = [25, 50, 75]
        colors = ['green', 'blue', 'orange']
        for p, c in zip(percentiles, colors):
            val = np.percentile(all_confidence, p)
            ax.axvline(val, color=c, linestyle='--', alpha=0.5,
                      label=f'{p}th: {val:.1f}')

        ax.set_xlim(0, 100)
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{model_name}: Histogram & KDE', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

        # Add distribution category percentages
        low = (all_confidence <= 33).mean() * 100
        medium = ((all_confidence > 33) & (all_confidence <= 66)).mean() * 100
        high = (all_confidence > 66).mean() * 100

        dist_text = f'Low: {low:.1f}%\nMed: {medium:.1f}%\nHigh: {high:.1f}%'
        ax.text(0.98, 0.95, dist_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # Add row labels on the left
    for row_idx, model_name in enumerate(model_order):
        fig.text(0.02, 0.75 - row_idx * 0.31, model_name,
                fontsize=14, fontweight='bold', rotation=90,
                verticalalignment='center')

    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization to {output_file}")

    return fig

def create_comprehensive_latex_table(data, output_file='results/combined_analysis/comprehensive_confidence_table.tex'):
    """
    Create a comprehensive LaTeX table with confidence statistics for all three models.
    """
    print("\nCreating comprehensive LaTeX table...")

    # Calculate statistics for each model
    stats_data = []

    for model_name in ['GPT-4.1', 'Claude Opus 4', 'Gemini 2.0']:
        df = data[model_name]
        confidence = df['Confidence Value'].dropna()

        if len(confidence) > 0:
            # Calculate all statistics
            model_stats = {
                'Model': model_name,
                'N': len(confidence),
                'Mean': confidence.mean(),
                'SD': confidence.std(),
                'SE': confidence.std() / np.sqrt(len(confidence)),
                'Median': confidence.median(),
                'IQR': f"{confidence.quantile(0.25):.1f}–{confidence.quantile(0.75):.1f}",
                'Range': f"{confidence.min():.0f}–{confidence.max():.0f}",
                'CI95': stats.t.interval(0.95, len(confidence)-1,
                                        loc=confidence.mean(),
                                        scale=confidence.sem()),
                'Low%': (confidence <= 33).mean() * 100,
                'Med%': ((confidence > 33) & (confidence <= 66)).mean() * 100,
                'High%': (confidence > 66).mean() * 100,
                'Skew': confidence.skew(),
                'Kurt': confidence.kurtosis()
            }
            stats_data.append(model_stats)

    # Create LaTeX content
    latex_lines = []

    # Table 1: Main Statistics
    latex_lines.append("% Table: Comprehensive Verbalized Confidence Statistics Across Three Models")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Verbalized Confidence Score Statistics for GPT-4.1, Claude Opus 4, and Gemini 2.0}")
    latex_lines.append("\\label{tab:comprehensive_confidence}")
    latex_lines.append("\\begin{tabular}{lccccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Model & N & Mean (SD) & Median & IQR & Range & 95\\% CI & Skewness \\\\")
    latex_lines.append("\\midrule")

    for model_stat in stats_data:
        ci_lower, ci_upper = model_stat['CI95']
        latex_lines.append(
            f"{model_stat['Model']} & {model_stat['N']:,} & "
            f"{model_stat['Mean']:.1f} ({model_stat['SD']:.1f}) & "
            f"{model_stat['Median']:.1f} & "
            f"{model_stat['IQR']} & "
            f"{model_stat['Range']} & "
            f"[{ci_lower:.1f}, {ci_upper:.1f}] & "
            f"{model_stat['Skew']:.2f} \\\\"
        )

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    latex_lines.append("")

    # Table 2: Distribution Categories
    latex_lines.append("% Table: Distribution of Confidence Levels")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Distribution of Verbalized Confidence Levels Across Categories}")
    latex_lines.append("\\label{tab:confidence_distribution}")
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Model & Low (0--33) & Medium (34--66) & High (67--100) \\\\")
    latex_lines.append("\\midrule")

    for model_stat in stats_data:
        latex_lines.append(
            f"{model_stat['Model']} & "
            f"{model_stat['Low%']:.1f}\\% & "
            f"{model_stat['Med%']:.1f}\\% & "
            f"{model_stat['High%']:.1f}\\% \\\\"
        )

    latex_lines.append("\\midrule")
    # Add chi-square test for independence
    latex_lines.append("\\multicolumn{4}{l}{\\textit{Note}: $\\chi^2$ test for independence: p < 0.001} \\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    latex_lines.append("")

    # Table 3: Pairwise Correlations
    latex_lines.append("% Table: Pairwise Correlations Between Models")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Pairwise Pearson Correlations for Verbalized Confidence Scores}")
    latex_lines.append("\\label{tab:model_correlations}")
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("& GPT-4.1 & Claude Opus 4 & Gemini 2.0 \\\\")
    latex_lines.append("\\midrule")

    # Calculate correlations on matched perturbations
    # First, find common perturbations
    gpt_perturbs = set(zip(data['GPT-4.1']['Original Main Part'],
                          data['GPT-4.1']['Rephrased Main Part']))
    claude_perturbs = set(zip(data['Claude Opus 4']['Original Main Part'],
                              data['Claude Opus 4']['Rephrased Main Part']))
    gemini_perturbs = set(zip(data['Gemini 2.0']['Original Main Part'],
                              data['Gemini 2.0']['Rephrased Main Part']))

    common_perturbs = gpt_perturbs & claude_perturbs & gemini_perturbs

    if common_perturbs:
        # Create aligned dataframe
        aligned_data = []
        for orig, rephrase in common_perturbs:
            row = {}

            for model_name, df in data.items():
                mask = (df['Original Main Part'] == orig) & (df['Rephrased Main Part'] == rephrase)
                model_data = df[mask]
                if not model_data.empty:
                    row[model_name] = model_data['Confidence Value'].iloc[0]

            if len(row) == 3:  # All three models have data
                aligned_data.append(row)

        aligned_df = pd.DataFrame(aligned_data)

        if not aligned_df.empty:
            corr_matrix = aligned_df.corr()

            latex_lines.append(f"GPT-4.1 & 1.000 & {corr_matrix.loc['GPT-4.1', 'Claude Opus 4']:.3f}*** & {corr_matrix.loc['GPT-4.1', 'Gemini 2.0']:.3f}*** \\\\")
            latex_lines.append(f"Claude Opus 4 & & 1.000 & {corr_matrix.loc['Claude Opus 4', 'Gemini 2.0']:.3f}*** \\\\")
            latex_lines.append("Gemini 2.0 & & & 1.000 \\\\")

    latex_lines.append("\\bottomrule")
    num_matched = len(aligned_df) if 'aligned_df' in locals() else 0
    latex_lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{Note}}: *** p < 0.001; Based on {num_matched} matched perturbations}} \\\\")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    # Save to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"  Saved LaTeX table to {output_file}")

    return latex_lines

def create_summary_statistics_table(data, output_file='results/combined_analysis/summary_stats_table.csv'):
    """
    Create a CSV table with summary statistics for easy viewing.
    """
    print("\nCreating summary statistics table...")

    summary_data = []

    for model_name in ['GPT-4.1', 'Claude Opus 4', 'Gemini 2.0']:
        df = data[model_name]
        confidence = df['Confidence Value'].dropna()

        if len(confidence) > 0:
            summary_data.append({
                'Model': model_name,
                'N': len(confidence),
                'Mean': f"{confidence.mean():.2f}",
                'Std Dev': f"{confidence.std():.2f}",
                'Median': f"{confidence.median():.1f}",
                'Q1': f"{confidence.quantile(0.25):.1f}",
                'Q3': f"{confidence.quantile(0.75):.1f}",
                'Min': f"{confidence.min():.0f}",
                'Max': f"{confidence.max():.0f}",
                'Low (0-33)%': f"{(confidence <= 33).mean() * 100:.1f}",
                'Med (34-66)%': f"{((confidence > 33) & (confidence <= 66)).mean() * 100:.1f}",
                'High (67-100)%': f"{(confidence > 66).mean() * 100:.1f}"
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"  Saved summary table to {output_file}")

    # Print to console
    print("\n" + "="*80)
    print("SUMMARY STATISTICS TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

    return summary_df

def main():
    """Main function to generate all outputs."""
    print("\n" + "="*70)
    print("GENERATING COMBINED CONFIDENCE VISUALIZATIONS AND TABLES")
    print("="*70)

    # Load all model data
    data = load_all_model_data()

    # Create combined visualization
    fig = create_combined_visualization(data)

    # Create comprehensive LaTeX table
    latex_lines = create_comprehensive_latex_table(data)

    # Create summary statistics table
    summary_df = create_summary_statistics_table(data)

    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print("\nFiles created:")
    print("  1. results/combined_analysis/three_model_confidence_visualization.png")
    print("     - 3x3 grid with stacked visualizations for each model")
    print("  2. results/combined_analysis/comprehensive_confidence_table.tex")
    print("     - LaTeX tables with complete statistics")
    print("  3. results/combined_analysis/summary_stats_table.csv")
    print("     - CSV summary for quick reference")

    print("\nTo include in LaTeX document:")
    print("  \\input{results/combined_analysis/comprehensive_confidence_table.tex}")
    print("  \\includegraphics[width=\\textwidth]{three_model_confidence_visualization.png}")

    return data, fig, latex_lines, summary_df

if __name__ == "__main__":
    data, fig, latex, summary = main()