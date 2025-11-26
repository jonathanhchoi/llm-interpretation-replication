#!/usr/bin/env python3
"""
Create the EXACT same visualization as in gemini_analysis/combined_confidence_visualization.png
but with all three models (GPT-4.1, Claude Opus 4, Gemini 2.0) stacked vertically.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Define the 5 prompts from perturbations.json
prompts = [
    ["An insurance policy contains the following exclusion: 'This policy does not cover flood damage.' "
     "A policyholder's basement is damaged by water that entered through the foundation after "
     "heavy rainfall overwhelmed the city's storm drains, causing widespread water accumulation. "
     "Does this fall within the flood exclusion?",
     ['flood_exclusion'],
     ['Covered', 'Not'],
     'Insurance flood exclusion (Example 1)'
    ],
    ["In a prenuptial agreement, if one party files a petition seeking to invalidate the agreement for lack of "
     "disclosure but that party's attorney had specifically advised them 'this agreement is probably not in your "
     "best interest' before signing and they signed anyway, have they 'waived their right to proper disclosure?'",
     ['prenup_disclosure'],
     ['Yes', 'No'],
     'Prenuptial disclosure waiver'
    ],
    ["Does the following contract term from 1961 naturally contemplate sales tax separate and apart from the sale "
     "price: 'For the sum of five hundred (500) dollars, the seller agrees to transfer ownership of the described "
     "automobile to the buyer'?",
     ['sales_tax'],
     ['Covered', 'Not'],
     'Sales tax contemplation (1961)'
    ],
    ["A contractor and business corresponded about construction of four closets. Emails show Business "
     "asked, 'can you build one closet?' Contractor replied, 'I can build that closet for you.' Business then said, "
     "'let's discuss the project when you arrive.' Written contract states 'build closets in office' without specifying "
     "number. How many closets are covered by the contract?",
     ['closet_count'],
     ['Covered', 'Not'],
     'Closet construction count'
    ],
    ["You are analyzing an insurance policy dispute. The policy holder has standard commercial general liability "
     "insurance. They are being sued because their security camera system allegedly records audio of passersby on "
     "the public sidewalk without consent, violating state privacy laws. "
     "The insurance company denies coverage citing the policy's exclusion for 'injury arising out of oral or "
     "written publication of material that violates a person's right of privacy.' "
     "The policy defines 'publication' as 'dissemination of information to the public.' "
     "The security system only stores recordings locally and has never shared them. "
     "Does the exclusion apply to deny coverage?",
     ['privacy_exclusion'],
     ['Covered', 'Not'],
     'Privacy exclusion for security recordings (Example 2)'
    ]
]

def load_model_data():
    """Load data for all three models."""
    data = {}

    # Load GPT-5
    print("Loading GPT-5 data...")
    combined_df = pd.read_excel('results/combined_results.xlsx')
    gpt5_df = combined_df[combined_df['Model'] == 'gpt-5'].copy()
    data['GPT-5'] = gpt5_df
    print(f"  Loaded {len(gpt5_df)} GPT-5 records")

    # Load Claude
    print("Loading Claude Opus 4.1 data...")
    claude_df = pd.read_excel('results/claude_opus_batch_perturbation_results.xlsx')
    data['Claude Opus 4.1'] = claude_df
    print(f"  Loaded {len(claude_df)} Claude records")

    # Load Gemini
    print("Loading Gemini 2.5 Pro data...")
    gemini_df = pd.read_excel('results/gemini_perturbation_results.xlsx')
    data['Gemini 2.5 Pro'] = gemini_df
    print(f"  Loaded {len(gemini_df)} Gemini records")

    return data

def create_single_model_visualization(ax, df, model_name, prompts):
    """Create visualization for a single model matching the exact style."""

    # Get unique prompts
    unique_prompts = df['Original Main Part'].unique()

    # Map the actual prompts in the data to the prompt definitions
    prompt_mapping = {}
    for data_prompt in unique_prompts:
        for idx, prompt_info in enumerate(prompts):
            if data_prompt[:50] in prompt_info[0] or prompt_info[0][:50] in data_prompt:
                prompt_mapping[data_prompt] = idx
                break

    # Define colors for each prompt (same colors as original)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Sort the prompts by their prompt index to ensure sequential ordering
    sorted_prompts = []
    for data_prompt in unique_prompts:
        if data_prompt in prompt_mapping:
            sorted_prompts.append((prompt_mapping[data_prompt], data_prompt))
    sorted_prompts.sort(key=lambda x: x[0])  # Sort by prompt index

    # Track actual plotting positions
    plot_position = 0
    plotted_prompts = []

    # Plot violin plots and jittered points for each prompt in sequential order
    for prompt_idx, data_prompt in sorted_prompts:
        prompt_data = df[df['Original Main Part'] == data_prompt]
        confidence_values = prompt_data['Confidence Value'].dropna()

        if len(confidence_values) == 0:
            print(f"  Skipping prompt {prompt_idx+1} for {model_name} - no valid data")
            continue

        plot_position += 1
        plotted_prompts.append(prompt_idx)

        # Calculate statistics
        mean_conf = confidence_values.mean()
        lower_percentile = np.percentile(confidence_values, 2.5)
        upper_percentile = np.percentile(confidence_values, 97.5)

        # Add violin plot (with lower alpha to not obscure points)
        violin_parts = ax.violinplot([confidence_values.values], [plot_position],
                                    widths=0.3, showmeans=False, showmedians=False, showextrema=False)
        for pc in violin_parts['bodies']:
            pc.set_facecolor(colors[prompt_idx % len(colors)])
            pc.set_edgecolor('none')
            pc.set_alpha(0.3)

        # Add jittered points
        np.random.seed(42 + prompt_idx)  # For reproducibility
        x_jittered = np.random.normal(plot_position, 0.08, size=len(confidence_values))
        ax.scatter(x_jittered, confidence_values.values, alpha=0.4, s=30,
                  color=colors[prompt_idx % len(colors)])

        # Add mean point (black dot)
        ax.scatter(plot_position, mean_conf, color='black', s=80, zorder=5)

        # Add error bars for 95% CI
        ax.plot([plot_position, plot_position], [lower_percentile, upper_percentile],
               color='black', linewidth=2, zorder=4)

        # Add caps to the error bars
        cap_width = 0.1
        ax.plot([plot_position - cap_width, plot_position + cap_width],
               [lower_percentile, lower_percentile], color='black', linewidth=2, zorder=4)
        ax.plot([plot_position - cap_width, plot_position + cap_width],
               [upper_percentile, upper_percentile], color='black', linewidth=2, zorder=4)

    # Add a horizontal line at 50 for reference (neutral confidence)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)

    # Set the x-ticks and labels - only show prompt numbers
    ax.set_xticks(range(1, plot_position + 1))
    ax.set_xticklabels([f"{idx+1}" for idx in plotted_prompts], fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Add labels - no x-label, we'll add it only to the bottom panel
    # ax.set_xlabel('Prompt Number', fontsize=16)  # Removed
    ax.set_ylabel('Confidence (0-100)', fontsize=16)

    # Set y-axis limits
    ax.set_ylim(0, 100)

    # Model name will be added as title above the plot instead
    # ax.text(0.02, 0.95, model_name, transform=ax.transAxes, fontsize=16,
    #         fontweight='bold', va='top',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return plotted_prompts, colors

def create_three_model_stacked_visualization(data, output_file='results/combined_analysis/three_model_stacked_visualization.png'):
    """Create stacked visualization with all three models."""

    print("\nCreating three-model stacked visualization...")

    # Create figure with 3 subplots stacked vertically
    # Reduced height to 70% of original (24 * 0.7 = 16.8)
    fig, axes = plt.subplots(3, 1, figsize=(14, 16.8))  # Tall figure for 3 stacked plots

    # Process each model
    models = ['GPT-5', 'Claude Opus 4.1', 'Gemini 2.5 Pro']
    all_plotted_prompts = None
    legend_colors = None

    for idx, model_name in enumerate(models):
        print(f"  Processing {model_name}...")
        plotted_prompts, colors = create_single_model_visualization(
            axes[idx], data[model_name], model_name, prompts
        )

        # Add title above each subplot
        axes[idx].set_title(model_name, fontsize=18, fontweight='bold', pad=10)

        # Store the first model's prompt info for legend
        if all_plotted_prompts is None:
            all_plotted_prompts = plotted_prompts
            legend_colors = colors

        # Add x-axis label only to the bottom panel
        if idx == 2:  # Bottom panel (Gemini)
            axes[idx].set_xlabel('Prompt Number', fontsize=16)

    # Legend removed as requested
    # # Create custom legend elements at the bottom of the figure
    # custom_legend = []
    # for prompt_idx in all_plotted_prompts:
    #     token_options = prompts[prompt_idx][2]
    #     first_token = token_options[0]
    #
    #     # Create colored patch for legend
    #     patch = mpatches.Patch(color=legend_colors[prompt_idx % len(legend_colors)],
    #                           label=f"Prompt {prompt_idx+1}: Pertaining to {first_token}")
    #     custom_legend.append(patch)
    #
    # # Add legend at the bottom of the figure
    # fig.legend(handles=custom_legend, loc='upper center', ncol=3,
    #           fontsize=14, frameon=True, bbox_to_anchor=(0.5, 0.06))

    # No overall title - removed as requested
    # fig.suptitle('Verbalized Confidence Distributions Across Three Models',
    #             fontsize=20, fontweight='bold', y=1.01)

    # Adjust layout - no need for extra bottom padding without legend
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  Saved visualization to {output_file}")

    return fig

def create_comprehensive_latex_table(data, output_file='results/combined_analysis/comprehensive_three_model_table.tex'):
    """Create comprehensive LaTeX table with detailed statistics for all three models."""

    print("\nCreating comprehensive LaTeX table...")

    latex_lines = []

    # Create the comprehensive table with all three models
    latex_lines.append("% Comprehensive Table: Verbalized Confidence Statistics for All Three Models")
    latex_lines.append("\\begin{table}[H]")
    latex_lines.append("\\centering")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\hline")
    latex_lines.append("\\textbf{Model /} & \\textbf{Mean} & \\textbf{Std Dev} & \\textbf{2.5th} & \\textbf{97.5th} & \\textbf{95\\% CI} \\\\")
    latex_lines.append("\\textbf{Prompt Number} & \\textbf{Confidence} & & \\textbf{Percentile} & \\textbf{Percentile} & \\textbf{Width} \\\\")
    latex_lines.append("\\hline")

    # Process each model
    for model_name in ['GPT-5', 'Claude Opus 4.1', 'Gemini 2.5 Pro']:
        df = data[model_name]

        # Add model name as section header
        latex_lines.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{model_name}}}}} \\\\")
        latex_lines.append("\\hline")

        # Get unique prompts and map them
        unique_prompts = df['Original Main Part'].unique()
        prompt_mapping = {}
        for data_prompt in unique_prompts:
            for idx, prompt_info in enumerate(prompts):
                if data_prompt[:50] in prompt_info[0] or prompt_info[0][:50] in data_prompt:
                    prompt_mapping[data_prompt] = idx
                    break

        # Calculate statistics for each prompt
        for prompt_idx in range(5):
            # Find the corresponding data prompt
            for data_prompt, mapped_idx in prompt_mapping.items():
                if mapped_idx == prompt_idx:
                    prompt_data = df[df['Original Main Part'] == data_prompt]
                    confidence_values = prompt_data['Confidence Value'].dropna()

                    if len(confidence_values) > 0:
                        mean_conf = confidence_values.mean()
                        std_conf = confidence_values.std()
                        percentile_2_5 = np.percentile(confidence_values, 2.5)
                        percentile_97_5 = np.percentile(confidence_values, 97.5)
                        ci_width = percentile_97_5 - percentile_2_5

                        latex_lines.append(f"{prompt_idx+1} & {mean_conf:.2f} & {std_conf:.2f} & "
                                         f"{percentile_2_5:.2f} & {percentile_97_5:.2f} & {ci_width:.2f} \\\\")
                    else:
                        latex_lines.append(f"{prompt_idx+1} & -- & -- & -- & -- & -- \\\\")
                    break
            else:
                latex_lines.append(f"{prompt_idx+1} & -- & -- & -- & -- & -- \\\\")

        if model_name != 'Gemini 2.5 Pro':  # Add separator between models except after last one
            latex_lines.append("\\hline")

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{Summary statistics for prompt perturbations with respect to verbalized confidence scores "
                      "for GPT-5, Claude Opus 4.1, and Gemini 2.5 Pro. This table presents confidence assessments from three "
                      "large language models on the same set of legal interpretation prompts. Each model was directly queried "
                      "for its confidence on a 0-100 scale. The same perturbations of five legal scenarios were tested across "
                      "all models. The 95\\% confidence interval width measures variation in each model's confidence assessments "
                      "across different phrasings of the same legal question.}")
    latex_lines.append("\\label{tab:three_model_confidence_stats}")
    latex_lines.append("\\end{table}")

    # Save to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"  Saved comprehensive LaTeX table to {output_file}")

    return latex_lines


def create_latex_table(data, output_file='results/combined_analysis/three_model_confidence_table.tex'):
    """Create LaTeX table with confidence statistics for all three models."""

    print("\nCreating LaTeX table...")

    # Calculate statistics for each model and prompt
    latex_lines = []

    latex_lines.append("% Table: Verbalized Confidence Statistics by Model and Prompt")
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Mean Verbalized Confidence Scores by Model and Prompt}")
    latex_lines.append("\\label{tab:model_prompt_confidence}")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Model & Prompt 1 & Prompt 2 & Prompt 3 & Prompt 4 & Prompt 5 & Overall \\\\")
    latex_lines.append("\\midrule")

    for model_name in ['GPT-5', 'Claude Opus 4.1', 'Gemini 2.5 Pro']:
        df = data[model_name]

        # Get unique prompts and map them
        unique_prompts = df['Original Main Part'].unique()
        prompt_mapping = {}
        for data_prompt in unique_prompts:
            for idx, prompt_info in enumerate(prompts):
                if data_prompt[:50] in prompt_info[0] or prompt_info[0][:50] in data_prompt:
                    prompt_mapping[data_prompt] = idx
                    break

        # Calculate mean for each prompt
        prompt_means = []
        for prompt_idx in range(5):
            # Find the corresponding data prompt
            for data_prompt, mapped_idx in prompt_mapping.items():
                if mapped_idx == prompt_idx:
                    prompt_data = df[df['Original Main Part'] == data_prompt]
                    confidence_values = prompt_data['Confidence Value'].dropna()
                    if len(confidence_values) > 0:
                        prompt_means.append(f"{confidence_values.mean():.1f}")
                    else:
                        prompt_means.append("--")
                    break
            else:
                prompt_means.append("--")

        # Overall mean
        overall_mean = df['Confidence Value'].dropna().mean()

        latex_lines.append(f"{model_name} & {' & '.join(prompt_means)} & {overall_mean:.1f} \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    # Save to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_lines))

    print(f"  Saved LaTeX table to {output_file}")

    return latex_lines

def main():
    """Main function."""

    print("\n" + "="*70)
    print("CREATING THREE-MODEL STACKED VISUALIZATION")
    print("(Exact style as gemini_analysis/combined_confidence_visualization.png)")
    print("="*70)

    # Load data
    data = load_model_data()

    # Create stacked visualization
    fig = create_three_model_stacked_visualization(data)

    # Create both LaTeX tables
    latex = create_latex_table(data)
    comprehensive_latex = create_comprehensive_latex_table(data)

    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print("\nFiles created:")
    print("  1. results/combined_analysis/three_model_stacked_visualization.png")
    print("     - Exact style as original, but with 3 models stacked")
    print("  2. results/combined_analysis/three_model_confidence_table.tex")
    print("     - LaTeX table with mean confidence by prompt and model")
    print("  3. results/combined_analysis/comprehensive_three_model_table.tex")
    print("     - Comprehensive table with detailed statistics for all three models")

    return data, fig

if __name__ == "__main__":
    data, fig = main()