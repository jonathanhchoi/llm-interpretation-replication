import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy import stats as scipy_stats
import json
import ast

# Set global font sizes for all plots - INCREASED SIZES
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 22
})

# Function to conduct normality tests on distribution data
def conduct_normality_tests(data, column_name, prompt_idx):
    """
    Conduct Kolmogorov-Smirnov and Anderson-Darling tests for normality.
    Returns a dictionary with test results.
    """
    # Extract the data
    values = data[column_name].values

    # Filter out non-finite values (NaN, inf, -inf)
    finite_mask = np.isfinite(values)
    values = values[finite_mask]

    # Check if we have enough valid data
    if len(values) == 0:
        print(f"Warning: No finite values found for prompt {prompt_idx + 1}, column {column_name}")
        return {
            'Prompt': prompt_idx + 1,
            'Distribution Mean': np.nan,
            'Distribution Std Dev': np.nan,
            'KS Statistic': np.nan,
            'KS p-value': np.nan,
            'KS Normal (p>0.05)': False,
            'AD Statistic': np.nan,
            'AD p-value': np.nan,
            'AD Critical Value (5%)': np.nan,
            'AD Normal (stat<crit)': False
        }

    if len(values) < 3:
        print(f"Warning: Insufficient data for normality tests (n={len(values)}) for prompt {prompt_idx + 1}, column {column_name}")
        return {
            'Prompt': prompt_idx + 1,
            'Distribution Mean': np.mean(values) if len(values) > 0 else np.nan,
            'Distribution Std Dev': np.std(values) if len(values) > 1 else np.nan,
            'KS Statistic': np.nan,
            'KS p-value': np.nan,
            'KS Normal (p>0.05)': False,
            'AD Statistic': np.nan,
            'AD p-value': np.nan,
            'AD Critical Value (5%)': np.nan,
            'AD Normal (stat<crit)': False
        }

    # Fit a normal distribution to the data
    mu, sigma = scipy_stats.norm.fit(values)

    # Conduct Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = scipy_stats.kstest(values, 'norm', args=(mu, sigma))

    # Conduct Anderson-Darling test
    ad_result = scipy_stats.anderson(values, 'norm')
    ad_statistic = ad_result.statistic
    ad_critical_values = ad_result.critical_values
    ad_significance_level = ad_result.significance_level

    # Determine if AD test indicates normality (at 5% significance)
    ad_normal = ad_statistic < ad_critical_values[2]  # Index 2 corresponds to 5% significance

    # Calculate p-value for Anderson-Darling test
    # We'll use an approximation as scipy doesn't directly provide AD p-values
    if ad_statistic > 10:
        ad_pvalue = 0.0001  # Very small p-value for large statistics
    elif ad_statistic > ad_critical_values[4]:  # Index 4 corresponds to 1% significance
        ad_pvalue = 0.005  # Between 0.5% and 1%
    elif ad_statistic > ad_critical_values[3]:  # Index 3 corresponds to 2.5% significance
        ad_pvalue = 0.015  # Between 1% and 2.5%
    elif ad_statistic > ad_critical_values[2]:  # Index 2 corresponds to 5% significance
        ad_pvalue = 0.035  # Between 2.5% and 5%
    elif ad_statistic > ad_critical_values[1]:  # Index 1 corresponds to 10% significance
        ad_pvalue = 0.075  # Between 5% and 10%
    else:
        ad_pvalue = 0.15  # Greater than 10% significance

    # Create a results dictionary
    results = {
        'Prompt': prompt_idx + 1,
        'Distribution Mean': mu,
        'Distribution Std Dev': sigma,
        'KS Statistic': ks_statistic,
        'KS p-value': ks_pvalue,
        'KS Normal (p>0.05)': ks_pvalue > 0.05,
        'AD Statistic': ad_statistic,
        'AD p-value': ad_pvalue,
        'AD Critical Value (5%)': ad_critical_values[2],
        'AD Normal (stat<crit)': ad_normal
    }

    return results

def create_qq_plot(data, column_name, prompt_idx, token_options, output_dir):
    """Create a Q-Q plot to assess normality."""
    # Extract the data
    values = data[column_name].dropna().values

    # Filter out non-finite values
    finite_mask = np.isfinite(values)
    values = values[finite_mask]

    if len(values) < 3:
        print(f"  Skipping Q-Q plot for prompt {prompt_idx + 1} {column_name} - insufficient data")
        return

    # Create the Q-Q plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Calculate theoretical quantiles and sample quantiles
    scipy_stats.probplot(values, dist="norm", plot=ax)

    # Customize the plot
    if column_name == 'Confidence Value':
        ax.set_title(f'Q-Q Plot for Prompt {prompt_idx + 1}: Confidence Values', fontsize=22)
        ax.set_xlabel('Theoretical Quantiles', fontsize=20)
        ax.set_ylabel('Sample Quantiles (Confidence)', fontsize=20)
    else:
        ax.set_title(f'Q-Q Plot for Prompt {prompt_idx + 1}', fontsize=22)
        ax.set_xlabel('Theoretical Quantiles', fontsize=20)
        ax.set_ylabel(f'Sample Quantiles ({column_name})', fontsize=20)

    ax.grid(True, alpha=0.3)

    # Set tick label sizes
    ax.tick_params(axis='both', labelsize=18)

    plt.tight_layout()

    # Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"qq_plot_prompt_{prompt_idx + 1}_{column_name.replace(' ', '_').lower()}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Q-Q plot saved: {filename}")

def create_confidence_histogram(data, prompt_idx, token_options, output_dir):
    """Create histogram of confidence values matching the style of analyze_perturbation_results.py."""
    # Extract confidence values
    confidence_values = data['Confidence Value'].dropna()

    if len(confidence_values) == 0:
        print(f"  Skipping confidence histogram for prompt {prompt_idx + 1} - no valid data")
        return

    # Create the histogram with exact same style as original
    plt.figure(figsize=(12, 8))  # Increased height to accommodate legend at bottom

    # Use seaborn for consistent style
    filtered_data = pd.DataFrame({'Confidence Value': confidence_values})
    sns.histplot(data=filtered_data, x='Confidence Value', bins=10)

    # Get the first token
    first_token = token_options[0]

    # Remove title and use larger font sizes for labels (matching original style)
    plt.xlabel(f'Confidence (0-100) for "{first_token}"', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)

    # Calculate statistics
    mean_conf = confidence_values.mean()
    lower_percentile = np.percentile(confidence_values, 2.5)
    upper_percentile = np.percentile(confidence_values, 97.5)

    # Add vertical lines for mean and 95% confidence interval
    plt.axvline(x=mean_conf, color='r', linestyle='--',
                label=f'Mean: {mean_conf:.1f}')
    plt.axvline(x=lower_percentile, color='g', linestyle=':',
                label=f'2.5th percentile: {lower_percentile:.1f}')
    plt.axvline(x=upper_percentile, color='g', linestyle=':',
                label=f'97.5th percentile: {upper_percentile:.1f}')

    # Add shaded region for 95% interval
    plt.axvspan(lower_percentile, upper_percentile, alpha=0.2, color='green')

    # Add a reference line at 50 (neutral confidence)
    plt.axvline(x=50, color='gray', linestyle='--', alpha=0.7)

    # Position legend at the bottom outside the plot
    plt.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Save the plot with extra padding at bottom for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Add extra space at bottom for legend

    # Save with high DPI and consistent naming
    filename = f"prompt_{prompt_idx + 1}_confidence_distribution.png"
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Confidence histogram saved: {filename}")

def create_latex_table(data, prompt_idx, prompt_info, output_dir):
    """Create LaTeX table summarizing the results for a prompt."""
    original_prompt, response_format, token_options = prompt_info

    # Calculate statistics for confidence values
    confidence_values = data['Confidence Value'].dropna()

    if len(confidence_values) == 0:
        print(f"  Skipping LaTeX table for prompt {prompt_idx + 1} - no valid data")
        return ""

    # Basic statistics
    mean_conf = confidence_values.mean()
    std_conf = confidence_values.std()
    median_conf = confidence_values.median()
    min_conf = confidence_values.min()
    max_conf = confidence_values.max()

    # Calculate percentiles
    p25 = confidence_values.quantile(0.25)
    p75 = confidence_values.quantile(0.75)
    p025 = confidence_values.quantile(0.025)
    p975 = confidence_values.quantile(0.975)

    # Count how many favor each token (using 50 as threshold)
    favor_token1 = (confidence_values > 50).sum()
    favor_token2 = (confidence_values < 50).sum()
    neutral = (confidence_values == 50).sum()

    # Create LaTeX table
    latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Prompt {prompt_idx + 1}: Confidence Analysis for "{token_options[0]}" vs "{token_options[1]}"}}
\\begin{{tabular}}{{lr}}
\\toprule
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\midrule
Mean Confidence & {mean_conf:.2f} \\\\
Std Deviation & {std_conf:.2f} \\\\
Median & {median_conf:.2f} \\\\
Min & {min_conf:.0f} \\\\
Max & {max_conf:.0f} \\\\
\\midrule
25th Percentile & {p25:.2f} \\\\
75th Percentile & {p75:.2f} \\\\
95\\% CI & [{p025:.2f}, {p975:.2f}] \\\\
\\midrule
Favor "{token_options[0]}" (>50) & {favor_token1} ({100*favor_token1/len(confidence_values):.1f}\\%) \\\\
Favor "{token_options[1]}" (<50) & {favor_token2} ({100*favor_token2/len(confidence_values):.1f}\\%) \\\\
Neutral (=50) & {neutral} ({100*neutral/len(confidence_values):.1f}\\%) \\\\
\\midrule
Total Samples & {len(confidence_values)} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # Save the table to a file
    table_file = output_dir / f"table_prompt_{prompt_idx + 1}.tex"
    with open(table_file, 'w') as f:
        f.write(latex_table)

    print(f"  LaTeX table saved: table_prompt_{prompt_idx + 1}.tex")

    return latex_table

def create_combined_confidence_visualization(df, prompts, output_dir):
    """Create a combined visualization matching the exact style of analyze_perturbation_results.py."""

    plt.figure(figsize=(14, 10))  # Increased height to accommodate legend at bottom

    # Get unique prompts
    unique_prompts = df['Original Main Part'].unique()

    # Map the actual prompts in the data to the prompt definitions
    prompt_mapping = {}
    for data_prompt in unique_prompts:
        for idx, prompt_info in enumerate(prompts):
            if data_prompt[:50] in prompt_info[0] or prompt_info[0][:50] in data_prompt:
                prompt_mapping[data_prompt] = idx
                break

    # Set up the plot
    ax = plt.subplot(111)

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
            print(f"Skipping prompt {prompt_idx+1} in combined confidence visualization - no valid data")
            continue

        plot_position += 1
        plotted_prompts.append(prompt_idx)

        # Calculate statistics
        mean_conf = confidence_values.mean()
        lower_percentile = np.percentile(confidence_values, 2.5)
        upper_percentile = np.percentile(confidence_values, 97.5)

        # Get token options
        token_options = prompts[prompt_idx][2]
        first_token = token_options[0]

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
        plt.scatter(x_jittered, confidence_values.values, alpha=0.4, s=30,
                   color=colors[prompt_idx % len(colors)])

        # Add mean point (black dot)
        plt.scatter(plot_position, mean_conf, color='black', s=80, zorder=5)

        # Add error bars for 95% CI
        plt.plot([plot_position, plot_position], [lower_percentile, upper_percentile],
                color='black', linewidth=2, zorder=4)

        # Add caps to the error bars
        cap_width = 0.1
        plt.plot([plot_position - cap_width, plot_position + cap_width],
                [lower_percentile, lower_percentile], color='black', linewidth=2, zorder=4)
        plt.plot([plot_position - cap_width, plot_position + cap_width],
                [upper_percentile, upper_percentile], color='black', linewidth=2, zorder=4)

    # Add a horizontal line at 50 for reference (neutral confidence)
    plt.axhline(y=50, color='gray', linestyle='--', alpha=0.7)

    # Set the x-ticks and labels - only show prompt numbers
    plt.xticks(range(1, plot_position + 1),
              [f"{idx+1}" for idx in plotted_prompts], fontsize=18)
    plt.yticks(fontsize=18)

    # Add labels but no title (matching original style - NO TITLE!)
    plt.xlabel('Prompt Number', fontsize=20)
    plt.ylabel('Confidence (0-100)', fontsize=20)

    # Set y-axis limits
    plt.ylim(0, 100)

    # Create custom legend elements
    custom_legend = []
    for prompt_idx in plotted_prompts:
        token_options = prompts[prompt_idx][2]
        first_token = token_options[0]
        custom_legend.append(plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=colors[prompt_idx % len(colors)], markersize=10,
                                       label=f"Prompt {prompt_idx+1}: Confidence for '{first_token}'"))

    # Add the legend at the bottom of the plot
    plt.legend(handles=custom_legend, fontsize=16, loc='upper center',
              bbox_to_anchor=(0.5, -0.15), ncol=1)

    # Adjust layout and save with extra space at bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Add extra space at bottom for legend

    # Save with high DPI (300) matching the original
    plt.savefig(output_dir / 'combined_confidence_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Combined confidence visualization saved: combined_confidence_visualization.png")

def analyze_claude_model(df, model_name, prompts, output_dir):
    """Perform complete analysis for Claude model (adapted for confidence-only data)."""
    print(f"\nStarting analysis for {model_name}...")

    # Check if we have enough data
    if len(df) < 100:
        print(f"WARNING: Only {len(df)} rows available for {model_name}. Statistical tests may be unreliable.")

    # Get unique prompts
    unique_prompts = df['Original Main Part'].unique()

    # Map the actual prompts in the data to the prompt definitions
    # We'll match based on the beginning of the prompt text
    prompt_mapping = {}
    for data_prompt in unique_prompts:
        for idx, prompt_info in enumerate(prompts):
            if data_prompt[:50] in prompt_info[0] or prompt_info[0][:50] in data_prompt:
                prompt_mapping[data_prompt] = idx
                break

    print(f"Found {len(unique_prompts)} unique prompts in data")
    print(f"Mapped to prompt indices: {list(prompt_mapping.values())}")

    # Collect all LaTeX tables
    all_tables = []

    # Create summary statistics
    summary_stats = []

    # Conduct normality tests for all prompts
    normality_results = []

    # Sort the prompts by their prompt index to ensure sequential processing
    sorted_prompts = []
    for original_prompt in unique_prompts:
        if original_prompt in prompt_mapping:
            sorted_prompts.append((prompt_mapping[original_prompt], original_prompt))
    sorted_prompts.sort(key=lambda x: x[0])  # Sort by prompt index

    # Process each unique original prompt in sequential order
    for idx, original_prompt in sorted_prompts:
        prompt_data = df[df['Original Main Part'] == original_prompt]
        prompt_info = prompts[idx]
        token_options = prompt_info[2]

        print(f"\nProcessing Prompt {idx + 1}: {len(prompt_data)} perturbations")
        print(f"  Tokens: '{token_options[0]}' vs '{token_options[1]}'")

        # Create confidence histogram
        create_confidence_histogram(prompt_data, idx, token_options, output_dir / "figures")

        # Create Q-Q plot for confidence values
        create_qq_plot(prompt_data, 'Confidence Value', idx, token_options, output_dir / "figures")

        # Create LaTeX table
        latex_table = create_latex_table(prompt_data, idx, prompt_info, output_dir)
        all_tables.append(latex_table)

        # Calculate statistics for confidence values
        confidence_values = prompt_data['Confidence Value'].dropna()

        if len(confidence_values) > 0:
            # Calculate percentiles for 95% interval
            lower_percentile = np.percentile(confidence_values, 2.5)
            upper_percentile = np.percentile(confidence_values, 97.5)
            mean_conf = confidence_values.mean()
            std_conf = confidence_values.std()
            min_conf = confidence_values.min()
            max_conf = confidence_values.max()

            stats = {
                'Prompt Number': idx + 1,
                'First Token': token_options[0],
                'Second Token': token_options[1],
                'Mean Confidence': mean_conf,
                'Std Dev': std_conf,
                'Min': min_conf,
                'Max': max_conf,
                '2.5th Percentile': lower_percentile,
                '97.5th Percentile': upper_percentile,
                '95% Interval Width': upper_percentile - lower_percentile,
                'Sample Size': len(confidence_values),
                'Favors First Token (>50)': (confidence_values > 50).sum(),
                'Favors Second Token (<50)': (confidence_values < 50).sum(),
                'Neutral (=50)': (confidence_values == 50).sum()
            }

            summary_stats.append(stats)

            # Conduct normality tests on confidence values
            norm_results = conduct_normality_tests(prompt_data, 'Confidence Value', idx)
            norm_results['Column'] = 'Confidence Value'
            normality_results.append(norm_results)

    # Save summary statistics
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
        print(f"\nSummary statistics saved to: {output_dir / 'summary_statistics.csv'}")

        # Also save as LaTeX table
        try:
            latex_summary = summary_df.to_latex(index=False, float_format="%.2f")
            with open(output_dir / 'summary_statistics.tex', 'w') as f:
                f.write(latex_summary)
            print(f"Summary LaTeX table saved to: {output_dir / 'summary_statistics.tex'}")
        except ImportError as e:
            print(f"Warning: Could not generate LaTeX summary table: {e}")
            # Alternative: create a simple LaTeX table manually
            with open(output_dir / 'summary_statistics.tex', 'w') as f:
                f.write("% Summary statistics table\n")
                f.write("\\begin{tabular}{lrrrrr}\n")
                f.write("\\toprule\n")
                f.write("Prompt & Mean & Std & Min & Max & N \\\\\n")
                f.write("\\midrule\n")
                for _, row in summary_df.iterrows():
                    f.write(f"{row['Prompt Number']} & {row['Mean Confidence']:.2f} & {row['Std Dev']:.2f} & {row['Min']:.0f} & {row['Max']:.0f} & {row['Sample Size']:.0f} \\\\\n")
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
            print(f"Basic LaTeX table saved to: {output_dir / 'summary_statistics.tex'}")

    # Save normality test results
    if normality_results:
        normality_df = pd.DataFrame(normality_results)
        normality_df.to_csv(output_dir / 'normality_tests.csv', index=False)
        print(f"Normality test results saved to: {output_dir / 'normality_tests.csv'}")

        # Print summary of normality tests
        print("\nNormality Test Summary:")
        print(f"  KS Test: {normality_df['KS Normal (p>0.05)'].sum()}/{len(normality_df)} prompts appear normal")
        print(f"  AD Test: {normality_df['AD Normal (stat<crit)'].sum()}/{len(normality_df)} prompts appear normal")

    # Save all LaTeX tables to a single file
    if all_tables:
        with open(output_dir / 'all_tables.tex', 'w') as f:
            f.write("% All LaTeX tables for Claude analysis\n\n")
            for table in all_tables:
                f.write(table)
                f.write("\n\n")
        print(f"All LaTeX tables saved to: {output_dir / 'all_tables.tex'}")

    # Create combined visualization
    create_combined_confidence_visualization(df, prompts, output_dir)

    # Generate overall statistics
    overall_stats = {
        'Model': model_name,
        'Total Perturbations': len(df),
        'Total Prompts': len(unique_prompts),
        'Overall Mean Confidence': df['Confidence Value'].mean(),
        'Overall Std Dev': df['Confidence Value'].std(),
        'Missing Confidence Values': df['Confidence Value'].isna().sum(),
    }

    overall_df = pd.DataFrame([overall_stats])
    overall_df.to_csv(output_dir / 'overall_statistics.csv', index=False)

    print(f"\nAnalysis complete for {model_name}!")
    print(f"Results saved to: {output_dir}")

# Main script
if __name__ == "__main__":
    # Configuration for Claude results
    CLAUDE_RESULTS_FILE = Path("G:/My Drive/Colab Notebooks/llm_interpretation_replication/results/claude_opus_batch_perturbation_results.xlsx")
    OUTPUT_DIR = Path("G:/My Drive/Colab Notebooks/llm_interpretation_replication/results/claude_analysis")

    # Check if the results file exists
    if not CLAUDE_RESULTS_FILE.exists():
        print(f"ERROR: Results file not found at {CLAUDE_RESULTS_FILE}")
        print("Please ensure the Claude perturbation results file exists.")
        exit(1)

    print(f"Reading Claude results from: {CLAUDE_RESULTS_FILE}")
    df = pd.read_excel(CLAUDE_RESULTS_FILE)

    print(f"Loaded {len(df)} rows of data")
    print(f"Columns: {list(df.columns)}")

    # Define prompts list (same as in original analysis)
    prompts = [
        ("An insurance policy contains the following exclusion: 'We will not pay for loss or damage caused directly or indirectly by flood, surface water, waves, tides, tidal waves, overflow of any body of water, or their spray, all whether driven by wind or not.' If an insured files a claim for damage caused by a levee failure rather than natural causes, does the exclusion apply? Please analyze whether the insurance policy would cover the damage or not.",
         "Answer only 'Covered' if insurance covers the loss or 'Not Covered' if it does not, without any other explanation.",
         ("Covered", "Not")),

        ("In a prenuptial agreement, if one party files a petition for divorce after seven years, withdraws it without serving it, and then files again after ten years, which filing date determines the number of years of marriage for calculating the payment amount? The agreement states that payment will be based on 'the number of full years they have been married at the time a Petition for Dissolution of Marriage is filed.' Please analyze whether 'a Petition' refers to the first or ultimate petition filed.",
         "Answer only 'First Petition' if the first filing date is used or 'Ultimate Petition' if the ultimate filing date is used, without any other explanation.",
         ("Ultimate", "First")),

        ("Does the following contract term from 1961 naturally include only existing affiliates at the time of contract, or does it potentially encompass affiliates that might be created over time? The term binds [Company] and its 'other affiliate[s]' to a 50/50 royalty split after deducting fees charged by third parties that intermediate in foreign markets. Please analyze whether the term 'other affiliate[s]' includes only existing affiliates or includes future affiliates as well.",
         "Answer only 'Existing Affiliates' or 'Future Affiliates', without any other explanation.",
         ("Existing", "Future")),

        ("A contractor and business corresponded about construction of a new foundry. The contractor offered to do the job either by offering an itemized list or charging cost + 10%. After a phone call where they allegedly agreed payment would be made 'in the usual manner', the foundry accepted in writing. If one party claims it is customary to pay 85% of payments due at the end of every month, but the other argues payments are only due upon substantial completion, how should the term 'usual manner' be interpreted? Does this term refer to the monthly installment payments or to payment upon completion?",
         "Answer only 'Monthly Installment Payments' or 'Payment Upon Completion', without any other explanation.",
         ("Monthly", "Payment")),

        ("You are analyzing an insurance policy dispute. The policy states: '[Insurer will pay for] the felonious abstraction of insured property (1) from within the premises by a person making felonious entry therein by actual force and violence, of which force and violence there are visible marks made by tools, explosives, electricity or chemicals.' A business has experienced a theft where there is clear evidence that a third party committed the burglary. No inside job is suspected. Based on these terms, would this policy provide compensation for losses resulting from this substantiated third-party burglary? Please analyze whether coverage would be provided.",
         "Answer only 'Covered' if insurance covers the loss or 'Not Covered' if it does not, without any other explanation.",
         ("Covered", "Not"))
    ]

    # Check which model(s) are in the dataset
    if 'Model' in df.columns:
        models = df['Model'].unique()
        print(f"\nModels found in dataset: {models}")
        model_name = models[0] if len(models) == 1 else "Claude_Combined"
    else:
        model_name = "claude-3-opus-20240229"  # Default based on the batch script

    print(f"\nAnalyzing as: {model_name}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    print(f"Output directory: {OUTPUT_DIR}")

    # Perform analysis
    print("\n" + "="*60)
    print("STARTING CLAUDE CONFIDENCE ANALYSIS")
    print("="*60)

    analyze_claude_model(df, model_name, prompts, OUTPUT_DIR)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - figures/: Histograms and Q-Q plots for each prompt")
    print("  - summary_statistics.csv: Statistical summary for all prompts")
    print("  - normality_tests.csv: Results of normality tests")
    print("  - all_tables.tex: LaTeX tables for inclusion in papers")
    print("  - overall_statistics.csv: Overall model statistics")
    print("  - combined_confidence_visualization.png: Combined visualization")