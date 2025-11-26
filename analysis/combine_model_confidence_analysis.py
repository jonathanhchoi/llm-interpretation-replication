"""
Combined Model Confidence Analysis Script

This script combines verbalized confidence scores from GPT-4, Claude Opus 4,
and Gemini 2.0, generating comprehensive comparison tables and statistics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ModelConfidenceAnalyzer:
    """Analyzer for combining and comparing confidence scores across models."""

    def __init__(self):
        """Initialize the analyzer with data paths."""
        self.base_path = "results/"
        self.models = {
            'GPT-4': 'gpt4_perturbation_results.xlsx',
            'Claude Opus 4': 'claude_opus_batch_perturbation_results.xlsx',
            'Gemini 2.0': 'gemini_perturbation_results.xlsx'
        }
        self.data = {}
        self.combined_df = None

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load perturbation results for all three models."""
        print("Loading data for all models...")

        for model_name, file_name in self.models.items():
            file_path = os.path.join(self.base_path, file_name)

            if os.path.exists(file_path):
                print(f"  Loading {model_name} from {file_path}")
                df = pd.read_excel(file_path)

                # Standardize column names
                df['Model_Name'] = model_name

                # Ensure we have confidence value column
                if 'Confidence Value' in df.columns:
                    df['Confidence'] = df['Confidence Value']
                elif 'confidence_value' in df.columns:
                    df['Confidence'] = df['confidence_value']

                self.data[model_name] = df
                print(f"    Loaded {len(df)} records for {model_name}")
            else:
                print(f"  Warning: File not found for {model_name}: {file_path}")
                print(f"    You may need to run perturb_prompts_gpt.py first")

        return self.data

    def combine_confidence_scores(self) -> pd.DataFrame:
        """
        Combine confidence scores from all models into a single DataFrame.
        Returns a DataFrame with one row per perturbation showing all model scores.
        """
        if not self.data:
            self.load_data()

        print("\nCombining confidence scores across models...")

        combined_results = []

        # Get unique perturbations (assuming all models evaluated same perturbations)
        if self.data:
            first_model_data = list(self.data.values())[0]
            unique_prompts = first_model_data[['Original Main Part', 'Rephrased Main Part']].drop_duplicates()

            for _, row in unique_prompts.iterrows():
                original = row['Original Main Part']
                rephrased = row['Rephrased Main Part']

                result_row = {
                    'Original Prompt': original,
                    'Perturbation': rephrased
                }

                # Get confidence for each model
                for model_name, df in self.data.items():
                    mask = (df['Original Main Part'] == original) & (df['Rephrased Main Part'] == rephrased)
                    model_data = df[mask]

                    if not model_data.empty:
                        confidence = model_data['Confidence'].iloc[0]
                        result_row[f'{model_name}_Confidence'] = confidence
                    else:
                        result_row[f'{model_name}_Confidence'] = np.nan

                combined_results.append(result_row)

        self.combined_df = pd.DataFrame(combined_results)

        # Add mean and std across models for each perturbation
        confidence_cols = [col for col in self.combined_df.columns if '_Confidence' in col]
        self.combined_df['Mean_Confidence'] = self.combined_df[confidence_cols].mean(axis=1)
        self.combined_df['Std_Confidence'] = self.combined_df[confidence_cols].std(axis=1)
        self.combined_df['Max_Confidence'] = self.combined_df[confidence_cols].max(axis=1)
        self.combined_df['Min_Confidence'] = self.combined_df[confidence_cols].min(axis=1)
        self.combined_df['Range_Confidence'] = self.combined_df['Max_Confidence'] - self.combined_df['Min_Confidence']

        print(f"Combined {len(self.combined_df)} unique perturbations")

        return self.combined_df

    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics for each model.
        """
        if self.combined_df is None:
            self.combine_confidence_scores()

        print("\nGenerating summary statistics...")

        summary_stats = []

        for model_name in self.models.keys():
            col_name = f'{model_name}_Confidence'

            if col_name in self.combined_df.columns:
                confidence_values = self.combined_df[col_name].dropna()

                if len(confidence_values) > 0:
                    # Basic statistics
                    stats_dict = {
                        'Model': model_name,
                        'N': len(confidence_values),
                        'Mean': confidence_values.mean(),
                        'Std Dev': confidence_values.std(),
                        'Median': confidence_values.median(),
                        'Min': confidence_values.min(),
                        'Max': confidence_values.max(),
                        'Q1 (25%)': confidence_values.quantile(0.25),
                        'Q3 (75%)': confidence_values.quantile(0.75),
                        'IQR': confidence_values.quantile(0.75) - confidence_values.quantile(0.25),
                        'Skewness': confidence_values.skew(),
                        'Kurtosis': confidence_values.kurtosis(),
                        'CV (%)': (confidence_values.std() / confidence_values.mean()) * 100 if confidence_values.mean() != 0 else np.nan
                    }

                    # Confidence intervals
                    ci_95 = stats.t.interval(0.95, len(confidence_values)-1,
                                            loc=confidence_values.mean(),
                                            scale=stats.sem(confidence_values))
                    stats_dict['95% CI Lower'] = ci_95[0]
                    stats_dict['95% CI Upper'] = ci_95[1]

                    # Distribution of confidence levels
                    stats_dict['% Low (0-33)'] = (confidence_values <= 33).mean() * 100
                    stats_dict['% Medium (34-66)'] = ((confidence_values > 33) & (confidence_values <= 66)).mean() * 100
                    stats_dict['% High (67-100)'] = (confidence_values > 66).mean() * 100

                    # Extreme values
                    stats_dict['% at 0'] = (confidence_values == 0).mean() * 100
                    stats_dict['% at 100'] = (confidence_values == 100).mean() * 100
                    stats_dict['% at 50'] = (confidence_values == 50).mean() * 100

                    summary_stats.append(stats_dict)

        summary_df = pd.DataFrame(summary_stats)

        # Add overall statistics across all models
        all_confidences = []
        for model_name in self.models.keys():
            col_name = f'{model_name}_Confidence'
            if col_name in self.combined_df.columns:
                all_confidences.extend(self.combined_df[col_name].dropna().tolist())

        if all_confidences:
            all_confidences = np.array(all_confidences)
            overall_stats = {
                'Model': 'Overall (All Models)',
                'N': len(all_confidences),
                'Mean': all_confidences.mean(),
                'Std Dev': all_confidences.std(),
                'Median': np.median(all_confidences),
                'Min': all_confidences.min(),
                'Max': all_confidences.max(),
                'Q1 (25%)': np.percentile(all_confidences, 25),
                'Q3 (75%)': np.percentile(all_confidences, 75),
                'IQR': np.percentile(all_confidences, 75) - np.percentile(all_confidences, 25),
                'Skewness': stats.skew(all_confidences),
                'Kurtosis': stats.kurtosis(all_confidences),
                'CV (%)': (all_confidences.std() / all_confidences.mean()) * 100 if all_confidences.mean() != 0 else np.nan
            }

            ci_95 = stats.t.interval(0.95, len(all_confidences)-1,
                                    loc=all_confidences.mean(),
                                    scale=stats.sem(all_confidences))
            overall_stats['95% CI Lower'] = ci_95[0]
            overall_stats['95% CI Upper'] = ci_95[1]

            overall_stats['% Low (0-33)'] = (all_confidences <= 33).mean() * 100
            overall_stats['% Medium (34-66)'] = ((all_confidences > 33) & (all_confidences <= 66)).mean() * 100
            overall_stats['% High (67-100)'] = (all_confidences > 66).mean() * 100

            overall_stats['% at 0'] = (all_confidences == 0).mean() * 100
            overall_stats['% at 100'] = (all_confidences == 100).mean() * 100
            overall_stats['% at 50'] = (all_confidences == 50).mean() * 100

            summary_df = pd.concat([summary_df, pd.DataFrame([overall_stats])], ignore_index=True)

        return summary_df

    def generate_per_prompt_statistics(self) -> pd.DataFrame:
        """
        Generate statistics grouped by original prompt.
        """
        if self.combined_df is None:
            self.combine_confidence_scores()

        print("\nGenerating per-prompt statistics...")

        prompt_stats = []

        for original_prompt in self.combined_df['Original Prompt'].unique():
            prompt_data = self.combined_df[self.combined_df['Original Prompt'] == original_prompt]

            stats_dict = {
                'Original Prompt': original_prompt[:50] + '...' if len(original_prompt) > 50 else original_prompt,
                'N Perturbations': len(prompt_data)
            }

            for model_name in self.models.keys():
                col_name = f'{model_name}_Confidence'
                if col_name in prompt_data.columns:
                    values = prompt_data[col_name].dropna()
                    if len(values) > 0:
                        stats_dict[f'{model_name} Mean'] = values.mean()
                        stats_dict[f'{model_name} Std'] = values.std()
                        stats_dict[f'{model_name} Range'] = values.max() - values.min()

            # Cross-model statistics
            stats_dict['Avg Cross-Model Std'] = prompt_data['Std_Confidence'].mean()
            stats_dict['Max Cross-Model Range'] = prompt_data['Range_Confidence'].max()

            prompt_stats.append(stats_dict)

        return pd.DataFrame(prompt_stats)

    def create_latex_tables(self, output_dir: str = 'results/combined_analysis'):
        """
        Create LaTeX formatted tables for publication.
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\nGenerating LaTeX tables...")

        # Get summary statistics
        summary_df = self.generate_summary_statistics()
        per_prompt_df = self.generate_per_prompt_statistics()

        latex_content = []

        # Table 1: Summary Statistics
        latex_content.append("% Table 1: Summary Statistics for Verbalized Confidence Scores")
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Summary Statistics for Verbalized Confidence Scores Across Models}")
        latex_content.append("\\label{tab:summary_stats}")
        latex_content.append("\\begin{tabular}{lccccccc}")
        latex_content.append("\\toprule")
        latex_content.append("Model & N & Mean & Std Dev & Median & Q1 & Q3 & 95\\% CI \\\\")
        latex_content.append("\\midrule")

        for _, row in summary_df.iterrows():
            model = row['Model']
            n = int(row['N'])
            mean = row['Mean']
            std = row['Std Dev']
            median = row['Median']
            q1 = row['Q1 (25%)']
            q3 = row['Q3 (75%)']
            ci_lower = row['95% CI Lower']
            ci_upper = row['95% CI Upper']

            if model == 'Overall (All Models)':
                latex_content.append("\\midrule")

            latex_content.append(f"{model} & {n} & {mean:.1f} & {std:.1f} & {median:.1f} & "
                               f"{q1:.1f} & {q3:.1f} & [{ci_lower:.1f}, {ci_upper:.1f}] \\\\")

        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        latex_content.append("")

        # Table 2: Distribution of Confidence Levels
        latex_content.append("% Table 2: Distribution of Confidence Levels")
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Distribution of Verbalized Confidence Levels (\\%)}")
        latex_content.append("\\label{tab:confidence_distribution}")
        latex_content.append("\\begin{tabular}{lcccccc}")
        latex_content.append("\\toprule")
        latex_content.append("Model & Low (0-33) & Medium (34-66) & High (67-100) & At 0 & At 50 & At 100 \\\\")
        latex_content.append("\\midrule")

        for _, row in summary_df.iterrows():
            model = row['Model']
            low = row['% Low (0-33)']
            medium = row['% Medium (34-66)']
            high = row['% High (67-100)']
            at_0 = row['% at 0']
            at_50 = row['% at 50']
            at_100 = row['% at 100']

            if model == 'Overall (All Models)':
                latex_content.append("\\midrule")

            latex_content.append(f"{model} & {low:.1f} & {medium:.1f} & {high:.1f} & "
                               f"{at_0:.1f} & {at_50:.1f} & {at_100:.1f} \\\\")

        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")
        latex_content.append("")

        # Table 3: Per-Prompt Comparison (first 5 prompts)
        latex_content.append("% Table 3: Per-Prompt Confidence Comparison")
        latex_content.append("\\begin{table}[htbp]")
        latex_content.append("\\centering")
        latex_content.append("\\caption{Mean Confidence Scores by Original Prompt}")
        latex_content.append("\\label{tab:per_prompt}")
        latex_content.append("\\begin{tabular}{p{5cm}cccc}")
        latex_content.append("\\toprule")
        latex_content.append("Prompt & GPT-4 & Claude & Gemini & Cross-Model Std \\\\")
        latex_content.append("\\midrule")

        for idx, row in per_prompt_df.head(5).iterrows():
            prompt = row['Original Prompt']
            gpt_mean = row.get('GPT-4 Mean', np.nan)
            claude_mean = row.get('Claude Opus 4 Mean', np.nan)
            gemini_mean = row.get('Gemini 2.0 Mean', np.nan)
            cross_std = row['Avg Cross-Model Std']

            latex_content.append(f"{prompt} & {gpt_mean:.1f} & {claude_mean:.1f} & "
                               f"{gemini_mean:.1f} & {cross_std:.1f} \\\\")
            if idx < 4:
                latex_content.append("\\addlinespace")

        latex_content.append("\\bottomrule")
        latex_content.append("\\end{tabular}")
        latex_content.append("\\end{table}")

        # Save LaTeX file
        latex_file = os.path.join(output_dir, 'confidence_tables.tex')
        with open(latex_file, 'w') as f:
            f.write('\n'.join(latex_content))

        print(f"LaTeX tables saved to {latex_file}")

        return latex_content

    def create_visualizations(self, output_dir: str = 'results/combined_analysis'):
        """
        Create comprehensive visualizations comparing all models.
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.combined_df is None:
            self.combine_confidence_scores()

        print("\nCreating visualizations...")

        # Prepare data for visualization
        plot_data = []
        for model_name in self.models.keys():
            col_name = f'{model_name}_Confidence'
            if col_name in self.combined_df.columns:
                values = self.combined_df[col_name].dropna()
                for val in values:
                    plot_data.append({'Model': model_name, 'Confidence': val})

        plot_df = pd.DataFrame(plot_data)

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Verbalized Confidence Score Comparison Across Models', fontsize=16, y=1.02)

        # 1. Violin plot
        ax = axes[0, 0]
        sns.violinplot(data=plot_df, x='Model', y='Confidence', ax=ax, inner='box')
        ax.set_title('Distribution Comparison (Violin Plot)')
        ax.set_ylabel('Confidence Score')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # 2. Box plot
        ax = axes[0, 1]
        sns.boxplot(data=plot_df, x='Model', y='Confidence', ax=ax)
        ax.set_title('Distribution Comparison (Box Plot)')
        ax.set_ylabel('Confidence Score')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # 3. Histogram overlay
        ax = axes[0, 2]
        for model_name in self.models.keys():
            col_name = f'{model_name}_Confidence'
            if col_name in self.combined_df.columns:
                values = self.combined_df[col_name].dropna()
                ax.hist(values, bins=20, alpha=0.5, label=model_name, density=True)
        ax.set_title('Normalized Histogram Overlay')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Mean with error bars
        ax = axes[1, 0]
        summary_stats = self.generate_summary_statistics()
        model_stats = summary_stats[summary_stats['Model'] != 'Overall (All Models)']

        x_pos = np.arange(len(model_stats))
        means = model_stats['Mean'].values
        stds = model_stats['Std Dev'].values

        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_stats['Model'].values, rotation=0)
        ax.set_title('Mean Confidence with Standard Deviation')
        ax.set_ylabel('Confidence Score')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 2, f'{mean:.1f}', ha='center', va='bottom')

        # 5. CDF comparison
        ax = axes[1, 1]
        for model_name in self.models.keys():
            col_name = f'{model_name}_Confidence'
            if col_name in self.combined_df.columns:
                values = self.combined_df[col_name].dropna().sort_values()
                cdf = np.arange(1, len(values) + 1) / len(values)
                ax.plot(values, cdf, label=model_name, linewidth=2)

        ax.set_title('Cumulative Distribution Function')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Cumulative Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)

        # 6. Scatter plot of model agreement
        ax = axes[1, 2]
        if len(self.models) >= 2:
            model_names = list(self.models.keys())
            col1 = f'{model_names[0]}_Confidence'
            col2 = f'{model_names[1]}_Confidence'

            if col1 in self.combined_df.columns and col2 in self.combined_df.columns:
                valid_data = self.combined_df[[col1, col2]].dropna()
                ax.scatter(valid_data[col1], valid_data[col2], alpha=0.5, s=20)

                # Add diagonal line
                ax.plot([0, 100], [0, 100], 'r--', alpha=0.5, label='Perfect Agreement')

                # Calculate correlation
                corr = valid_data[col1].corr(valid_data[col2])
                ax.set_title(f'{model_names[0]} vs {model_names[1]} (r={corr:.3f})')
                ax.set_xlabel(f'{model_names[0]} Confidence')
                ax.set_ylabel(f'{model_names[1]} Confidence')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)

        plt.tight_layout()

        # Save figure
        fig_path = os.path.join(output_dir, 'confidence_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {fig_path}")

        plt.show()

        return fig

    def save_results(self, output_dir: str = 'results/combined_analysis'):
        """
        Save all analysis results to files.
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\nSaving all results...")

        # Save combined confidence scores
        if self.combined_df is not None:
            combined_file = os.path.join(output_dir, 'combined_confidence_scores.csv')
            self.combined_df.to_csv(combined_file, index=False)
            print(f"  Combined scores saved to {combined_file}")

        # Save summary statistics
        summary_df = self.generate_summary_statistics()
        summary_file = os.path.join(output_dir, 'summary_statistics.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"  Summary statistics saved to {summary_file}")

        # Save per-prompt statistics
        per_prompt_df = self.generate_per_prompt_statistics()
        per_prompt_file = os.path.join(output_dir, 'per_prompt_statistics.csv')
        per_prompt_df.to_csv(per_prompt_file, index=False)
        print(f"  Per-prompt statistics saved to {per_prompt_file}")

        # Create and save LaTeX tables
        self.create_latex_tables(output_dir)

        # Create and save visualizations
        self.create_visualizations(output_dir)

        print(f"\nAll results saved to {output_dir}")

    def print_summary(self):
        """
        Print a formatted summary of the analysis.
        """
        summary_df = self.generate_summary_statistics()

        print("\n" + "="*80)
        print("VERBALIZED CONFIDENCE ANALYSIS SUMMARY")
        print("="*80)

        print("\nOverall Statistics:")
        print("-"*40)

        overall = summary_df[summary_df['Model'] == 'Overall (All Models)']
        if not overall.empty:
            row = overall.iloc[0]
            print(f"Total Samples: {int(row['N'])}")
            print(f"Mean Confidence: {row['Mean']:.2f} +/- {row['Std Dev']:.2f}")
            print(f"Median: {row['Median']:.2f}")
            print(f"Range: [{row['Min']:.0f}, {row['Max']:.0f}]")
            print(f"95% CI: [{row['95% CI Lower']:.2f}, {row['95% CI Upper']:.2f}]")

        print("\nModel Comparison:")
        print("-"*40)

        model_stats = summary_df[summary_df['Model'] != 'Overall (All Models)']

        # Create comparison table
        print(f"{'Model':<15} {'Mean':>10} {'Std':>10} {'Median':>10} {'N':>10}")
        print("-"*55)

        for _, row in model_stats.iterrows():
            print(f"{row['Model']:<15} {row['Mean']:>10.2f} {row['Std Dev']:>10.2f} "
                  f"{row['Median']:>10.2f} {int(row['N']):>10}")

        print("\nConfidence Level Distribution (%):")
        print("-"*40)

        print(f"{'Model':<15} {'Low (0-33)':>15} {'Medium (34-66)':>15} {'High (67-100)':>15}")
        print("-"*60)

        for _, row in model_stats.iterrows():
            print(f"{row['Model']:<15} {row['% Low (0-33)']:>15.1f} "
                  f"{row['% Medium (34-66)']:>15.1f} {row['% High (67-100)']:>15.1f}")

        print("\nKey Findings:")
        print("-"*40)

        # Find model with highest/lowest mean confidence
        highest_mean_idx = model_stats['Mean'].idxmax()
        lowest_mean_idx = model_stats['Mean'].idxmin()
        highest_model = model_stats.loc[highest_mean_idx, 'Model']
        lowest_model = model_stats.loc[lowest_mean_idx, 'Model']

        print(f"* Highest mean confidence: {highest_model} ({model_stats.loc[highest_mean_idx, 'Mean']:.2f})")
        print(f"* Lowest mean confidence: {lowest_model} ({model_stats.loc[lowest_mean_idx, 'Mean']:.2f})")

        # Find model with highest variability
        highest_std_idx = model_stats['Std Dev'].idxmax()
        highest_std_model = model_stats.loc[highest_std_idx, 'Model']
        print(f"* Highest variability: {highest_std_model} (std = {model_stats.loc[highest_std_idx, 'Std Dev']:.2f})")

        # Model correlations if available
        if self.combined_df is not None:
            print("\nModel Correlations:")
            print("-"*40)

            confidence_cols = [col for col in self.combined_df.columns if '_Confidence' in col]
            if len(confidence_cols) >= 2:
                corr_matrix = self.combined_df[confidence_cols].corr()

                for i, col1 in enumerate(confidence_cols):
                    for j, col2 in enumerate(confidence_cols):
                        if i < j:
                            model1 = col1.replace('_Confidence', '')
                            model2 = col2.replace('_Confidence', '')
                            corr = corr_matrix.loc[col1, col2]
                            if not np.isnan(corr):
                                print(f"* {model1} vs {model2}: r = {corr:.3f}")

        print("\n" + "="*80)


def main():
    """Main function to run the combined analysis."""
    analyzer = ModelConfidenceAnalyzer()

    # Load data for all models
    analyzer.load_data()

    # Combine confidence scores
    combined_df = analyzer.combine_confidence_scores()

    # Generate and print summary
    analyzer.print_summary()

    # Save all results
    analyzer.save_results()

    return analyzer

if __name__ == "__main__":
    analyzer = main()