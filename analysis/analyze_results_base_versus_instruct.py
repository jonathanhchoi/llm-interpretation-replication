import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set up the directory paths
BASE_DIR = "G:/My Drive/Computational/llm_interpretation"
RESULTS_DIR = os.path.join(BASE_DIR, "analysis_results")

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set global font sizes for all plots
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 22
})

# Read the data
df = pd.read_csv(os.path.join(BASE_DIR, "model_comparison_results.csv"))

# Print unique values in relevant columns for debugging
print("\nUnique model families:", df['model_family'].unique())
print("\nUnique base_or_instruct values:", df['base_or_instruct'].unique())
print("\nUnique models:", df['model'].unique())

# Drop Mistral model family
df = df[df['model_family'] != 'mistral']
print("\nModel families after dropping Mistral:", df['model_family'].unique())

# Function to process data for a model pair
def process_model_pair(df, base_model, instruct_model):
    # Get data for both models
    base_data = df[df['model'] == base_model].copy()
    instruct_data = df[df['model'] == instruct_model].copy()
    
    # Merge the data on prompt
    paired_data = pd.merge(base_data, instruct_data, on='prompt', suffixes=('_base', '_instruct'))
    
    # Drop rows where either model has zero probabilities
    valid_rows = (
        (paired_data['yes_prob_base'] > 0) & 
        (paired_data['no_prob_base'] > 0) & 
        (paired_data['yes_prob_instruct'] > 0) & 
        (paired_data['no_prob_instruct'] > 0)
    )
    
    # Calculate relative probability for base and instruct models
    paired_data['rel_prob_base'] = paired_data['yes_prob_base'] / (paired_data['yes_prob_base'] + paired_data['no_prob_base'])
    paired_data['rel_prob_instruct'] = paired_data['yes_prob_instruct'] / (paired_data['yes_prob_instruct'] + paired_data['no_prob_instruct'])
    
    return paired_data[valid_rows]

# Get unique model families
model_families = df['model_family'].unique()

# Create figure for relative probability magnitude differences
plt.figure(figsize=(15, 8))
avg_magnitude_diffs = []
model_pair_names = []

# Create figure for per-prompt differences
plt.figure(figsize=(15, 10))
all_prompt_diffs = []
all_prompts = []
all_model_families = []

# List to store statistics for each model family
model_statistics = []

# Process each model family
for family in model_families:
    print(f"\nProcessing family: {family}")
    
    # Get base and instruct models for this family
    base_models = df[(df['model_family'] == family) & (df['base_or_instruct'] == 'base')]['model']
    instruct_models = df[(df['model_family'] == family) & (df['base_or_instruct'] == 'instruct')]['model']
    
    if len(base_models) == 0:
        print(f"No base model found for family {family}")
        continue
    if len(instruct_models) == 0:
        print(f"No instruct model found for family {family}")
        continue
        
    base_model = base_models.iloc[0]
    instruct_model = instruct_models.iloc[0]
    
    print(f"Analyzing {family} models:")
    print(f"Base: {base_model}")
    print(f"Instruct: {instruct_model}")
    
    # Process the pair
    paired_data = process_model_pair(df, base_model, instruct_model)
    
    if len(paired_data) == 0:
        print(f"No valid data for {family} after filtering zero probabilities")
        continue
    
    # Use relative probability instead of log odds
    rel_prob_base = paired_data['rel_prob_base']
    rel_prob_instruct = paired_data['rel_prob_instruct']
    
    # Calculate correlation
    correlation, p_value = stats.pearsonr(rel_prob_base, rel_prob_instruct)
    print(f"Correlation between base and instruct relative probabilities: {correlation:.3f} (p={p_value:.3f})")
    print(f"Number of valid prompt pairs: {len(paired_data)}")
    
    # Calculate magnitude differences
    magnitude_diff = rel_prob_instruct - rel_prob_base
    avg_magnitude_diffs.append(np.mean(magnitude_diff))
    model_pair_names.append(family)
    
    # Calculate statistics for current model family
    mean_diff = np.mean(magnitude_diff)
    std_diff = np.std(magnitude_diff)
    lower_percentile = np.percentile(magnitude_diff, 2.5)
    upper_percentile = np.percentile(magnitude_diff, 97.5)
    ci_width = upper_percentile - lower_percentile
    
    # Save statistics for this model family
    model_statistics.append({
        'Model_Family': family,
        'Mean': mean_diff,
        'Std_Dev': std_diff,
        'Lower_CI_95': lower_percentile,
        'Upper_CI_95': upper_percentile,
        'CI_Width': ci_width,
        'Num_Samples': len(magnitude_diff)
    })
    
    # Store prompt-level differences
    differences = rel_prob_instruct - rel_prob_base
    all_prompt_diffs.extend(differences)
    all_prompts.extend(paired_data['prompt'])
    all_model_families.extend([family] * len(differences))

if not model_pair_names:  # Check if we have any valid data
    print("\nNo valid model pairs found for analysis!")
    exit()

# Plot 1: Average magnitude differences
plt.figure(1)
plt.bar(model_pair_names, avg_magnitude_diffs)
plt.xticks(rotation=45, ha='right')
plt.title('Average Difference in Relative Probability\n(Instruct - Base)')
plt.ylabel('Difference in Relative Probability')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'rel_prob_differences.png'))

# Plot 2: Per-prompt differences with violin plots
prompt_diff_df = pd.DataFrame({
    'Difference': all_prompt_diffs,
    'Prompt': all_prompts,
    'Model Family': all_model_families
})

# Create a new figure for the violin plot
plt.figure(figsize=(15, 10))
ax = plt.subplot(111)

# Define colors for each model family
unique_families = prompt_diff_df['Model Family'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_families)))

# Plot violin plots and jittered points for each model family
for idx, family in enumerate(unique_families):
    family_data = prompt_diff_df[prompt_diff_df['Model Family'] == family]
    
    # Calculate statistics
    mean_diff = family_data['Difference'].mean()
    lower_percentile = np.percentile(family_data['Difference'], 2.5)
    upper_percentile = np.percentile(family_data['Difference'], 97.5)
    
    # Add violin plot (with lower alpha to not obscure points)
    violin_parts = ax.violinplot([family_data['Difference']], [idx + 1], 
                                widths=0.3, showmeans=False, showmedians=False, showextrema=False)
    for pc in violin_parts['bodies']:
        pc.set_facecolor(colors[idx])
        pc.set_edgecolor('none')
        pc.set_alpha(0.3)
    
    # Add jittered points
    x_jittered = np.random.normal(idx + 1, 0.08, size=len(family_data))
    plt.scatter(x_jittered, family_data['Difference'], alpha=0.4, s=30, 
               color=colors[idx], label=f'{family}' if idx == 0 else "")
    
    # Add mean point
    plt.scatter(idx + 1, mean_diff, color='black', s=80, zorder=5)
    
    # Add error bars for 95% CI
    plt.plot([idx + 1, idx + 1], [lower_percentile, upper_percentile], 
            color='black', linewidth=2, zorder=4)
    
    # Add caps to the error bars
    cap_width = 0.1
    plt.plot([idx + 1 - cap_width, idx + 1 + cap_width], 
            [lower_percentile, lower_percentile], color='black', linewidth=2, zorder=4)
    plt.plot([idx + 1 - cap_width, idx + 1 + cap_width], 
            [upper_percentile, upper_percentile], color='black', linewidth=2, zorder=4)

# Add a horizontal line at 0 for reference
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Set the x-ticks and labels
plt.xticks(range(1, len(unique_families) + 1), unique_families, rotation=45, ha='right')

# Add labels
plt.ylabel('Relative Probability Difference (Instruct - Base)')

# Create custom legend elements
custom_legend = []
for idx, family in enumerate(unique_families):
    custom_legend.append(plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=colors[idx], markersize=10, 
                                   label=f"{family}"))

# Add the legend
plt.legend(handles=custom_legend, fontsize=16, loc='best')

# Adjust layout and save
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'prompt_rel_prob_differences.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create a heatmap of differences by prompt
pivot_df = prompt_diff_df.pivot_table(
    index='Prompt',
    columns='Model Family',
    values='Difference',
    aggfunc='mean'
)

# Print the number of rows in the pivot_df (number of prompts)
print(f"\nNumber of rows in heatmap (unique prompts): {len(pivot_df)}")

# Increased width from 12 to 16 to prevent x-axis labels from being smushed together
# Increased height by 1.5 times to accommodate all 49 rows properly
plt.figure(figsize=(18, len(pivot_df) * 0.4))
sns.heatmap(pivot_df, center=0, cmap='RdBu_r', fmt='.2f')
# Removed title as requested
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'prompt_rel_prob_heatmap.png'))

# Create a DataFrame with model statistics
model_stats_df = pd.DataFrame(model_statistics)

# Save model statistics to CSV
model_stats_df.to_csv(os.path.join(RESULTS_DIR, 'model_rel_prob_statistics.csv'), index=False)

# Save the processed data
prompt_diff_df.to_csv(os.path.join(RESULTS_DIR, 'prompt_rel_prob_differences.csv'), index=False)
pivot_df.to_csv(os.path.join(RESULTS_DIR, 'prompt_rel_prob_heatmap_data.csv'))

print("\nAnalysis complete. Files saved in:", RESULTS_DIR)
print("Generated files:")
print("1. rel_prob_differences.png - Bar plot of average relative probability differences")
print("2. prompt_rel_prob_differences.png - Violin plot of differences by model family with jittered points and 95% CI")
print("3. prompt_rel_prob_heatmap.png - Heatmap of differences by prompt and model family")
print("4. prompt_rel_prob_differences.csv - Raw differences data")
print("5. prompt_rel_prob_heatmap_data.csv - Processed heatmap data")
print("6. model_rel_prob_statistics.csv - Statistics per model family (mean, std, 95% CI, CI width)")

# =============================================================================
# OVERALL MAE EFFECT ACROSS ALL THREE MODELS (with bootstrap CI and p-value)
# =============================================================================
print("\n" + "=" * 80)
print("OVERALL MAE EFFECT ACROSS ALL MODEL FAMILIES")
print("(Bootstrap resampling per question per model)")
print("=" * 80)

# Reshape prompt_diff_df to have one row per (prompt, model_family) combination
# with the absolute difference (MAE contribution)
prompt_diff_df['Abs_Difference'] = prompt_diff_df['Difference'].abs()

# Get unique prompts and model families
unique_prompts = prompt_diff_df['Prompt'].unique()
unique_model_families = prompt_diff_df['Model Family'].unique()
n_prompts = len(unique_prompts)
n_models = len(unique_model_families)

print(f"\nNumber of prompts: {n_prompts}")
print(f"Number of model families: {n_models}")
print(f"Total observations: {len(prompt_diff_df)}")

# Create a pivot table: rows = prompts, columns = model families, values = differences
pivot_for_bootstrap = prompt_diff_df.pivot_table(
    index='Prompt',
    columns='Model Family',
    values='Difference',
    aggfunc='mean'
).dropna()

print(f"Prompts with data for all models: {len(pivot_for_bootstrap)}")

# Bootstrap parameters
n_bootstrap = 10000
np.random.seed(42)  # For reproducibility

# Store bootstrap results
bootstrap_mae_overall = []  # MAE across all models and questions
bootstrap_mean_diff = []    # Mean signed difference (to test directionality)

for b in range(n_bootstrap):
    # Resample prompts with replacement
    sampled_prompts = np.random.choice(pivot_for_bootstrap.index, size=len(pivot_for_bootstrap), replace=True)

    # Get the differences for sampled prompts (all models)
    sampled_data = pivot_for_bootstrap.loc[sampled_prompts]

    # Calculate MAE: mean of absolute differences across all models and sampled prompts
    all_diffs = sampled_data.values.flatten()
    bootstrap_mae_overall.append(np.mean(np.abs(all_diffs)))
    bootstrap_mean_diff.append(np.mean(all_diffs))

# Convert to arrays
bootstrap_mae_overall = np.array(bootstrap_mae_overall)
bootstrap_mean_diff = np.array(bootstrap_mean_diff)

# Calculate overall observed statistics
observed_mae = prompt_diff_df['Abs_Difference'].mean()
observed_mean_diff = prompt_diff_df['Difference'].mean()

# Calculate 95% confidence intervals (percentile method)
mae_ci_lower = np.percentile(bootstrap_mae_overall, 2.5)
mae_ci_upper = np.percentile(bootstrap_mae_overall, 97.5)

mean_diff_ci_lower = np.percentile(bootstrap_mean_diff, 2.5)
mean_diff_ci_upper = np.percentile(bootstrap_mean_diff, 97.5)

# Calculate p-value for MAE > 0 (test if there's any effect)
# Since MAE is always >= 0, we use a permutation test to see if the MAE
# is significantly greater than what we'd expect under the null hypothesis
# that instruct and base models have no systematic difference

# Permutation test: randomly flip signs of differences
n_permutations = 10000
perm_maes = []
all_diffs_observed = pivot_for_bootstrap.values.flatten()

for _ in range(n_permutations):
    # Randomly flip signs (simulates null hypothesis of no directional effect)
    signs = np.random.choice([-1, 1], size=len(all_diffs_observed))
    perm_diffs = all_diffs_observed * signs
    perm_maes.append(np.mean(np.abs(perm_diffs)))

perm_maes = np.array(perm_maes)

# P-value: proportion of permuted MAEs >= observed MAE
p_value_mae = np.mean(perm_maes >= observed_mae)

# P-value for mean difference != 0 (two-sided test)
# Under null, mean difference should be centered around 0
p_value_mean_diff = np.mean(np.abs(bootstrap_mean_diff - np.mean(bootstrap_mean_diff)) >= np.abs(observed_mean_diff - np.mean(bootstrap_mean_diff)))
# Alternative: use proportion of bootstrap samples crossing zero
if observed_mean_diff > 0:
    p_value_mean_diff_bootstrap = 2 * np.mean(bootstrap_mean_diff <= 0)
else:
    p_value_mean_diff_bootstrap = 2 * np.mean(bootstrap_mean_diff >= 0)

# Print results
print("\n--- Overall MAE (Mean Absolute Error) ---")
print(f"MAE (Instruct - Base): {observed_mae:.4f}")
print(f"95% CI: [{mae_ci_lower:.4f}, {mae_ci_upper:.4f}]")
print(f"P-value (permutation test): {p_value_mae:.4f}")

print("\n--- Mean Signed Difference ---")
print(f"Mean Difference (Instruct - Base): {observed_mean_diff:.4f}")
print(f"95% CI: [{mean_diff_ci_lower:.4f}, {mean_diff_ci_upper:.4f}]")
print(f"P-value (bootstrap, two-sided): {p_value_mean_diff_bootstrap:.4f}")

# Interpretation
print("\n--- Interpretation ---")
if p_value_mae < 0.05:
    print(f"The MAE of {observed_mae:.4f} is statistically significant (p = {p_value_mae:.4f}).")
    print("Instruction tuning does have a meaningful effect on model responses.")
else:
    print(f"The MAE is not statistically significant (p = {p_value_mae:.4f}).")

if mean_diff_ci_lower > 0:
    print(f"Instruction-tuned models consistently produce HIGHER relative probabilities")
    print(f"(more likely to say 'Yes') than base models.")
elif mean_diff_ci_upper < 0:
    print(f"Instruction-tuned models consistently produce LOWER relative probabilities")
    print(f"(more likely to say 'No') than base models.")
else:
    print(f"The direction of the effect is not consistent across all models/questions.")
    print(f"(95% CI includes zero: [{mean_diff_ci_lower:.4f}, {mean_diff_ci_upper:.4f}])")

# Per-model family breakdown
print("\n--- Per-Model Family MAE ---")
for family in unique_model_families:
    family_data = prompt_diff_df[prompt_diff_df['Model Family'] == family]
    family_mae = family_data['Abs_Difference'].mean()
    family_mean_diff = family_data['Difference'].mean()
    print(f"{family}: MAE = {family_mae:.4f}, Mean Diff = {family_mean_diff:.4f}")

# Save overall statistics to CSV
overall_stats = {
    'Metric': ['MAE', 'MAE_CI_Lower', 'MAE_CI_Upper', 'MAE_P_Value',
               'Mean_Diff', 'Mean_Diff_CI_Lower', 'Mean_Diff_CI_Upper', 'Mean_Diff_P_Value',
               'N_Prompts', 'N_Models', 'N_Bootstrap'],
    'Value': [observed_mae, mae_ci_lower, mae_ci_upper, p_value_mae,
              observed_mean_diff, mean_diff_ci_lower, mean_diff_ci_upper, p_value_mean_diff_bootstrap,
              n_prompts, n_models, n_bootstrap]
}
overall_stats_df = pd.DataFrame(overall_stats)
overall_stats_df.to_csv(os.path.join(RESULTS_DIR, 'overall_mae_statistics.csv'), index=False)

print(f"\n7. overall_mae_statistics.csv - Overall MAE effect with bootstrap CI and p-value") 