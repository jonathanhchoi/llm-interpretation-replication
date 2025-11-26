"""
Power Analysis for LLM Evaluation Study
Determines sample size needed to detect differences between models and baseline
"""

import numpy as np
from scipy import stats
import pandas as pd

def calculate_required_sample_size(observed_mae_diff, observed_std, alpha=0.05, power=0.80, margin_factor=1.5):
    """
    Calculate required sample size to detect a difference from baseline.

    Uses approximation formula for one-sample t-test sample size calculation.

    Args:
        observed_mae_diff: Observed MAE difference from baseline
        observed_std: Observed standard deviation of differences
        alpha: Significance level (default 0.05)
        power: Desired statistical power (default 0.80)
        margin_factor: Safety margin multiplier (default 1.5 for 50% margin)

    Returns:
        dict: Required sample sizes at different power levels
    """

    # Calculate effect size (Cohen's d)
    effect_size = abs(observed_mae_diff) / observed_std if observed_std > 0 else 0

    # Calculate sample sizes for different power levels
    sample_sizes = {}
    power_levels = [0.70, 0.80, 0.85, 0.90, 0.95]

    for target_power in power_levels:
        if effect_size > 0:
            # One-sample t-test sample size calculation
            # Using approximation: n = (z_alpha/2 + z_beta)^2 / d^2
            # where d is Cohen's d (effect size)

            # Get z-scores for alpha and beta (1-power)
            z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
            z_beta = stats.norm.ppf(target_power)

            # Calculate sample size
            n = ((z_alpha + z_beta) / effect_size) ** 2

            # Add small correction for t-distribution (approximation)
            n = n * (1 + 1/(4*(n-1))) if n > 2 else n

            sample_sizes[f'power_{int(target_power*100)}'] = {
                'raw': int(np.ceil(n)),
                'with_margin': int(np.ceil(n * margin_factor))
            }
        else:
            sample_sizes[f'power_{int(target_power*100)}'] = {
                'raw': np.inf,
                'with_margin': np.inf
            }

    return {
        'effect_size': effect_size,
        'sample_sizes': sample_sizes,
        'observed_mae_diff': observed_mae_diff,
        'observed_std': observed_std
    }

def simulate_power_at_sample_size(mae_diff, std, sample_size, n_simulations=10000, alpha=0.05):
    """
    Simulate the actual power achieved at a given sample size.

    Args:
        mae_diff: True MAE difference from baseline
        std: Standard deviation of differences
        sample_size: Sample size to test
        n_simulations: Number of simulations to run
        alpha: Significance level

    Returns:
        float: Estimated power (proportion of simulations detecting significance)
    """
    np.random.seed(42)
    significant_count = 0

    for _ in range(n_simulations):
        # Generate sample data
        sample = np.random.normal(mae_diff, std, sample_size)

        # One-sample t-test against null hypothesis (difference = 0)
        t_stat, p_value = stats.ttest_1samp(sample, 0)

        if p_value < alpha:
            significant_count += 1

    return significant_count / n_simulations

def main():
    print("=" * 70)
    print("POWER ANALYSIS FOR LLM EVALUATION STUDY")
    print("=" * 70)

    # Current study results (from the output provided)
    current_results = {
        'GPT': {
            'mae': 0.205,
            'mae_std': 0.126,
            'mae_diff_from_50': 0.032,
            'ci_lower': -0.017,
            'ci_upper': 0.082,
            'n': 50
        },
        'GEMINI': {
            'mae': 0.225,
            'mae_std': 0.122,
            'mae_diff_from_50': 0.052,
            'ci_lower': -0.000,
            'ci_upper': 0.103,
            'n': 50
        },
        'Claude': {
            'mae': 0.232,
            'mae_std': 0.129,
            'mae_diff_from_50': 0.059,
            'ci_lower': 0.008,
            'ci_upper': 0.109,
            'n': 50
        },
        'Always_50_baseline': {
            'mae': 0.180,
            'mae_std': 0.106,
            'n': 50
        }
    }

    print("\nCurrent Study Results (N=50 questions):")
    print("-" * 70)

    for model, results in current_results.items():
        if model != 'Always_50_baseline':
            print(f"\n{model}:")
            print(f"  MAE: {results['mae']:.3f} ± {results['mae_std']:.3f}")
            print(f"  Difference from 50% baseline: {results['mae_diff_from_50']:.3f}")
            print(f"  95% CI for difference: [{results['ci_lower']:.3f}, {results['ci_upper']:.3f}]")

            # Check if CI includes zero
            includes_zero = results['ci_lower'] <= 0 <= results['ci_upper']
            if includes_zero:
                print(f"  ⚠️  CI includes zero - not significantly different from baseline")
            else:
                print(f"  ✓ CI excludes zero - significantly different from baseline")

    print(f"\nBaseline (Always 50%):")
    print(f"  MAE: {current_results['Always_50_baseline']['mae']:.3f} ± {current_results['Always_50_baseline']['mae_std']:.3f}")

    # Power analysis for each model
    print("\n" + "=" * 70)
    print("POWER ANALYSIS RESULTS")
    print("=" * 70)

    power_results = {}

    for model in ['GPT', 'GEMINI', 'Claude']:
        results = current_results[model]

        # For power analysis, we use the observed difference and its standard deviation
        # We estimate the standard deviation of differences from the MAE std
        # This is an approximation - ideally we'd have the actual paired differences
        diff_std = results['mae_std']

        power_analysis = calculate_required_sample_size(
            observed_mae_diff=results['mae_diff_from_50'],
            observed_std=diff_std,
            alpha=0.05,
            power=0.80,
            margin_factor=1.5
        )

        power_results[model] = power_analysis

        print(f"\n{model}:")
        print(f"  Observed effect size (Cohen's d): {power_analysis['effect_size']:.3f}")

        if power_analysis['effect_size'] < 0.2:
            print("  (Small effect size - harder to detect)")
        elif power_analysis['effect_size'] < 0.5:
            print("  (Small-to-medium effect size)")
        elif power_analysis['effect_size'] < 0.8:
            print("  (Medium effect size)")
        else:
            print("  (Large effect size)")

        print("\n  Required sample sizes:")
        for power_level, sizes in power_analysis['sample_sizes'].items():
            power_pct = power_level.replace('power_', '')
            if sizes['raw'] != np.inf:
                print(f"    {power_pct}% power: {sizes['raw']} questions")
                print(f"                    {sizes['with_margin']} questions (with 50% margin)")
            else:
                print(f"    {power_pct}% power: Cannot be achieved (effect too small)")

    # Calculate actual power at current sample size
    print("\n" + "=" * 70)
    print("ESTIMATED POWER AT CURRENT SAMPLE SIZE (N=50)")
    print("=" * 70)

    for model in ['GPT', 'GEMINI', 'Claude']:
        results = current_results[model]
        estimated_power = simulate_power_at_sample_size(
            mae_diff=results['mae_diff_from_50'],
            std=results['mae_std'],
            sample_size=50,
            n_simulations=10000
        )
        print(f"{model}: {estimated_power:.1%} power to detect difference from baseline")

    # Recommend minimum sample size
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Find the maximum required sample size across all models for 80% power
    max_required = 0
    max_required_with_margin = 0
    limiting_model = None

    for model, analysis in power_results.items():
        required = analysis['sample_sizes']['power_80']['raw']
        required_margin = analysis['sample_sizes']['power_80']['with_margin']
        if required != np.inf and required > max_required:
            max_required = required
            max_required_with_margin = required_margin
            limiting_model = model

    if max_required > 0:
        print(f"\nTo achieve 80% power for all models:")
        print(f"  Minimum sample size: {max_required} questions")
        print(f"  Recommended (with 50% safety margin): {max_required_with_margin} questions")
        print(f"  Limiting factor: {limiting_model} (smallest effect size)")

        # Calculate sample size for 90% power
        max_required_90 = 0
        max_required_90_margin = 0
        for model, analysis in power_results.items():
            required = analysis['sample_sizes']['power_90']['raw']
            required_margin = analysis['sample_sizes']['power_90']['with_margin']
            if required != np.inf and required > max_required_90:
                max_required_90 = required
                max_required_90_margin = required_margin

        print(f"\nFor more robust results (90% power):")
        print(f"  Minimum sample size: {max_required_90} questions")
        print(f"  Recommended (with 50% safety margin): {max_required_90_margin} questions")

    # Additional considerations
    print("\n" + "=" * 70)
    print("ADDITIONAL CONSIDERATIONS")
    print("=" * 70)

    print("\n1. Effect Size Interpretation:")
    print("   - GPT shows the smallest effect (hardest to detect)")
    print("   - Claude shows the largest effect (easiest to detect)")
    print("   - All effects are relatively small, requiring larger samples")

    print("\n2. Current Study Status:")
    print("   - Only Claude shows significant difference (CI excludes zero)")
    print("   - GPT and Gemini CIs include zero (not significant at α=0.05)")
    print("   - Study is underpowered for GPT and Gemini")

    print("\n3. Sample Size Planning:")
    print("   - Consider the research goals and resources available")
    print("   - More questions improve power but increase cost/time")
    print("   - Consider stratified sampling across question types")

    print("\n4. Alternative Approaches:")
    print("   - Use paired comparisons if possible (more powerful)")
    print("   - Consider one-tailed tests if direction is hypothesized")
    print("   - Pool data across similar models for initial screening")

if __name__ == "__main__":
    main()