"""
Analyze MAE differences between base and instruction-tuned models across 100 questions.

This script:
1. Loads human survey data from both Survey 1 (50 questions) and Survey 2 (50 questions)
2. Loads LLM model comparison results (base vs instruct)
3. Calculates MAE (Mean Absolute Error) between model predictions and human averages
4. Computes per-model family MAE differences with 95% CI and p-values via bootstrap
5. Computes overall average MAE difference across all model families

The analysis bootstraps per question per model to properly account for the correlation
structure (same questions asked to all models).
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')

# Survey data files
SURVEY1_PATH = os.path.join(DATA_DIR, 'word_meaning_survey_results.csv')
SURVEY2_PATH = os.path.join(DATA_DIR, 'word_meaning_survey_results_part_2.csv')

# Model comparison results (contains both base and instruct)
# Original file (49 questions from Survey 1 only)
MODEL_COMPARISON_PATH_ORIGINAL = os.path.join(DATA_DIR, 'model_comparison_results.csv')

# New file with 100 questions (if available, from run_base_vs_instruct_100q.py)
MODEL_COMPARISON_PATH_100Q = os.path.join(
    os.path.dirname(BASE_DIR),  # Go up from survey_analysis
    'analysis', 'base_vs_instruct_100q_results.csv'
)

# Alternative location (Google Drive)
MODEL_COMPARISON_PATH_GDRIVE = "G:/My Drive/Computational/llm_interpretation/base_vs_instruct_100q_results.csv"

# Use 100q file if available, otherwise fall back to original
if os.path.exists(MODEL_COMPARISON_PATH_100Q):
    MODEL_COMPARISON_PATH = MODEL_COMPARISON_PATH_100Q
    print(f"Using 100-question results: {MODEL_COMPARISON_PATH_100Q}")
elif os.path.exists(MODEL_COMPARISON_PATH_GDRIVE):
    MODEL_COMPARISON_PATH = MODEL_COMPARISON_PATH_GDRIVE
    print(f"Using 100-question results from GDrive: {MODEL_COMPARISON_PATH_GDRIVE}")
else:
    MODEL_COMPARISON_PATH = MODEL_COMPARISON_PATH_ORIGINAL
    print(f"Using original results (49 questions): {MODEL_COMPARISON_PATH_ORIGINAL}")

# Model families to analyze (those with both base and instruct versions)
# The script will automatically validate data quality and skip models with bad data
MODEL_FAMILIES = {
    'Falcon': {
        'base': 'tiiuae/falcon-7b',
        'instruct': 'tiiuae/falcon-7b-instruct'
    },
    'StableLM': {
        'base': 'stabilityai/stablelm-base-alpha-7b',
        'instruct': 'stabilityai/stablelm-tuned-alpha-7b'
    },
    'RedPajama': {
        'base': 'togethercomputer/RedPajama-INCITE-7B-Base',
        'instruct': 'togethercomputer/RedPajama-INCITE-7B-Instruct'
    },
    'BLOOM': {
        'base': 'bigscience/bloom-7b1',
        'instruct': 'bigscience/bloomz-7b1'
    },
    'Pythia-Dolly': {
        'base': 'EleutherAI/pythia-6.9b',
        'instruct': 'databricks/dolly-v2-7b'
    },
    'Mistral': {
        'base': 'mistralai/Mistral-7B-v0.1',
        'instruct': 'mistralai/Mistral-7B-Instruct-v0.2'
    }
}

# Minimum standard deviation threshold - models with lower std are considered invalid
MIN_STD_THRESHOLD = 0.01

# Minimum valid data threshold - models with more NaN than this fraction are excluded
MAX_NAN_FRACTION = 0.5

# Bootstrap parameters
N_BOOTSTRAP = 10000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_and_clean_survey_data(filepaths):
    """Load survey data and apply exclusion criteria."""
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    dfs = []
    for survey_idx, filepath in enumerate(filepaths, start=1):
        df_temp = pd.read_csv(filepath)
        # Skip header description rows
        df_temp = df_temp[2:].reset_index(drop=True)

        # Rename question columns to include survey prefix
        rename_dict = {}
        for group in range(1, 6):
            for question in range(1, 12):
                old_col = f'Q{group}_{question}'
                if old_col in df_temp.columns:
                    new_col = f'S{survey_idx}_Q{group}_{question}'
                    rename_dict[old_col] = new_col

        df_temp = df_temp.rename(columns=rename_dict)
        dfs.append(df_temp)

    df = pd.concat(dfs, ignore_index=True)

    # Convert Duration to numeric
    df['Duration (in seconds)'] = pd.to_numeric(df['Duration (in seconds)'], errors='coerce')

    # Get question columns
    question_cols = []
    for survey_idx in range(1, len(filepaths) + 1):
        for group in range(1, 6):
            for question in range(1, 12):
                col = f'S{survey_idx}_Q{group}_{question}'
                if col in df.columns:
                    question_cols.append(col)
                    df[col] = pd.to_numeric(df[col], errors='coerce')

    return df, question_cols


def apply_exclusion_criteria(df, question_cols):
    """Apply exclusion criteria: duration, identical responses, attention checks."""
    initial_count = len(df)

    # 1. Filter by completion time (< 20% of median)
    median_duration = df['Duration (in seconds)'].median()
    min_duration = 0.2 * median_duration
    df = df[df['Duration (in seconds)'] >= min_duration]

    # 2. Filter identical slider values
    identical_excluded = []
    for idx, row in df.iterrows():
        substantive_questions = [q for q in question_cols
                                  if not q.endswith('_8') and pd.notna(row.get(q))]
        if len(substantive_questions) > 1:
            values = [row[q] for q in substantive_questions]
            if len(set(values)) == 1:
                identical_excluded.append(idx)
    df = df.drop(identical_excluded)

    # 3. Filter attention check failures
    attention_cols = [col for col in question_cols if col.endswith('_8')]
    attention_failed = []
    for idx, row in df.iterrows():
        for attention_col in attention_cols:
            if pd.notna(row.get(attention_col)):
                if row[attention_col] != 100:
                    attention_failed.append(idx)
                    break
    df = df.drop(attention_failed)

    print(f"Exclusion criteria: {initial_count} -> {len(df)} respondents")
    return df


def extract_question_text(filepaths):
    """Extract question text from survey headers."""
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    question_mapping = {}
    for survey_idx, filepath in enumerate(filepaths, start=1):
        df_raw = pd.read_csv(filepath)
        headers = df_raw.iloc[0]

        for col in df_raw.columns:
            if col.startswith('Q') and '_' in col:
                text = headers[col]
                if pd.notna(text) and isinstance(text, str):
                    if ' - ' in text:
                        question_text = text.split(' - ')[-1].strip()
                        new_col = f'S{survey_idx}_{col}'
                        question_mapping[new_col] = question_text

    return question_mapping


def calculate_human_averages(df, question_cols):
    """Calculate average human response (0-1 scale) for each question."""
    human_avgs = {}
    for q_col in question_cols:
        if not q_col.endswith('_8'):  # Skip attention checks
            responses = df[q_col].dropna()
            if len(responses) > 0:
                # Convert 0-100 scale to 0-1
                human_avgs[q_col] = responses.mean() / 100.0
    return human_avgs


def load_model_data():
    """Load model comparison results."""
    df = pd.read_csv(MODEL_COMPARISON_PATH)

    # Calculate relative probability
    df['relative_prob'] = df['yes_prob'] / (df['yes_prob'] + df['no_prob'])

    # Handle cases where both probs are 0
    df['relative_prob'] = df['relative_prob'].fillna(0.5)

    return df


def match_prompts_to_questions(question_mapping, model_df):
    """Match LLM prompts to survey question IDs."""
    # Create reverse mapping: question text -> question ID
    text_to_id = {v: k for k, v in question_mapping.items()}

    # Match prompts
    matches = {}
    for prompt in model_df['prompt'].unique():
        if prompt in text_to_id:
            matches[prompt] = text_to_id[prompt]

    return matches


def validate_model_data(model_df, model_name):
    """
    Validate that a model has valid data (not all NaN, not constant values).

    Returns:
        (is_valid, reason)
    """
    data = model_df[model_df['model'] == model_name]['relative_prob']

    if len(data) == 0:
        return False, "No data found"

    # Check for all NaN
    nan_fraction = data.isna().sum() / len(data)
    if nan_fraction > MAX_NAN_FRACTION:
        return False, f"{nan_fraction*100:.0f}% NaN values"

    # Check for constant values (low standard deviation)
    valid_data = data.dropna()
    if len(valid_data) > 1:
        std = valid_data.std()
        if std < MIN_STD_THRESHOLD:
            return False, f"Constant values (std={std:.4f})"

    return True, "OK"


# =============================================================================
# BOOTSTRAP ANALYSIS FUNCTIONS
# =============================================================================

def calculate_mae_per_model(model_df, human_avgs, matches, model_name):
    """Calculate MAE for a single model against human averages."""
    model_data = model_df[model_df['model'] == model_name]

    errors = []
    for _, row in model_data.iterrows():
        prompt = row['prompt']
        if prompt in matches:
            q_id = matches[prompt]
            if q_id in human_avgs:
                model_prob = row['relative_prob']
                human_prob = human_avgs[q_id]
                errors.append(abs(model_prob - human_prob))

    if len(errors) > 0:
        return np.mean(errors), errors
    return None, []


def bootstrap_mae_difference(base_errors, instruct_errors, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Bootstrap the MAE difference between instruct and base models.

    Returns:
        dict with observed_diff, ci_lower, ci_upper, p_value
    """
    np.random.seed(seed)

    # Observed values
    base_mae = np.mean(base_errors)
    instruct_mae = np.mean(instruct_errors)
    observed_diff = instruct_mae - base_mae

    # Need paired errors (same questions)
    n_questions = min(len(base_errors), len(instruct_errors))
    base_errors = np.array(base_errors[:n_questions])
    instruct_errors = np.array(instruct_errors[:n_questions])

    # Bootstrap
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Resample question indices with replacement
        indices = np.random.choice(n_questions, size=n_questions, replace=True)
        boot_base_mae = np.mean(base_errors[indices])
        boot_instruct_mae = np.mean(instruct_errors[indices])
        bootstrap_diffs.append(boot_instruct_mae - boot_base_mae)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Confidence interval
    alpha = 1 - CONFIDENCE_LEVEL
    ci_lower = np.percentile(bootstrap_diffs, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)

    # P-value (two-sided test for difference != 0)
    # Use bootstrap distribution centered at 0
    centered_diffs = bootstrap_diffs - np.mean(bootstrap_diffs)
    p_value = np.mean(np.abs(centered_diffs) >= np.abs(observed_diff))

    # Alternative: proportion crossing zero
    if observed_diff > 0:
        p_value_alt = 2 * np.mean(bootstrap_diffs <= 0)
    else:
        p_value_alt = 2 * np.mean(bootstrap_diffs >= 0)
    p_value_alt = min(p_value_alt, 1.0)

    return {
        'base_mae': base_mae,
        'instruct_mae': instruct_mae,
        'observed_diff': observed_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value_alt,
        'n_questions': n_questions,
        'bootstrap_diffs': bootstrap_diffs
    }


def bootstrap_overall_mae_difference(all_family_errors, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Bootstrap the overall MAE difference across all model families.

    Resamples questions (maintaining correlation across models for each question).

    Args:
        all_family_errors: dict of {family_name: {'base': [errors], 'instruct': [errors]}}
    """
    np.random.seed(seed)

    # Get number of questions (should be same for all)
    n_questions = None
    for family, errors in all_family_errors.items():
        n = min(len(errors['base']), len(errors['instruct']))
        if n_questions is None:
            n_questions = n
        else:
            n_questions = min(n_questions, n)

    # Convert to arrays
    family_names = list(all_family_errors.keys())
    base_error_matrix = np.array([all_family_errors[f]['base'][:n_questions] for f in family_names])
    instruct_error_matrix = np.array([all_family_errors[f]['instruct'][:n_questions] for f in family_names])

    # Observed overall MAE difference
    obs_base_mae = np.mean(base_error_matrix)
    obs_instruct_mae = np.mean(instruct_error_matrix)
    observed_diff = obs_instruct_mae - obs_base_mae

    # Bootstrap by resampling questions (columns), keeping all models for each question
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Resample question indices
        q_indices = np.random.choice(n_questions, size=n_questions, replace=True)

        boot_base = base_error_matrix[:, q_indices]
        boot_instruct = instruct_error_matrix[:, q_indices]

        boot_diff = np.mean(boot_instruct) - np.mean(boot_base)
        bootstrap_diffs.append(boot_diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Confidence interval
    alpha = 1 - CONFIDENCE_LEVEL
    ci_lower = np.percentile(bootstrap_diffs, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)

    # P-value
    if observed_diff > 0:
        p_value = 2 * np.mean(bootstrap_diffs <= 0)
    else:
        p_value = 2 * np.mean(bootstrap_diffs >= 0)
    p_value = min(p_value, 1.0)

    return {
        'base_mae': obs_base_mae,
        'instruct_mae': obs_instruct_mae,
        'observed_diff': observed_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'n_questions': n_questions,
        'n_models': len(family_names),
        'bootstrap_diffs': bootstrap_diffs
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 80)
    print("BASE vs INSTRUCTION-TUNED MODEL MAE ANALYSIS")
    print("Comparing model predictions to human average responses")
    print("=" * 80)

    # Load survey data
    print("\n1. Loading survey data...")
    survey_files = [SURVEY1_PATH, SURVEY2_PATH]

    # Check which files exist
    existing_files = [f for f in survey_files if os.path.exists(f)]
    print(f"   Found {len(existing_files)} survey file(s)")

    if len(existing_files) == 0:
        print("   ERROR: No survey files found!")
        return

    survey_df, question_cols = load_and_clean_survey_data(existing_files)
    survey_df = apply_exclusion_criteria(survey_df, question_cols)

    # Extract question text mapping
    question_mapping = extract_question_text(existing_files)
    # Remove attention checks
    question_mapping = {k: v for k, v in question_mapping.items() if not k.endswith('_8')}
    print(f"   Total questions (excluding attention checks): {len(question_mapping)}")

    # Calculate human averages
    human_avgs = calculate_human_averages(survey_df, question_cols)
    print(f"   Questions with valid human responses: {len(human_avgs)}")

    # Load model data
    print("\n2. Loading model comparison data...")
    model_df = load_model_data()
    print(f"   Total model responses: {len(model_df)}")
    print(f"   Unique models: {model_df['model'].nunique()}")
    print(f"   Unique prompts: {model_df['prompt'].nunique()}")

    # Match prompts to questions
    matches = match_prompts_to_questions(question_mapping, model_df)
    print(f"   Matched prompts to survey questions: {len(matches)}")

    # Analyze each model family
    print("\n3. Calculating MAE for each model family...")
    print("-" * 80)

    results = {}
    all_family_errors = {}

    for family_name, models in MODEL_FAMILIES.items():
        base_model = models['base']
        instruct_model = models['instruct']

        # Check if models exist in data
        if base_model not in model_df['model'].values:
            print(f"   {family_name}: Base model not found in data, skipping")
            continue
        if instruct_model not in model_df['model'].values:
            print(f"   {family_name}: Instruct model not found in data, skipping")
            continue

        # Validate data quality
        base_valid, base_reason = validate_model_data(model_df, base_model)
        inst_valid, inst_reason = validate_model_data(model_df, instruct_model)

        if not base_valid:
            print(f"   {family_name}: Base model invalid ({base_reason}), skipping")
            continue
        if not inst_valid:
            print(f"   {family_name}: Instruct model invalid ({inst_reason}), skipping")
            continue

        # Calculate MAE for each model
        base_mae, base_errors = calculate_mae_per_model(model_df, human_avgs, matches, base_model)
        instruct_mae, instruct_errors = calculate_mae_per_model(model_df, human_avgs, matches, instruct_model)

        if base_mae is None or instruct_mae is None:
            print(f"   {family_name}: Could not calculate MAE, skipping")
            continue

        # Store errors for overall analysis
        all_family_errors[family_name] = {
            'base': base_errors,
            'instruct': instruct_errors
        }

        # Bootstrap the difference
        result = bootstrap_mae_difference(base_errors, instruct_errors)
        results[family_name] = result

        # Determine significance stars
        p = result['p_value']
        if p < 0.01:
            stars = '***'
        elif p < 0.05:
            stars = '**'
        elif p < 0.1:
            stars = '*'
        else:
            stars = ''

        print(f"   {family_name}:")
        print(f"      Base MAE:     {result['base_mae']:.3f}")
        print(f"      Instruct MAE: {result['instruct_mae']:.3f}")
        print(f"      Difference:   {result['observed_diff']:+.3f} [{result['ci_lower']:+.3f}, {result['ci_upper']:+.3f}] {stars}")
        print(f"      P-value:      {result['p_value']:.4f}")
        print(f"      N questions:  {result['n_questions']}")
        print()

    # Calculate overall MAE difference
    print("\n4. Calculating overall MAE difference across all model families...")
    print("-" * 80)

    if len(all_family_errors) > 0:
        overall = bootstrap_overall_mae_difference(all_family_errors)

        p = overall['p_value']
        if p < 0.01:
            stars = '***'
        elif p < 0.05:
            stars = '**'
        elif p < 0.1:
            stars = '*'
        else:
            stars = ''

        print(f"   Overall (across {overall['n_models']} model families):")
        print(f"      Base MAE:     {overall['base_mae']:.3f}")
        print(f"      Instruct MAE: {overall['instruct_mae']:.3f}")
        print(f"      Difference:   {overall['observed_diff']:+.3f} [{overall['ci_lower']:+.3f}, {overall['ci_upper']:+.3f}] {stars}")
        print(f"      P-value:      {overall['p_value']:.4f}")
        print(f"      N questions:  {overall['n_questions']}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Model':<12} {'Base MAE':<20} {'Instruct MAE':<20} {'Difference':<25}")
    print(f"{'':12} {'[95% CI]':<20} {'[95% CI]':<20} {'[95% CI]':<25}")
    print("-" * 80)

    for family_name, result in results.items():
        p = result['p_value']
        if p < 0.01:
            stars = '***'
        elif p < 0.05:
            stars = '**'
        elif p < 0.1:
            stars = '*'
        else:
            stars = ''

        # For individual model CIs, we need to bootstrap them separately
        # For now, just show point estimates and difference CI
        base_str = f"{result['base_mae']:.3f}"
        instruct_str = f"{result['instruct_mae']:.3f}"
        diff_str = f"{result['observed_diff']:+.3f}{stars} [{result['ci_lower']:+.3f}, {result['ci_upper']:+.3f}]"

        print(f"{family_name:<12} {base_str:<20} {instruct_str:<20} {diff_str:<25}")

    print("-" * 80)
    if len(all_family_errors) > 0:
        p = overall['p_value']
        if p < 0.01:
            stars = '***'
        elif p < 0.05:
            stars = '**'
        elif p < 0.1:
            stars = '*'
        else:
            stars = ''

        diff_str = f"{overall['observed_diff']:+.3f}{stars} [{overall['ci_lower']:+.3f}, {overall['ci_upper']:+.3f}]"
        print(f"{'OVERALL':<12} {overall['base_mae']:.3f}{'':<14} {overall['instruct_mae']:.3f}{'':<14} {diff_str:<25}")

    print("\nStatistical significance: *** p < 0.01, ** p < 0.05, * p < 0.1")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"Confidence level: {CONFIDENCE_LEVEL*100:.0f}%")

    # Generate LaTeX table
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)
    print(r"""
\begin{table}[H]
\centering
\begin{tabular}{lccc}
\hline
\textbf{Model} & \textbf{Base MAE} & \textbf{Post-trained MAE} & \textbf{Difference} \\
\hline""")

    for family_name, result in results.items():
        p = result['p_value']
        if p < 0.01:
            stars = '***'
        elif p < 0.05:
            stars = '**'
        elif p < 0.1:
            stars = '*'
        else:
            stars = ''

        diff_sign = '+' if result['observed_diff'] >= 0 else ''
        ci_l_sign = '+' if result['ci_lower'] >= 0 else ''
        ci_u_sign = '+' if result['ci_upper'] >= 0 else ''

        print(f"{family_name} & {result['base_mae']:.3f} & {result['instruct_mae']:.3f} & "
              f"{diff_sign}{result['observed_diff']:.3f}{stars} [{ci_l_sign}{result['ci_lower']:.3f}, {ci_u_sign}{result['ci_upper']:.3f}] \\\\")

    print(r"\hline")

    if len(all_family_errors) > 0:
        p = overall['p_value']
        if p < 0.01:
            stars = '***'
        elif p < 0.05:
            stars = '**'
        elif p < 0.1:
            stars = '*'
        else:
            stars = ''

        diff_sign = '+' if overall['observed_diff'] >= 0 else ''
        ci_l_sign = '+' if overall['ci_lower'] >= 0 else ''
        ci_u_sign = '+' if overall['ci_upper'] >= 0 else ''

        print(f"\\textbf{{Overall}} & {overall['base_mae']:.3f} & {overall['instruct_mae']:.3f} & "
              f"{diff_sign}{overall['observed_diff']:.3f}{stars} [{ci_l_sign}{overall['ci_lower']:.3f}, {ci_u_sign}{overall['ci_upper']:.3f}] \\\\")

    print(r"""\hline
\end{tabular}
\caption{Mean Absolute Error (MAE) comparing base and post-trained models against average human responses. Lower MAE indicates better alignment with human judgments. Positive differences indicate worse performance after post-training. Statistical significance: *** $p < 0.01$, ** $p < 0.05$, * $p < 0.1$.}
\label{tab:human_llm_errors_100q}
\end{table}""")

    # Save results to CSV
    output_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(output_dir, exist_ok=True)

    results_data = []
    for family_name, result in results.items():
        results_data.append({
            'Model_Family': family_name,
            'Base_MAE': result['base_mae'],
            'Instruct_MAE': result['instruct_mae'],
            'Difference': result['observed_diff'],
            'CI_Lower': result['ci_lower'],
            'CI_Upper': result['ci_upper'],
            'P_Value': result['p_value'],
            'N_Questions': result['n_questions']
        })

    if len(all_family_errors) > 0:
        results_data.append({
            'Model_Family': 'OVERALL',
            'Base_MAE': overall['base_mae'],
            'Instruct_MAE': overall['instruct_mae'],
            'Difference': overall['observed_diff'],
            'CI_Lower': overall['ci_lower'],
            'CI_Upper': overall['ci_upper'],
            'P_Value': overall['p_value'],
            'N_Questions': overall['n_questions']
        })

    results_df = pd.DataFrame(results_data)
    output_path = os.path.join(output_dir, 'base_vs_instruct_mae_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
