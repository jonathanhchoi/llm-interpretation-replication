import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, bootstrap
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_survey_data(filepaths):
    """Load the survey data and perform initial cleaning.

    Args:
        filepaths: A string (single file path) or list of file paths to load and combine
    """
    # Handle single filepath or list of filepaths
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    dfs = []
    for survey_idx, filepath in enumerate(filepaths, start=1):
        df_temp = pd.read_csv(filepath)
        # Get the actual data rows (skip the header description row)
        df_temp = df_temp[2:].reset_index(drop=True)

        # Rename question columns to include survey prefix (S1_, S2_, etc.)
        # This distinguishes questions from different surveys
        rename_dict = {}
        for group in range(1, 6):
            for question in range(1, 12):
                old_col = f'Q{group}_{question}'
                if old_col in df_temp.columns:
                    new_col = f'S{survey_idx}_Q{group}_{question}'
                    rename_dict[old_col] = new_col

        df_temp = df_temp.rename(columns=rename_dict)
        dfs.append(df_temp)

    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)

    # Convert Duration to numeric
    df['Duration (in seconds)'] = pd.to_numeric(df['Duration (in seconds)'], errors='coerce')

    # Get question columns (S1_Q1_1 through S2_Q5_11 for 2 surveys = 10 groups)
    question_cols = []
    for survey_idx in range(1, len(filepaths) + 1):
        for group in range(1, 6):
            for question in range(1, 12):
                col = f'S{survey_idx}_Q{group}_{question}'
                if col in df.columns:
                    question_cols.append(col)
                    # Convert to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')

    return df, question_cols

def load_llm_data(filepath):
    """Load LLM comparison results."""
    df = pd.read_csv(filepath)
    return df

def apply_exclusion_criteria(df, question_cols):
    """Apply the three exclusion criteria and return cleaned data."""
    initial_count = len(df)
    exclusion_stats = {}
    
    # 1. Filter by completion time (< 20% of median)
    median_duration = df['Duration (in seconds)'].median()
    min_duration = 0.2 * median_duration
    duration_excluded = df[df['Duration (in seconds)'] < min_duration]
    exclusion_stats['duration_excluded'] = len(duration_excluded)
    exclusion_stats['median_duration'] = median_duration
    exclusion_stats['min_duration_threshold'] = min_duration
    df = df[df['Duration (in seconds)'] >= min_duration]
    
    # 2. Filter identical slider values
    identical_excluded = []
    for idx, row in df.iterrows():
        respondent_questions = []
        for col in question_cols:
            if pd.notna(row[col]):
                respondent_questions.append(col)
        
        # Exclude attention check questions (8th question in each group)
        substantive_questions = [q for q in respondent_questions if not q.endswith('_8')]
        
        if len(substantive_questions) > 1:
            values = [row[q] for q in substantive_questions]
            if len(set(values)) == 1:  # All values identical
                identical_excluded.append(idx)
    
    exclusion_stats['identical_excluded'] = len(identical_excluded)
    df = df.drop(identical_excluded)
    
    # 3. Filter attention check failures
    # Extract all attention check columns (those ending with _8)
    attention_cols = [col for col in question_cols if col.endswith('_8')]

    attention_failed = []
    for idx, row in df.iterrows():
        # Check all attention check questions across all surveys/groups
        for attention_col in attention_cols:
            if pd.notna(row[attention_col]):
                if row[attention_col] != 100:
                    attention_failed.append(idx)
                    break
    
    exclusion_stats['attention_failed'] = len(attention_failed)
    df = df.drop(attention_failed)
    
    exclusion_stats['final_count'] = len(df)
    exclusion_stats['total_excluded'] = initial_count - len(df)
    
    return df, exclusion_stats

def extract_question_text(survey_df_raw, filepaths):
    """Extract question text from survey headers.

    Args:
        survey_df_raw: The survey dataframe (not used, kept for backwards compatibility)
        filepaths: Path(s) to survey file(s) to extract question text from
    """
    # Handle single filepath or list of filepaths
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    question_mapping = {}

    # Process each survey file
    for survey_idx, filepath in enumerate(filepaths, start=1):
        df_raw = pd.read_csv(filepath)
        headers = df_raw.iloc[0]  # Get the row with question text

        for col in df_raw.columns:
            if col.startswith('Q') and '_' in col:
                text = headers[col]
                if pd.notna(text) and isinstance(text, str):
                    # Extract the actual question part
                    if ' - ' in text:
                        question_text = text.split(' - ')[-1].strip()
                        # Use the prefixed column name (S1_Q1_1, S2_Q1_1, etc.)
                        new_col = f'S{survey_idx}_{col}'
                        question_mapping[new_col] = question_text

    return question_mapping

def match_survey_to_llm_questions(survey_df, llm_df, survey_filepaths=None):
    """Match survey questions to LLM prompts.

    Args:
        survey_df: Survey dataframe
        llm_df: LLM dataframe
        survey_filepaths: Path(s) to survey file(s) for extracting question text
    """
    # Get question mapping from survey
    if survey_filepaths:
        question_mapping = extract_question_text(survey_df, survey_filepaths)
    else:
        question_mapping = extract_question_text(survey_df, 'word_meaning_survey_results.csv')

    # Filter out attention check questions (S*_Q*_8)
    question_mapping = {k: v for k, v in question_mapping.items() if not k.endswith('_8')}

    # Create reverse mapping from prompt text to question ID
    prompt_to_question = {}
    for q_id, q_text in question_mapping.items():
        prompt_to_question[q_text] = q_id

    # Match with LLM data
    llm_prompts = llm_df['prompt'].unique()
    matches = {}

    for prompt in llm_prompts:
        if prompt in prompt_to_question:
            matches[prompt] = prompt_to_question[prompt]

    return matches, question_mapping

def calculate_human_responses_by_question(df, question_cols):
    """Calculate average human response for each question."""
    question_stats = {}
    
    for question in question_cols:
        if not question.endswith('_8'):  # Skip attention checks
            responses = df[question].dropna()
            if len(responses) > 0:
                question_stats[question] = {
                    'mean': np.mean(responses),
                    'std': np.std(responses),
                    'n': len(responses),
                    'responses': responses.tolist()
                }
    
    return question_stats

def calculate_llm_responses_by_question(llm_df):
    """Calculate average LLM response for each question."""
    # Group by prompt and calculate mean relative probability
    llm_stats = {}
    
    for prompt in llm_df['prompt'].unique():
        prompt_data = llm_df[llm_df['prompt'] == prompt]
        # relative_prob represents P(yes) / (P(yes) + P(no))
        llm_stats[prompt] = {
            'mean': np.mean(prompt_data['relative_prob']),
            'std': np.std(prompt_data['relative_prob']),
            'n': len(prompt_data),
            'model_responses': prompt_data['relative_prob'].tolist()
        }
    
    return llm_stats

def calculate_pearson_with_bootstrap(x, y, n_bootstrap=1000, confidence_level=0.95):
    """Calculate Pearson correlation with bootstrap confidence intervals."""
    # Calculate base correlation
    corr, p_value = pearsonr(x, y)
    
    # Bootstrap for confidence intervals
    def pearson_statistic(x, y):
        return pearsonr(x, y)[0]
    
    # Create paired data
    data = np.column_stack((x, y))
    
    # Bootstrap resampling
    bootstrap_corrs = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, len(data), size=len(data))
        boot_data = data[indices]
        boot_corr = pearson_statistic(boot_data[:, 0], boot_data[:, 1])
        bootstrap_corrs.append(boot_corr)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    ci_lower = np.percentile(bootstrap_corrs, lower_percentile)
    ci_upper = np.percentile(bootstrap_corrs, upper_percentile)
    
    # Standard error
    se = np.std(bootstrap_corrs)
    
    return {
        'correlation': corr,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'standard_error': se,
        'bootstrap_distribution': bootstrap_corrs
    }

def calculate_human_llm_correlation(human_stats, llm_stats, matches):
    """Calculate correlation between human and LLM responses."""
    human_means = []
    llm_means = []
    matched_questions = []
    
    for llm_prompt, survey_q in matches.items():
        if survey_q in human_stats and llm_prompt in llm_stats:
            # Convert human response (0-100) to probability (0-1)
            human_prob = human_stats[survey_q]['mean'] / 100.0
            llm_prob = llm_stats[llm_prompt]['mean']
            
            human_means.append(human_prob)
            llm_means.append(llm_prob)
            matched_questions.append({
                'survey_question': survey_q,
                'llm_prompt': llm_prompt,
                'human_mean': human_prob,
                'llm_mean': llm_prob
            })
    
    if len(human_means) >= 2:
        result = calculate_pearson_with_bootstrap(
            np.array(human_means), 
            np.array(llm_means)
        )
        result['n_questions'] = len(human_means)
        result['matched_questions'] = matched_questions
        return result
    else:
        return None

def calculate_per_item_agreement_humans(df, question_cols):
    """Calculate per-item agreement for humans (average pairwise agreement between humans for each question)."""
    item_agreements = {}
    all_item_avg_agreements = []
    
    for question in question_cols:
        if not question.endswith('_8'):  # Skip attention checks
            # Get all responses for this question
            responses = df[question].dropna().values
            
            if len(responses) >= 2:
                # Calculate pairwise absolute differences for this question
                pairwise_diffs = []
                
                # For each pair of respondents
                for i in range(len(responses)):
                    for j in range(i + 1, len(responses)):
                        # Calculate absolute difference (on 0-100 scale)
                        diff = abs(responses[i] - responses[j])
                        # Convert to agreement (100 - diff) / 100 to get 0-1 scale
                        agreement = (100 - diff) / 100
                        pairwise_diffs.append(agreement)
                
                if pairwise_diffs:
                    avg_agreement = np.mean(pairwise_diffs)
                    item_agreements[question] = {
                        'mean_agreement': avg_agreement,
                        'std_agreement': np.std(pairwise_diffs),
                        'n_pairs': len(pairwise_diffs),
                        'response_variance': np.var(responses),
                        'n_responses': len(responses)
                    }
                    all_item_avg_agreements.append(avg_agreement)
    
    # Calculate bootstrap CI for overall mean
    if all_item_avg_agreements:
        bootstrap_means = []
        for _ in range(1000):
            indices = np.random.randint(0, len(all_item_avg_agreements), size=len(all_item_avg_agreements))
            boot_sample = [all_item_avg_agreements[i] for i in indices]
            bootstrap_means.append(np.mean(boot_sample))
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
    else:
        ci_lower = ci_upper = 0
    
    return {
        'per_item': item_agreements,
        'overall_mean': np.mean(all_item_avg_agreements) if all_item_avg_agreements else 0,
        'overall_std': np.std(all_item_avg_agreements) if all_item_avg_agreements else 0,
        'n_items': len(all_item_avg_agreements),
        'overall_mean_ci_lower': ci_lower,
        'overall_mean_ci_upper': ci_upper
    }

def calculate_per_item_agreement_llms(llm_df):
    """Calculate per-item agreement for LLMs (average pairwise agreement between models for each prompt)."""
    item_agreements = {}
    all_item_avg_agreements = []
    
    prompts = llm_df['prompt'].unique()
    models = llm_df['model'].unique()
    
    for prompt in prompts:
        # Get all model responses for this prompt
        prompt_data = llm_df[llm_df['prompt'] == prompt]
        model_responses = []
        
        for model in models:
            model_prob = prompt_data[prompt_data['model'] == model]['relative_prob'].values
            if len(model_prob) > 0 and not np.isnan(model_prob[0]):
                model_responses.append(float(model_prob[0]))  # Ensure float type
        
        if len(model_responses) >= 2:
            # Calculate pairwise absolute differences for this prompt
            pairwise_diffs = []
            
            # For each pair of models
            for i in range(len(model_responses)):
                for j in range(i + 1, len(model_responses)):
                    # Calculate absolute difference (both are on 0-1 scale)
                    diff = abs(model_responses[i] - model_responses[j])
                    # Convert to agreement
                    agreement = 1 - diff
                    pairwise_diffs.append(agreement)
            
            if pairwise_diffs:
                avg_agreement = float(np.mean(pairwise_diffs))  # Ensure float
                item_agreements[prompt] = {
                    'mean_agreement': avg_agreement,
                    'std_agreement': float(np.std(pairwise_diffs)) if len(pairwise_diffs) > 1 else 0.0,
                    'n_pairs': len(pairwise_diffs),
                    'response_variance': float(np.var(model_responses)) if len(model_responses) > 1 else 0.0,
                    'n_models': len(model_responses)
                }
                all_item_avg_agreements.append(avg_agreement)
    
    # Calculate bootstrap CI for overall mean
    if all_item_avg_agreements:
        bootstrap_means = []
        for _ in range(1000):
            indices = np.random.randint(0, len(all_item_avg_agreements), size=len(all_item_avg_agreements))
            boot_sample = [all_item_avg_agreements[i] for i in indices]
            bootstrap_means.append(np.mean(boot_sample))
        ci_lower = float(np.percentile(bootstrap_means, 2.5))
        ci_upper = float(np.percentile(bootstrap_means, 97.5))
    else:
        ci_lower = ci_upper = 0.0
    
    return {
        'per_item': item_agreements,
        'overall_mean': float(np.mean(all_item_avg_agreements)) if all_item_avg_agreements else 0.0,
        'overall_std': float(np.std(all_item_avg_agreements)) if all_item_avg_agreements else 0.0,
        'n_items': len(all_item_avg_agreements),
        'overall_mean_ci_lower': ci_lower,
        'overall_mean_ci_upper': ci_upper
    }

def calculate_human_cross_prompt_correlations(df, question_cols, n_bootstrap=100):
    """Calculate pairwise correlations between humans within each group with bootstrapping."""
    all_correlations = []
    group_results = {}

    # Extract unique group identifiers from question_cols (e.g., S1_Q1, S1_Q2, ..., S2_Q1, S2_Q2, ...)
    groups = set()
    for col in question_cols:
        # Parse column name like S1_Q1_1 -> S1_Q1
        parts = col.split('_')
        if len(parts) >= 3:
            group_id = f"{parts[0]}_Q{parts[1][1:]}"  # S1_Q1, S2_Q3, etc.
            groups.add(group_id)

    # Process each group separately
    for group_id in sorted(groups):
        # Get questions for this group (excluding attention check *_8)
        group_questions = [col for col in question_cols if col.startswith(group_id + '_') and not col.endswith('_8')]

        if len(group_questions) == 0:
            continue

        # Get the first question column to check for respondents
        first_question = group_questions[0] if group_questions else None
        if first_question is None:
            continue

        # Get respondents who answered this group
        group_respondents = df[df[first_question].notna()].copy()

        if len(group_respondents) < 2:
            continue
            
        # Create data matrix for this group (questions x respondents)
        data_matrix = []
        respondent_ids = []
        
        for idx in group_respondents.index:
            respondent_data = []
            for question in group_questions:
                value = group_respondents.loc[idx, question]
                if pd.notna(value):
                    respondent_data.append(float(value) / 100.0)
                else:
                    respondent_data.append(np.nan)
            
            # Include if answered at least 5 questions
            if sum(pd.notna(v) for v in respondent_data) >= 5:
                data_matrix.append(respondent_data)
                respondent_ids.append(idx)
        
        if len(respondent_ids) < 2:
            continue
            
        # Convert to DataFrame (questions as rows, respondents as columns)
        data_df = pd.DataFrame(data_matrix, index=respondent_ids, columns=group_questions).T
        
        # Calculate correlation matrix between respondents
        corr_matrix = data_df.corr(method='pearson')
        
        # Extract pairwise correlations
        group_correlations = []
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                corr_value = corr_matrix.iloc[i, j]
                if not np.isnan(corr_value):
                    group_correlations.append(corr_value)
                    all_correlations.append(corr_value)
        
        group_results[group_id] = {
            'n_respondents': len(respondent_ids),
            'n_pairs': len(group_correlations),
            'mean_correlation': np.mean(group_correlations) if group_correlations else 0,
            'correlations': group_correlations
        }
    
    # Calculate overall statistics
    base_mean = np.mean(all_correlations) if all_correlations else 0
    
    # Bootstrap by resampling within groups
    bootstrap_means = []

    for _ in range(n_bootstrap):
        boot_correlations = []

        for group_id in sorted(groups):
            if group_id not in group_results:
                continue

            group_questions = [col for col in question_cols if col.startswith(group_id + '_') and not col.endswith('_8')]
            n_questions = len(group_questions)

            if n_questions < 2:
                continue

            # Sample questions with replacement
            sampled_indices = np.random.choice(n_questions, size=n_questions, replace=True)
            sampled_questions = [group_questions[i] for i in sampled_indices]

            # Get the first question column to check for respondents
            first_question = group_questions[0]

            # Get respondents for this group
            group_respondents = df[df[first_question].notna()].copy()
            
            # Create bootstrap data matrix
            boot_data_matrix = []
            respondent_ids = []
            
            for idx in group_respondents.index:
                respondent_data = []
                for question in sampled_questions:
                    value = group_respondents.loc[idx, question]
                    if pd.notna(value):
                        respondent_data.append(float(value) / 100.0)
                    else:
                        respondent_data.append(np.nan)
                
                if sum(pd.notna(v) for v in respondent_data) >= 5:
                    boot_data_matrix.append(respondent_data)
                    respondent_ids.append(idx)
            
            if len(respondent_ids) >= 2:
                # Calculate correlations
                boot_df = pd.DataFrame(boot_data_matrix, index=respondent_ids, columns=range(n_questions)).T
                boot_corr_matrix = boot_df.corr(method='pearson')
                
                for i in range(len(boot_corr_matrix)):
                    for j in range(i + 1, len(boot_corr_matrix)):
                        corr_value = boot_corr_matrix.iloc[i, j]
                        if not np.isnan(corr_value):
                            boot_correlations.append(corr_value)
        
        if boot_correlations:
            bootstrap_means.append(np.mean(boot_correlations))
    
    # Calculate confidence intervals
    if bootstrap_means:
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
    else:
        ci_lower = ci_upper = base_mean
    
    return {
        'group_results': group_results,
        'pairwise_correlations': all_correlations,
        'mean_correlation': base_mean,
        'std_correlation': np.std(all_correlations) if all_correlations else 0,
        'n_pairs': len(all_correlations),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def calculate_llm_cross_prompt_correlations(llm_df, human_question_mapping, n_bootstrap=100):
    """Calculate pairwise correlations between models within groups matching human groups."""
    all_correlations = []
    group_results = {}

    # Map prompts to groups based on matched survey questions
    prompt_to_group = {}
    for prompt, survey_q in human_question_mapping.items():
        if survey_q:  # If there's a match
            # Extract group identifier from S1_Q1_1 -> S1_Q1
            parts = survey_q.split('_')
            if len(parts) >= 3:
                group_id = f"{parts[0]}_Q{parts[1][1:]}"  # S1_Q1, S2_Q3, etc.
                prompt_to_group[prompt] = group_id

    # Get unique groups
    groups = set(prompt_to_group.values())

    # Process each group separately
    for group_id in sorted(groups):
        # Get prompts for this group
        group_prompts = [p for p, g in prompt_to_group.items() if g == group_id]

        # Skip attention check questions (8th question in each group)
        group_prompts = [p for p in group_prompts if not p.endswith('_8')]

        if len(group_prompts) < 2:
            continue
        
        # Create pivot table for this group
        group_data = llm_df[llm_df['prompt'].isin(group_prompts)]
        pivot_df = group_data.pivot_table(index='prompt', columns='model', values='relative_prob')
        
        if len(pivot_df) < 2:
            continue
        
        # Calculate correlation matrix
        corr_matrix = pivot_df.corr(method='pearson')
        
        # Extract pairwise correlations
        group_correlations = []
        models = list(corr_matrix.columns)
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                corr_value = corr_matrix.iloc[i, j]
                if not np.isnan(corr_value):
                    group_correlations.append(corr_value)
                    all_correlations.append(corr_value)
        
        group_results[group_id] = {
            'n_prompts': len(group_prompts),
            'n_models': len(models),
            'n_pairs': len(group_correlations),
            'mean_correlation': np.mean(group_correlations) if group_correlations else 0,
            'correlations': group_correlations
        }
    
    # Calculate overall statistics
    base_mean = np.mean(all_correlations) if all_correlations else 0
    
    # Bootstrap by resampling within groups
    bootstrap_means = []

    for _ in range(n_bootstrap):
        boot_correlations = []

        for group_id in sorted(groups):
            if group_id not in group_results:
                continue

            # Get prompts for this group
            group_prompts = [p for p, g in prompt_to_group.items() if g == group_id]
            group_prompts = [p for p in group_prompts if not p.endswith('_8')]
            n_prompts = len(group_prompts)
            
            if n_prompts < 2:
                continue
            
            # Sample prompts with replacement
            sampled_indices = np.random.choice(n_prompts, size=n_prompts, replace=True)
            sampled_prompts = [group_prompts[i] for i in sampled_indices]
            
            # Create bootstrap data
            group_data = llm_df[llm_df['prompt'].isin(group_prompts)]
            pivot_df = group_data.pivot_table(index='prompt', columns='model', values='relative_prob')
            
            # Create bootstrap pivot table
            boot_pivot = pivot_df.loc[sampled_prompts]
            
            # Calculate correlation matrix
            boot_corr_matrix = boot_pivot.corr(method='pearson')
            
            # Extract correlations
            models = list(boot_corr_matrix.columns)
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    corr_value = boot_corr_matrix.iloc[i, j]
                    if not np.isnan(corr_value):
                        boot_correlations.append(corr_value)
        
        if boot_correlations:
            bootstrap_means.append(np.mean(boot_correlations))
    
    # Calculate confidence intervals
    if bootstrap_means:
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
    else:
        ci_lower = ci_upper = base_mean
    
    return {
        'group_results': group_results,
        'pairwise_correlations': all_correlations,
        'mean_correlation': base_mean,
        'std_correlation': np.std(all_correlations) if all_correlations else 0,
        'n_pairs': len(all_correlations),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def calculate_cross_prompt_difference_ci(df, llm_df, question_cols, human_question_mapping, n_bootstrap=1000):
    """Calculate confidence interval for the difference between human and LLM cross-prompt correlations using bootstrap."""
    # Map prompts to groups based on matched survey questions
    prompt_to_group = {}
    for prompt, survey_q in human_question_mapping.items():
        if survey_q:  # If there's a match
            # Extract group identifier from S1_Q1_1 -> S1_Q1
            parts = survey_q.split('_')
            if len(parts) >= 3:
                group_id = f"{parts[0]}_Q{parts[1][1:]}"  # S1_Q1, S2_Q3, etc.
                prompt_to_group[prompt] = group_id

    # Extract unique group identifiers from question_cols
    groups_human = set()
    for col in question_cols:
        parts = col.split('_')
        if len(parts) >= 3:
            group_id = f"{parts[0]}_Q{parts[1][1:]}"
            groups_human.add(group_id)

    # Get unique LLM groups
    groups_llm = set(prompt_to_group.values())

    # Use intersection of both to ensure we only process groups with both human and LLM data
    groups = groups_human.intersection(groups_llm)
    
    # Bootstrap to calculate difference CI
    bootstrap_differences = []
    
    for boot_iter in range(n_bootstrap):
        human_boot_correlations = []
        llm_boot_correlations = []

        # Process each group separately
        for group_id in sorted(groups):
            # Bootstrap human correlations for this group
            group_questions = [col for col in question_cols if col.startswith(group_id + '_') and not col.endswith('_8')]
            n_questions = len(group_questions)

            if n_questions < 2:
                continue

            # Get the first question column to check for respondents
            first_question = group_questions[0]

            # Get respondents who answered this group
            group_respondents = df[df[first_question].notna()].copy()
            
            if len(group_respondents) >= 2:
                # Sample questions with replacement
                sampled_indices = np.random.choice(n_questions, size=n_questions, replace=True)
                sampled_questions = [group_questions[i] for i in sampled_indices]
                
                # Create bootstrap data matrix
                boot_data_matrix = []
                respondent_ids = []
                
                for idx in group_respondents.index:
                    respondent_data = []
                    for question in sampled_questions:
                        value = group_respondents.loc[idx, question]
                        if pd.notna(value):
                            respondent_data.append(float(value) / 100.0)
                        else:
                            respondent_data.append(np.nan)
                    
                    if sum(pd.notna(v) for v in respondent_data) >= 5:
                        boot_data_matrix.append(respondent_data)
                        respondent_ids.append(idx)
                
                if len(respondent_ids) >= 2:
                    # Calculate correlations
                    boot_df = pd.DataFrame(boot_data_matrix, index=respondent_ids, columns=range(n_questions)).T
                    boot_corr_matrix = boot_df.corr(method='pearson')
                    
                    for i in range(len(boot_corr_matrix)):
                        for j in range(i + 1, len(boot_corr_matrix)):
                            corr_value = boot_corr_matrix.iloc[i, j]
                            if not np.isnan(corr_value):
                                human_boot_correlations.append(corr_value)
            
            # Bootstrap LLM correlations for this group
            group_prompts = [p for p, g in prompt_to_group.items() if g == group_id]
            group_prompts = [p for p in group_prompts if not p.endswith('_8')]
            n_prompts = len(group_prompts)
            
            if n_prompts >= 2:
                # Sample prompts with replacement
                sampled_indices = np.random.choice(n_prompts, size=n_prompts, replace=True)
                sampled_prompts = [group_prompts[i] for i in sampled_indices]
                
                # Create bootstrap data
                group_data = llm_df[llm_df['prompt'].isin(group_prompts)]
                pivot_df = group_data.pivot_table(index='prompt', columns='model', values='relative_prob')
                
                if len(pivot_df) >= 2:
                    # Create bootstrap pivot table
                    boot_pivot = pivot_df.loc[sampled_prompts]
                    
                    # Calculate correlation matrix
                    boot_corr_matrix = boot_pivot.corr(method='pearson')
                    
                    # Extract correlations
                    models = list(boot_corr_matrix.columns)
                    for i in range(len(models)):
                        for j in range(i + 1, len(models)):
                            corr_value = boot_corr_matrix.iloc[i, j]
                            if not np.isnan(corr_value):
                                llm_boot_correlations.append(corr_value)
        
        # Calculate means for this bootstrap iteration
        if human_boot_correlations and llm_boot_correlations:
            human_mean = np.mean(human_boot_correlations)
            llm_mean = np.mean(llm_boot_correlations)
            difference = human_mean - llm_mean
            bootstrap_differences.append(difference)
    
    # Calculate confidence intervals for the difference
    if bootstrap_differences:
        ci_lower = np.percentile(bootstrap_differences, 2.5)
        ci_upper = np.percentile(bootstrap_differences, 97.5)
        mean_difference = np.mean(bootstrap_differences)
    else:
        ci_lower = ci_upper = mean_difference = None
    
    return {
        'mean_difference': mean_difference,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_bootstrap': len(bootstrap_differences)
    }

def calculate_meta_correlation(human_agreements, llm_agreements, matches):
    """Calculate correlation between human and LLM per-item agreement patterns."""
    # Compare per-item agreement between humans and LLMs
    
    # Match human and LLM items
    matched_human_agreements = []
    matched_llm_agreements = []
    
    for llm_prompt, survey_q in matches.items():
        if survey_q in human_agreements['per_item'] and llm_prompt in llm_agreements['per_item']:
            matched_human_agreements.append(human_agreements['per_item'][survey_q]['mean_agreement'])
            matched_llm_agreements.append(llm_agreements['per_item'][llm_prompt]['mean_agreement'])
    
    # Calculate correlation between human and LLM agreement patterns
    if len(matched_human_agreements) >= 2:
        corr_result = calculate_pearson_with_bootstrap(
            np.array(matched_human_agreements),
            np.array(matched_llm_agreements)
        )
        
        result = {
            'correlation': corr_result['correlation'],
            'p_value': corr_result['p_value'],
            'ci_lower': corr_result['ci_lower'],
            'ci_upper': corr_result['ci_upper'],
            'n_matched_items': len(matched_human_agreements),
            'human_mean_agreement': human_agreements['overall_mean'],
            'human_std_agreement': human_agreements['overall_std'],
            'llm_mean_agreement': llm_agreements['overall_mean'],
            'llm_std_agreement': llm_agreements['overall_std'],
            'interpretation': 'Correlation between human and LLM per-item agreement patterns'
        }
    else:
        result = {
            'correlation': None,
            'n_matched_items': len(matched_human_agreements),
            'human_mean_agreement': human_agreements['overall_mean'],
            'human_std_agreement': human_agreements['overall_std'],
            'llm_mean_agreement': llm_agreements['overall_mean'],
            'llm_std_agreement': llm_agreements['overall_std'],
            'interpretation': 'Insufficient matched items for correlation'
        }
    
    return result

def generate_comprehensive_report(exclusion_stats, human_stats, llm_stats, 
                                human_llm_corr, human_item_agreement, llm_item_agreement, 
                                meta_corr, matches, human_cross_prompt_corr=None, llm_cross_prompt_corr=None,
                                cross_prompt_diff_ci=None):
    """Generate comprehensive analysis report."""
    print("\n" + "="*80)
    print("CONSOLIDATED SURVEY ANALYSIS - HUMAN vs LLM ORDINARY MEANING AGREEMENT")
    print("="*80)
    
    print("\nEXCLUSION STATISTICS:")
    print(f"  Initial respondents: {exclusion_stats['final_count'] + exclusion_stats['total_excluded']}")
    print(f"  Excluded for short duration: {exclusion_stats['duration_excluded']}")
    print(f"  Excluded for identical responses: {exclusion_stats['identical_excluded']}")
    print(f"  Excluded for attention check failure: {exclusion_stats['attention_failed']}")
    print(f"  Total excluded: {exclusion_stats['total_excluded']}")
    print(f"  Final sample size: {exclusion_stats['final_count']}")
    
    print("\n" + "-"*80)
    print("QUESTION MATCHING:")
    print(f"  Total survey questions: {len(human_stats)}")
    print(f"  Total LLM prompts: {len(llm_stats)}")
    print(f"  Successfully matched: {len(matches)}")
    
    print("\n" + "-"*80)
    print("HUMAN-LLM CORRELATION (Question-Level Agreement):")
    if human_llm_corr:
        print(f"  Pearson correlation: {human_llm_corr['correlation']:.3f}")
        print(f"  95% CI: [{human_llm_corr['ci_lower']:.3f}, {human_llm_corr['ci_upper']:.3f}]")
        print(f"  Standard error: {human_llm_corr['standard_error']:.3f}")
        print(f"  p-value: {human_llm_corr['p_value']:.4f}")
        print(f"  Number of questions: {human_llm_corr['n_questions']}")
    else:
        print("  Insufficient matched questions for correlation")
    
    print("\n" + "-"*80)
    print("PER-ITEM AGREEMENT (Average agreement between raters for each item):")
    
    print("\n  Human per-item agreement:")
    print(f"    Mean agreement across items: {human_item_agreement['overall_mean']:.3f}")
    print(f"    95% CI: [{human_item_agreement.get('overall_mean_ci_lower', 0):.3f}, {human_item_agreement.get('overall_mean_ci_upper', 0):.3f}]")
    print(f"    Std across items: {human_item_agreement['overall_std']:.3f}")
    print(f"    Number of items: {human_item_agreement['n_items']}")
    
    print("\n  LLM per-item agreement:")
    print(f"    Mean agreement across items: {llm_item_agreement['overall_mean']:.3f}")
    print(f"    95% CI: [{llm_item_agreement.get('overall_mean_ci_lower', 0):.3f}, {llm_item_agreement.get('overall_mean_ci_upper', 0):.3f}]")
    print(f"    Std across items: {llm_item_agreement['overall_std']:.3f}")
    print(f"    Number of items: {llm_item_agreement['n_items']}")
    
    print("\n" + "-"*80)
    print("CROSS-PROMPT CORRELATIONS (How similarly raters rank items):")
    
    if human_cross_prompt_corr:
        print("\n  Human cross-prompt correlations (within groups):")
        print(f"    Mean correlation between respondent pairs: {human_cross_prompt_corr['mean_correlation']:.3f}")
        print(f"    95% CI: [{human_cross_prompt_corr['ci_lower']:.3f}, {human_cross_prompt_corr['ci_upper']:.3f}]")
        print(f"    Std of correlations: {human_cross_prompt_corr['std_correlation']:.3f}")
        print(f"    Number of respondent pairs: {human_cross_prompt_corr['n_pairs']}")
        if 'group_results' in human_cross_prompt_corr:
            for group, stats in sorted(human_cross_prompt_corr['group_results'].items()):
                print(f"    {group}: {stats['n_respondents']} respondents, mean corr = {stats['mean_correlation']:.3f}")
    
    if llm_cross_prompt_corr:
        print("\n  LLM cross-prompt correlations (within groups):")
        print(f"    Mean correlation between model pairs: {llm_cross_prompt_corr['mean_correlation']:.3f}")
        print(f"    95% CI: [{llm_cross_prompt_corr['ci_lower']:.3f}, {llm_cross_prompt_corr['ci_upper']:.3f}]")
        print(f"    Std of correlations: {llm_cross_prompt_corr['std_correlation']:.3f}")
        print(f"    Number of model pairs: {llm_cross_prompt_corr['n_pairs']}")
        if 'group_results' in llm_cross_prompt_corr:
            for group, stats in sorted(llm_cross_prompt_corr['group_results'].items()):
                print(f"    {group}: {stats['n_prompts']} prompts, {stats['n_models']} models, mean corr = {stats['mean_correlation']:.3f}")
    
    if cross_prompt_diff_ci and human_cross_prompt_corr and llm_cross_prompt_corr:
        print("\n  Difference in cross-prompt correlations (Human - LLM):")
        print(f"    Mean difference: {cross_prompt_diff_ci['mean_difference']:.3f}")
        print(f"    95% CI: [{cross_prompt_diff_ci['ci_lower']:.3f}, {cross_prompt_diff_ci['ci_upper']:.3f}]")
        print(f"    Bootstrap iterations: {cross_prompt_diff_ci['n_bootstrap']}")
    
    print("\n" + "-"*80)
    print("META-CORRELATION (Agreement Pattern Comparison):")
    if meta_corr:
        if meta_corr['correlation'] is not None:
            print(f"  Correlation between human and LLM per-item agreement patterns: {meta_corr['correlation']:.3f}")
            print(f"  95% CI: [{meta_corr['ci_lower']:.3f}, {meta_corr['ci_upper']:.3f}]")
            print(f"  p-value: {meta_corr['p_value']:.4f}")
            print(f"  Number of matched items: {meta_corr['n_matched_items']}")
        else:
            print(f"  {meta_corr['interpretation']}")
        print(f"\n  Human mean per-item agreement: {meta_corr['human_mean_agreement']:.3f}")
        print(f"  LLM mean per-item agreement: {meta_corr['llm_mean_agreement']:.3f}")
    else:
        print("  Unable to calculate meta-correlation")
    
    print("\n" + "-"*80)
    print("INTERPRETATION:")
    if human_llm_corr:
        print(f"\nThe correlation between average human and LLM responses is {human_llm_corr['correlation']:.3f},")
        print(f"indicating {'strong' if abs(human_llm_corr['correlation']) > 0.7 else 'moderate' if abs(human_llm_corr['correlation']) > 0.4 else 'weak'} agreement")
        print(f"between humans and LLMs on ordinary meaning judgments.")
    
    if meta_corr:
        print(f"\nThe per-item agreement patterns show that humans have")
        print(f"mean agreement of {human_item_agreement['overall_mean']:.3f} compared to LLMs' {llm_item_agreement['overall_mean']:.3f},")
        print(f"suggesting {'humans' if human_item_agreement['overall_mean'] > llm_item_agreement['overall_mean'] else 'LLMs'} are more consistent in their ordinary meaning judgments.")
    
    print("\n" + "="*80)

def save_results(output_file, exclusion_stats, human_stats, llm_stats, 
                human_llm_corr, human_item_agreement, llm_item_agreement, meta_corr, matches,
                human_cross_prompt_corr=None, llm_cross_prompt_corr=None, cross_prompt_diff_ci=None):
    """Save all results to a JSON file."""
    results = {
        'exclusion_stats': exclusion_stats,
        'matching_stats': {
            'n_human_questions': len(human_stats),
            'n_llm_prompts': len(llm_stats),
            'n_matched': len(matches),
            'matches': matches
        },
        'human_llm_correlation': {
            'correlation': human_llm_corr['correlation'] if human_llm_corr else None,
            'ci_lower': human_llm_corr['ci_lower'] if human_llm_corr else None,
            'ci_upper': human_llm_corr['ci_upper'] if human_llm_corr else None,
            'standard_error': human_llm_corr['standard_error'] if human_llm_corr else None,
            'p_value': human_llm_corr['p_value'] if human_llm_corr else None,
            'n_questions': human_llm_corr['n_questions'] if human_llm_corr else 0
        },
        'per_item_agreement': {
            'human': {
                'overall_mean': human_item_agreement['overall_mean'],
                'overall_mean_ci_lower': human_item_agreement.get('overall_mean_ci_lower', 0),
                'overall_mean_ci_upper': human_item_agreement.get('overall_mean_ci_upper', 0),
                'overall_std': human_item_agreement['overall_std'],
                'n_items': human_item_agreement['n_items'],
                'per_item_details': human_item_agreement['per_item']
            },
            'llm': {
                'overall_mean': llm_item_agreement['overall_mean'],
                'overall_mean_ci_lower': llm_item_agreement.get('overall_mean_ci_lower', 0),
                'overall_mean_ci_upper': llm_item_agreement.get('overall_mean_ci_upper', 0),
                'overall_std': llm_item_agreement['overall_std'],
                'n_items': llm_item_agreement['n_items'],
                'per_item_details': llm_item_agreement['per_item']
            }
        },
        'meta_correlation': meta_corr if meta_corr else {},
        'cross_prompt_correlations': {
            'human': {
                'mean_correlation': human_cross_prompt_corr['mean_correlation'] if human_cross_prompt_corr else None,
                'ci_lower': human_cross_prompt_corr['ci_lower'] if human_cross_prompt_corr else None,
                'ci_upper': human_cross_prompt_corr['ci_upper'] if human_cross_prompt_corr else None,
                'std_correlation': human_cross_prompt_corr['std_correlation'] if human_cross_prompt_corr else None,
                'n_pairs': human_cross_prompt_corr['n_pairs'] if human_cross_prompt_corr else None
            },
            'llm': {
                'mean_correlation': llm_cross_prompt_corr['mean_correlation'] if llm_cross_prompt_corr else None,
                'ci_lower': llm_cross_prompt_corr['ci_lower'] if llm_cross_prompt_corr else None,
                'ci_upper': llm_cross_prompt_corr['ci_upper'] if llm_cross_prompt_corr else None,
                'std_correlation': llm_cross_prompt_corr['std_correlation'] if llm_cross_prompt_corr else None,
                'n_pairs': llm_cross_prompt_corr['n_pairs'] if llm_cross_prompt_corr else None
            },
            'difference': {
                'mean_difference': cross_prompt_diff_ci['mean_difference'] if cross_prompt_diff_ci else None,
                'ci_lower': cross_prompt_diff_ci['ci_lower'] if cross_prompt_diff_ci else None,
                'ci_upper': cross_prompt_diff_ci['ci_upper'] if cross_prompt_diff_ci else None,
                'n_bootstrap': cross_prompt_diff_ci['n_bootstrap'] if cross_prompt_diff_ci else None
            }
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

def main():
    # Load data
    print("Loading survey data...")
    import os
    # Get the base directory (parent of survey_analysis)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    survey_files = [
        os.path.join(data_dir, 'word_meaning_survey_results.csv'),
        os.path.join(data_dir, 'word_meaning_survey_results_part_2.csv')
    ]
    survey_df, question_cols = load_and_clean_survey_data(survey_files)
    print(f"Loaded {len(survey_df)} total survey responses from {len(survey_files)} files")

    print("Loading LLM data...")
    # Use combined results with both Survey 1 and Survey 2
    llm_df = load_llm_data(os.path.join(data_dir, 'instruct_model_comparison_results_combined.csv'))
    
    # Apply exclusion criteria
    print("Applying exclusion criteria...")
    survey_clean, exclusion_stats = apply_exclusion_criteria(survey_df, question_cols)
    
    # Match questions
    print("Matching survey questions to LLM prompts...")
    matches, question_mapping = match_survey_to_llm_questions(survey_df, llm_df, survey_files)
    
    # Calculate human responses by question
    print("Calculating human response statistics...")
    human_stats = calculate_human_responses_by_question(survey_clean, question_cols)
    
    # Calculate LLM responses by question
    print("Calculating LLM response statistics...")
    llm_stats = calculate_llm_responses_by_question(llm_df)
    
    # Calculate human-LLM correlation
    print("Calculating human-LLM correlation...")
    human_llm_corr = calculate_human_llm_correlation(human_stats, llm_stats, matches)
    
    # Calculate per-item agreements
    print("Calculating human per-item agreements...")
    human_item_agreement = calculate_per_item_agreement_humans(survey_clean, question_cols)
    
    print("Calculating LLM per-item agreements...")
    llm_item_agreement = calculate_per_item_agreement_llms(llm_df)
    
    # Calculate cross-prompt correlations
    print("Calculating human cross-prompt correlations...")
    human_cross_prompt_corr = calculate_human_cross_prompt_correlations(survey_clean, question_cols)
    
    print("Calculating LLM cross-prompt correlations...")
    llm_cross_prompt_corr = calculate_llm_cross_prompt_correlations(llm_df, matches)
    
    # Calculate difference in cross-prompt correlations with CI
    print("Calculating confidence interval for cross-prompt correlation difference...")
    cross_prompt_diff_ci = calculate_cross_prompt_difference_ci(survey_clean, llm_df, question_cols, matches)
    
    # Calculate meta-correlation
    print("Calculating meta-correlation...")
    meta_corr = calculate_meta_correlation(human_item_agreement, llm_item_agreement, matches)
    
    # Generate report
    generate_comprehensive_report(
        exclusion_stats, human_stats, llm_stats,
        human_llm_corr, human_item_agreement, llm_item_agreement,
        meta_corr, matches, human_cross_prompt_corr, llm_cross_prompt_corr,
        cross_prompt_diff_ci
    )
    
    # Save results
    save_results(
        'consolidated_analysis_results.json',
        exclusion_stats, human_stats, llm_stats,
        human_llm_corr, human_item_agreement, llm_item_agreement,
        meta_corr, matches, human_cross_prompt_corr, llm_cross_prompt_corr,
        cross_prompt_diff_ci
    )

if __name__ == "__main__":
    main()