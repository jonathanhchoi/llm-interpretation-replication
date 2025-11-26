"""
Evaluation of Closed-Source LLMs on Ordinary Meaning Questions
Analyzes GPT-4.1, Gemini, and Claude Opus 4.1 on 50 questions about ordinary meaning
Measures both relative probabilities (where available) and verbalized confidence
"""

import os
import time
import json
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, bootstrap
import openai
import google.generativeai as genai
from anthropic import Anthropic
from config import OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Get the base directory for the project (parent of analysis folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Model configurations
GPT_MODEL = "gpt-4-0125-preview"  # GPT-4.1
GEMINI_MODEL = "gemini-2.0-flash-exp"  # Latest Gemini with logprobs
CLAUDE_MODEL = "claude-opus-4-1-20250805"  # Claude Opus 4.1

# Initialize API clients
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Rate limiting configuration
GPT_DELAY = 0.5  # seconds between GPT API calls
GEMINI_DELAY = 6.0  # seconds between Gemini API calls (rate limited)
CLAUDE_DELAY = 1.0  # seconds between Claude API calls

# Output configuration
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "closed_source_evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Caching configuration
CACHE_FILE = os.path.join(OUTPUT_DIR, "api_cache.json")
USE_CACHE = True  # Set to False to force fresh API calls

def load_ordinary_meaning_questions():
    """Load all ordinary meaning questions from both parts of the dataset."""
    questions = []

    # Load part 1 questions (first 50)
    part1_path = os.path.join(BASE_DIR, "data", "instruct_model_comparison_results.csv")
    df_part1 = pd.read_csv(part1_path)
    questions_part1 = df_part1['prompt'].unique()[:50]  # Get first 50 unique questions
    questions.extend(questions_part1)

    # Load part 2 questions - extract from the actual survey results
    # since question_list_part_2.txt contains duplicate questions
    part2_path = os.path.join(BASE_DIR, "data", "word_meaning_survey_results_part_2.csv")
    survey_df_part2 = pd.read_csv(part2_path, skiprows=1)
    questions_part2 = []

    # Extract questions from column headers
    for col in survey_df_part2.columns:
        if 'Left = No, Right = Yes' in col:
            parts = col.split(' - ')
            if len(parts) >= 2:
                question = parts[-1].strip()
                if question.endswith('?') and question not in questions_part2:
                    questions_part2.append(question)

    # Take the first 50 unique questions from part 2 (there are 55 total)
    questions.extend(questions_part2[:50])

    print(f"Loaded {len(questions_part1)} questions from part 1 and {len(questions_part2[:50])} questions from part 2")
    print(f"Total questions: {len(questions)}")
    return list(questions)

def load_human_survey_data(return_full_data=False):
    """Load and process human survey data for comparison from both parts.

    This function now properly combines responses from both survey parts.
    For questions that appear in both parts, it pools all responses together
    before calculating statistics.

    Args:
        return_full_data: If True, returns both aggregated means and full response distributions

    Returns:
        If return_full_data is False: Dictionary of question -> mean response (0-1 scale)
        If return_full_data is True: Tuple of (means_dict, full_responses_dict)
    """
    # Use a dictionary to collect all responses for each question
    question_responses = {}

    # Load part 1 survey results (skip the first header row)
    part1_survey_path = os.path.join(BASE_DIR, "data", "word_meaning_survey_results.csv")
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
    part2_survey_path = os.path.join(BASE_DIR, "data", "word_meaning_survey_results_part_2.csv")
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
                        # Convert 0-100 scale to 0-1 scale and add to existing responses
                        if question not in question_responses:
                            question_responses[question] = []
                        question_responses[question].extend((valid_values / 100.0).tolist())

    # Calculate means from combined responses
    human_data = {}
    full_responses = {} if return_full_data else None

    for question, responses in question_responses.items():
        human_data[question] = np.mean(responses)
        if return_full_data:
            full_responses[question] = responses

    print(f"Loaded human survey data for {len(human_data)} questions total")

    # Print sample sizes for verification
    sample_sizes = [len(responses) for responses in question_responses.values()]
    print(f"Response counts per question - Min: {min(sample_sizes)}, Max: {max(sample_sizes)}, Mean: {np.mean(sample_sizes):.1f}")

    if return_full_data:
        return human_data, full_responses
    return human_data

def evaluate_gpt(prompt, binary_format=True):
    """
    Evaluate a prompt using GPT-4.1 with logprobs.

    Returns:
        tuple: (response_text, token_1_prob, token_2_prob, confidence_value, weighted_confidence)
    """
    try:
        if binary_format:
            # Binary yes/no format with logprobs
            formatted_prompt = f"{prompt} Answer only 'Yes' or 'No'."

            response = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0,
                max_tokens=10,
                logprobs=True,
                top_logprobs=20
            )

            # Extract response text
            response_text = response.choices[0].message.content.strip()

            # Extract logprobs for Yes/No
            token_1_prob = 0.0
            token_2_prob = 0.0

            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                first_token_data = response.choices[0].logprobs.content[0]
                for logprob_item in first_token_data.top_logprobs:
                    if logprob_item.token.lower() in ['yes', 'y']:
                        token_1_prob = np.exp(logprob_item.logprob)
                    elif logprob_item.token.lower() in ['no', 'n']:
                        token_2_prob = np.exp(logprob_item.logprob)

            return response_text, token_1_prob, token_2_prob, None, None

        else:
            # Confidence format
            formatted_prompt = f"{prompt} How confident are you that the answer is 'Yes', on a scale from 0 (not confident) to 100 (most confident)? Answer only with a number."

            response = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0,
                max_tokens=10,
                logprobs=True,
                top_logprobs=20
            )

            response_text = response.choices[0].message.content.strip()

            # Extract confidence value
            try:
                confidence_value = int(''.join(filter(str.isdigit, response_text)))
            except:
                confidence_value = None

            # Calculate weighted confidence from logprobs
            weighted_confidence = calculate_weighted_confidence_gpt(response.choices[0].logprobs)

            return response_text, None, None, confidence_value, weighted_confidence

    except Exception as e:
        print(f"Error with GPT evaluation: {e}")
        return None, 0.0, 0.0, None, None

def calculate_weighted_confidence_gpt(logprobs_data):
    """Calculate weighted confidence from GPT logprobs."""
    if not logprobs_data or not logprobs_data.content:
        return None

    try:
        # Collect all possible confidence values and their probabilities
        confidence_probs = {}

        # Look at first few tokens to capture full numbers
        for token_data in logprobs_data.content[:3]:
            for logprob_item in token_data.top_logprobs:
                token = logprob_item.token.strip()
                if token.isdigit():
                    value = int(token)
                    if 0 <= value <= 100:
                        prob = np.exp(logprob_item.logprob)
                        if value not in confidence_probs:
                            confidence_probs[value] = 0
                        confidence_probs[value] += prob

        # Normalize probabilities
        total_prob = sum(confidence_probs.values())
        if total_prob > 0:
            normalized = {k: v/total_prob for k, v in confidence_probs.items()}
            # Calculate weighted average
            weighted_avg = sum(value * prob for value, prob in normalized.items())
            return weighted_avg

    except Exception as e:
        print(f"Error calculating weighted confidence: {e}")

    return None

def evaluate_gemini(prompt, binary_format=True):
    """
    Evaluate a prompt using Gemini with logprobs.

    Returns:
        tuple: (response_text, token_1_prob, token_2_prob, confidence_value, weighted_confidence)
    """
    try:
        generation_config = {
            "temperature": 0.0,
            "max_output_tokens": 10,
            "response_logprobs": True,
            "logprobs": 19  # Maximum allowed
        }

        if binary_format:
            formatted_prompt = f"{prompt} Answer only 'Yes' or 'No'."
        else:
            formatted_prompt = f"{prompt} How confident are you that the answer is 'Yes', on a scale from 0 (not confident) to 100 (most confident)? Answer only with a number."

        response = gemini_model.generate_content(
            formatted_prompt,
            generation_config=generation_config
        )

        response_text = response.text.strip() if response.text else ""

        if binary_format:
            # Extract token probabilities for Yes/No
            token_1_prob = 0.0
            token_2_prob = 0.0

            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
                    if hasattr(candidate.logprobs_result, 'top_candidates'):
                        first_position = candidate.logprobs_result.top_candidates[0]
                        for cand in first_position.candidates:
                            if cand.token.lower() in ['yes', 'y']:
                                token_1_prob = np.exp(cand.log_probability)
                            elif cand.token.lower() in ['no', 'n']:
                                token_2_prob = np.exp(cand.log_probability)

            return response_text, token_1_prob, token_2_prob, None, None
        else:
            # Extract confidence value
            try:
                confidence_value = int(''.join(filter(str.isdigit, response_text)))
            except:
                confidence_value = None

            # Calculate weighted confidence from logprobs
            weighted_confidence = None
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'logprobs_result'):
                    weighted_confidence = calculate_weighted_confidence_gemini(candidate.logprobs_result)

            return response_text, None, None, confidence_value, weighted_confidence

    except Exception as e:
        print(f"Error with Gemini evaluation: {e}")
        return None, 0.0, 0.0, None, None

def calculate_weighted_confidence_gemini(logprobs_data):
    """Calculate weighted confidence from Gemini logprobs (handles multi-token numbers)."""
    if not logprobs_data or not hasattr(logprobs_data, 'top_candidates'):
        return None

    try:
        # Get available positions
        first_pos = logprobs_data.top_candidates[0] if len(logprobs_data.top_candidates) > 0 else None
        second_pos = logprobs_data.top_candidates[1] if len(logprobs_data.top_candidates) > 1 else None
        third_pos = logprobs_data.top_candidates[2] if len(logprobs_data.top_candidates) > 2 else None

        if not first_pos:
            return None

        # Store probabilities for different number formations
        one_digit_probs = {}   # e.g., "5" alone
        two_digit_probs = {}   # e.g., "5" + "0" = 50
        three_digit_probs = {} # e.g., "1" + "0" + "0" = 100

        # Process all first position candidates
        for first_cand in first_pos.candidates[:19]:
            first_token = first_cand.token.strip()
            first_log_prob = first_cand.log_probability
            first_prob = np.exp(first_log_prob)

            # Check if it's a single digit
            if first_token.isdigit() and len(first_token) == 1:
                first_digit = int(first_token)

                # Track probability of this being a standalone single digit
                one_digit_prob_for_this = first_prob
                two_digit_prob_sum = 0.0

                # Check for two-digit formations
                if second_pos and 1 <= first_digit <= 9:
                    for second_cand in second_pos.candidates[:19]:
                        second_token = second_cand.token.strip()
                        second_log_prob = second_cand.log_probability
                        second_prob = np.exp(second_log_prob)

                        if second_token.isdigit() and len(second_token) == 1:
                            second_digit = int(second_token)
                            two_digit_value = first_digit * 10 + second_digit

                            # Check for three-digit formation (only 100 is valid)
                            if two_digit_value == 10 and third_pos:
                                for third_cand in third_pos.candidates[:19]:
                                    third_token = third_cand.token.strip()
                                    third_log_prob = third_cand.log_probability
                                    third_prob = np.exp(third_log_prob)

                                    if third_token == "0":
                                        # This forms 100
                                        combined_prob = first_prob * second_prob * third_prob
                                        if 100 not in three_digit_probs:
                                            three_digit_probs[100] = 0
                                        three_digit_probs[100] += combined_prob

                            # Add two-digit number probability
                            if 10 <= two_digit_value <= 99:
                                combined_prob = first_prob * second_prob

                                # For "10", subtract probability that continues to form 100
                                if two_digit_value == 10 and third_pos:
                                    third_zero_prob = 0.0
                                    for third_cand in third_pos.candidates[:19]:
                                        if third_cand.token.strip() == "0":
                                            third_zero_prob = np.exp(third_cand.log_probability)
                                            break
                                    combined_prob *= (1 - third_zero_prob)

                                if two_digit_value not in two_digit_probs:
                                    two_digit_probs[two_digit_value] = 0
                                two_digit_probs[two_digit_value] += combined_prob
                                two_digit_prob_sum += combined_prob

                # Calculate remaining probability for single digit
                if second_pos and 1 <= first_digit <= 9:
                    # Find total probability of second position being a digit
                    second_digit_prob = 0.0
                    for second_cand in second_pos.candidates[:19]:
                        if second_cand.token.strip().isdigit() and len(second_cand.token.strip()) == 1:
                            second_digit_prob += np.exp(second_cand.log_probability)

                    # Single digit prob = first prob * (1 - prob of second being digit)
                    one_digit_prob_for_this *= (1 - second_digit_prob)

                # Add single digit probability (0-9)
                if 0 <= first_digit <= 9:
                    if first_digit not in one_digit_probs:
                        one_digit_probs[first_digit] = 0
                    one_digit_probs[first_digit] += one_digit_prob_for_this

            # Check if it's already a complete number token (Gemini sometimes outputs full numbers)
            elif first_token.isdigit():
                value = int(first_token)
                if value == 100:
                    if 100 not in three_digit_probs:
                        three_digit_probs[100] = 0
                    three_digit_probs[100] += first_prob
                elif 10 <= value <= 99:
                    if value not in two_digit_probs:
                        two_digit_probs[value] = 0
                    two_digit_probs[value] += first_prob
                elif 0 <= value <= 9:
                    if value not in one_digit_probs:
                        one_digit_probs[value] = 0
                    one_digit_probs[value] += first_prob

        # Combine all probabilities
        all_probs = {}
        all_probs.update(one_digit_probs)
        all_probs.update(two_digit_probs)
        all_probs.update(three_digit_probs)

        # Calculate total probability mass and weighted average
        total_prob = sum(all_probs.values())

        if total_prob > 0 and all_probs:
            # Normalize probabilities
            normalized = {k: v/total_prob for k, v in all_probs.items()}

            # Calculate weighted average
            weighted_avg = sum(value * prob for value, prob in normalized.items())
            return weighted_avg

    except Exception as e:
        print(f"Error calculating Gemini weighted confidence: {e}")

    return None

def evaluate_random_baseline(prompt, binary_format=True):
    """
    Random baseline that returns random predictions.

    Returns:
        tuple: (response_text, token_1_prob, token_2_prob, confidence_value, weighted_confidence)
    """
    if binary_format:
        # Random Yes/No
        response_text = random.choice(['Yes', 'No'])
        # Random probabilities that sum to ~1
        yes_prob = random.random()
        no_prob = 1 - yes_prob
        return response_text, yes_prob, no_prob, None, None
    else:
        # Random confidence between 0-100
        confidence_value = random.randint(0, 100)
        return str(confidence_value), None, None, confidence_value, confidence_value

def evaluate_normal_baseline(prompt, binary_format=True, mean=50, std=None):
    """
    Normal distribution baseline centered at 50 with empirical or specified std.

    Args:
        prompt: The prompt (unused, kept for consistency)
        binary_format: Whether to return binary or confidence format
        mean: Mean of the normal distribution (default 50)
        std: Standard deviation (if None, will be set empirically)

    Returns:
        tuple: (response_text, token_1_prob, token_2_prob, confidence_value, weighted_confidence)
    """
    # Use a default std if not specified (will be updated with empirical value)
    if std is None:
        std = 15  # Reasonable default, will be overridden with empirical value

    if binary_format:
        # Convert normal distribution sample to probability
        confidence_sample = np.random.normal(mean, std)
        # Clip to [0, 100] range
        confidence_sample = np.clip(confidence_sample, 0, 100)
        # Convert to probability
        prob = confidence_sample / 100.0

        # Decide Yes/No based on probability
        response_text = 'Yes' if prob > 0.5 else 'No'
        yes_prob = prob
        no_prob = 1 - prob
        return response_text, yes_prob, no_prob, None, None
    else:
        # Sample from normal distribution for confidence
        confidence_value = np.random.normal(mean, std)
        # Clip to [0, 100] range
        confidence_value = int(np.clip(confidence_value, 0, 100))
        return str(confidence_value), None, None, confidence_value, confidence_value

def evaluate_claude(prompt, binary_format=True):
    """
    Evaluate a prompt using Claude Opus 4.1.
    Note: Claude doesn't provide logprobs, so we only get verbalized confidence.

    Returns:
        tuple: (response_text, token_1_prob, token_2_prob, confidence_value, weighted_confidence)
    """
    try:
        if binary_format:
            formatted_prompt = f"{prompt} Answer only 'Yes' or 'No'."
        else:
            formatted_prompt = f"{prompt} How confident are you that the answer is 'Yes', on a scale from 0 (not confident) to 100 (most confident)? Answer only with a number."

        response = anthropic_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": formatted_prompt}]
        )

        response_text = response.content[0].text.strip() if response.content else ""

        if binary_format:
            # Claude doesn't provide logprobs
            return response_text, 0.0, 0.0, None, None
        else:
            # Extract confidence value
            try:
                confidence_value = int(''.join(filter(str.isdigit, response_text)))
            except:
                confidence_value = None

            # Claude doesn't provide logprobs for weighted confidence
            return response_text, None, None, confidence_value, confidence_value

    except Exception as e:
        print(f"Error with Claude evaluation: {e}")
        return None, 0.0, 0.0, None, None

def load_cached_results():
    """Load cached API results if available."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cached = json.load(f)
                print(f"Loaded cached results from {CACHE_FILE}")
                return cached
        except Exception as e:
            print(f"Error loading cache: {e}")
    return {}

def save_cached_results(cache):
    """Save API results to cache."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
        print(f"Saved cache to {CACHE_FILE}")
    except Exception as e:
        print(f"Error saving cache: {e}")

def is_cached_result_complete(cached_result):
    """Check if a cached result has all required fields with non-None values.

    Note: Random baseline fields are optional since they were added later.
    """
    # Core model fields that must be present
    required_fields = [
        'gpt_response', 'gpt_yes_prob', 'gpt_no_prob', 'gpt_confidence', 'gpt_weighted_confidence',
        'gemini_response', 'gemini_yes_prob', 'gemini_no_prob', 'gemini_confidence', 'gemini_weighted_confidence',
        'claude_response', 'claude_confidence'
    ]

    for field in required_fields:
        if field not in cached_result or cached_result[field] is None:
            return False

    # Random fields are optional (added later to the system)
    # If they exist, they should be complete
    has_any_random = any(f in cached_result for f in ['random_response', 'random_confidence'])
    if has_any_random:
        random_fields = ['random_response', 'random_yes_prob', 'random_no_prob', 'random_confidence']
        for field in random_fields:
            if field not in cached_result or cached_result[field] is None:
                return False

    return True

def evaluate_all_models(questions):
    """Evaluate all models on all questions."""
    results = []

    # Load cache if using cached results
    cache = load_cached_results() if USE_CACHE else {}
    cache_updated = False

    for idx, question in enumerate(questions):
        print(f"\nProcessing question {idx+1}/{len(questions)}: {question[:80]}...")

        # Check if we have cached results for this question
        question_key = question[:100]  # Use first 100 chars as key

        # Check if we have COMPLETE cached results
        has_complete_cache = (USE_CACHE and
                             question_key in cache and
                             is_cached_result_complete(cache[question_key]))

        if has_complete_cache:
            print("  Using cached results (complete)...")
            cached_result = cache[question_key]
            gpt_binary = (cached_result.get('gpt_response'),
                         cached_result.get('gpt_yes_prob', 0.0),
                         cached_result.get('gpt_no_prob', 0.0),
                         None, None)
            gpt_conf = (None, None, None,
                       cached_result.get('gpt_confidence'),
                       cached_result.get('gpt_weighted_confidence'))
            gemini_binary = (cached_result.get('gemini_response'),
                           cached_result.get('gemini_yes_prob', 0.0),
                           cached_result.get('gemini_no_prob', 0.0),
                           None, None)
            gemini_conf = (None, None, None,
                         cached_result.get('gemini_confidence'),
                         cached_result.get('gemini_weighted_confidence'))
            claude_binary = (cached_result.get('claude_response'),
                           0.0, 0.0, None, None)
            claude_conf = (None, None, None,
                         cached_result.get('claude_confidence'),
                         cached_result.get('claude_confidence'))  # Claude uses same for both
            random_binary = (cached_result.get('random_response'),
                           cached_result.get('random_yes_prob', 0.5),
                           cached_result.get('random_no_prob', 0.5),
                           None, None)
            random_conf = (None, None, None,
                         cached_result.get('random_confidence'),
                         cached_result.get('random_confidence'))
        else:
            # Make fresh API calls or re-run incomplete evaluations
            if USE_CACHE and question_key in cache:
                print("  Cached results incomplete, re-running missing evaluations...")
                cached_result = cache[question_key]
            else:
                print("  Making fresh API calls...")
                cached_result = None

            # Check what needs to be evaluated
            needs_gpt = (cached_result is None or
                        cached_result.get('gpt_response') is None or
                        cached_result.get('gpt_confidence') is None)
            needs_gemini = (cached_result is None or
                           cached_result.get('gemini_response') is None or
                           cached_result.get('gemini_confidence') is None)
            needs_claude = (cached_result is None or
                           cached_result.get('claude_response') is None or
                           cached_result.get('claude_confidence') is None)
            needs_random = (cached_result is None or
                           cached_result.get('random_response') is None or
                           cached_result.get('random_confidence') is None)

            # GPT-4.1 evaluation
            if needs_gpt:
                print("  Evaluating GPT-4.1...")
                # Binary format
                gpt_binary = evaluate_gpt(question, binary_format=True)
                time.sleep(GPT_DELAY)
                # Confidence format
                gpt_conf = evaluate_gpt(question, binary_format=False)
                time.sleep(GPT_DELAY)
            else:
                print("  Using cached GPT-4.1 results...")
                gpt_binary = (cached_result.get('gpt_response'),
                            cached_result.get('gpt_yes_prob', 0.0),
                            cached_result.get('gpt_no_prob', 0.0),
                            None, None)
                gpt_conf = (None, None, None,
                          cached_result.get('gpt_confidence'),
                          cached_result.get('gpt_weighted_confidence'))

            # Gemini evaluation
            if needs_gemini:
                print("  Evaluating Gemini...")
                # Binary format
                gemini_binary = evaluate_gemini(question, binary_format=True)
                time.sleep(GEMINI_DELAY)
                # Confidence format
                gemini_conf = evaluate_gemini(question, binary_format=False)
                time.sleep(GEMINI_DELAY)
            else:
                print("  Using cached Gemini results...")
                gemini_binary = (cached_result.get('gemini_response'),
                              cached_result.get('gemini_yes_prob', 0.0),
                              cached_result.get('gemini_no_prob', 0.0),
                              None, None)
                gemini_conf = (None, None, None,
                            cached_result.get('gemini_confidence'),
                            cached_result.get('gemini_weighted_confidence'))

            # Claude evaluation
            if needs_claude:
                print("  Evaluating Claude Opus 4.1...")
                # Binary format
                claude_binary = evaluate_claude(question, binary_format=True)
                time.sleep(CLAUDE_DELAY)
                # Confidence format
                claude_conf = evaluate_claude(question, binary_format=False)
                time.sleep(CLAUDE_DELAY)
            else:
                print("  Using cached Claude results...")
                claude_binary = (cached_result.get('claude_response'),
                              0.0, 0.0, None, None)
                claude_conf = (None, None, None,
                            cached_result.get('claude_confidence'),
                            cached_result.get('claude_confidence'))

            # Random baseline evaluation
            if needs_random:
                print("  Evaluating Random Baseline...")
                # Binary format
                random_binary = evaluate_random_baseline(question, binary_format=True)
                # Confidence format
                random_conf = evaluate_random_baseline(question, binary_format=False)
            else:
                print("  Using cached Random baseline results...")
                random_binary = (cached_result.get('random_response'),
                              cached_result.get('random_yes_prob', 0.5),
                              cached_result.get('random_no_prob', 0.5),
                              None, None)
                random_conf = (None, None, None,
                            cached_result.get('random_confidence'),
                            cached_result.get('random_confidence'))

            cache_updated = True

        # Store results
        result = {
            'question': question,
            'gpt_response': gpt_binary[0],
            'gpt_yes_prob': gpt_binary[1],
            'gpt_no_prob': gpt_binary[2],
            'gpt_relative_prob': gpt_binary[1] / (gpt_binary[1] + gpt_binary[2]) if (gpt_binary[1] + gpt_binary[2]) > 0 else 0.5,
            'gpt_confidence': gpt_conf[3],
            'gpt_weighted_confidence': gpt_conf[4],
            'gemini_response': gemini_binary[0],
            'gemini_yes_prob': gemini_binary[1],
            'gemini_no_prob': gemini_binary[2],
            'gemini_relative_prob': gemini_binary[1] / (gemini_binary[1] + gemini_binary[2]) if (gemini_binary[1] + gemini_binary[2]) > 0 else 0.5,
            'gemini_confidence': gemini_conf[3],
            'gemini_weighted_confidence': gemini_conf[4],
            'claude_response': claude_binary[0],
            'claude_confidence': claude_conf[3],
            'random_response': random_binary[0],
            'random_yes_prob': random_binary[1],
            'random_no_prob': random_binary[2],
            'random_relative_prob': random_binary[1] / (random_binary[1] + random_binary[2]) if (random_binary[1] + random_binary[2]) > 0 else 0.5,
            'random_confidence': random_conf[3]
        }
        results.append(result)

        # Update cache if we made fresh API calls or had incomplete results
        if not has_complete_cache:
            cache[question_key] = result

        # Save intermediate results
        if (idx + 1) % 10 == 0:
            df = pd.DataFrame(results)
            df.to_csv(f"{OUTPUT_DIR}/intermediate_results_{idx+1}.csv", index=False)
            print(f"  Saved intermediate results ({idx+1} questions processed)")

    # Save updated cache if needed
    if cache_updated:
        save_cached_results(cache)

    return pd.DataFrame(results)

def calculate_correlations(df):
    """Calculate correlations between models."""
    correlations = {}

    # Relative probability correlations (GPT vs Gemini only, since Claude doesn't have logprobs)
    if 'gpt_relative_prob' in df.columns and 'gemini_relative_prob' in df.columns:
        corr, p_value = pearsonr(df['gpt_relative_prob'].fillna(0.5),
                                 df['gemini_relative_prob'].fillna(0.5))
        correlations['gpt_gemini_relative_prob'] = {'correlation': corr, 'p_value': p_value}

    # Weighted confidence correlations
    pairs = [
        ('gpt_weighted_confidence', 'gemini_weighted_confidence'),
        ('gpt_weighted_confidence', 'claude_confidence'),
        ('gemini_weighted_confidence', 'claude_confidence')
    ]

    for col1, col2 in pairs:
        if col1 in df.columns and col2 in df.columns:
            # Convert to 0-1 scale for comparison
            vals1 = df[col1].fillna(50) / 100
            vals2 = df[col2].fillna(50) / 100
            corr, p_value = pearsonr(vals1, vals2)
            correlations[f"{col1.split('_')[0]}_{col2.split('_')[0]}_confidence"] = {
                'correlation': corr,
                'p_value': p_value
            }

    return correlations

def bootstrap_mae(values, n_bootstrap=10000, confidence_level=0.95, seed=42):
    """Calculate bootstrapped confidence intervals for MAE.

    Args:
        values: List of absolute errors
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals (default 95%)
        seed: Random seed for reproducibility

    Returns:
        tuple: (mean_mae, ci_lower, ci_upper)
    """
    if not values or len(values) == 0:
        return None, None, None

    values_array = np.array(values)

    # Use scipy's bootstrap for confidence intervals
    rng = np.random.default_rng(seed)
    res = bootstrap(
        (values_array,),
        np.mean,
        n_resamples=n_bootstrap,
        confidence_level=confidence_level,
        random_state=rng,
        method='percentile'
    )

    mean_mae = np.mean(values_array)
    ci_lower = res.confidence_interval.low
    ci_upper = res.confidence_interval.high

    return mean_mae, ci_lower, ci_upper

def bootstrap_mae_difference(model_values, baseline_values, n_bootstrap=10000, confidence_level=0.95, seed=42):
    """Calculate bootstrapped confidence intervals and p-value for the difference between model MAE and baseline MAE.

    Args:
        model_values: List of absolute errors for the model
        baseline_values: List of absolute errors for the baseline (or scalar for fixed baseline)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        seed: Random seed

    Returns:
        tuple: (mean_difference, ci_lower, ci_upper, p_value)
    """
    if not model_values or len(model_values) == 0:
        return None, None, None, None

    model_array = np.array(model_values)

    # Handle both scalar and array baseline values
    if np.isscalar(baseline_values):
        baseline_array = np.full_like(model_array, baseline_values)
    else:
        baseline_array = np.array(baseline_values)
        if len(baseline_array) != len(model_array):
            # If lengths don't match, compute mean baseline MAE
            baseline_mae = np.mean(baseline_array)
            baseline_array = np.full_like(model_array, baseline_mae)

    # Calculate observed difference
    observed_diff = np.mean(model_array) - np.mean(baseline_array)

    # Bootstrap for confidence intervals and p-value
    rng = np.random.default_rng(seed)
    n = len(model_array)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = rng.choice(n, size=n, replace=True)
        model_sample = model_array[idx]
        baseline_sample = baseline_array[idx] if len(baseline_array) == n else baseline_array

        # Calculate difference in MAE
        diff = np.mean(model_sample) - np.mean(baseline_sample)
        bootstrap_diffs.append(diff)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Calculate confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha/2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha/2))

    # Calculate two-tailed p-value
    # Null hypothesis: difference = 0
    p_value = np.mean(np.abs(bootstrap_diffs - np.mean(bootstrap_diffs)) >= np.abs(observed_diff))

    # Alternative p-value calculation (proportion of bootstrap samples on wrong side of 0)
    if observed_diff > 0:
        p_value = 2 * min(np.mean(bootstrap_diffs <= 0), np.mean(bootstrap_diffs >= 0))
    else:
        p_value = 2 * min(np.mean(bootstrap_diffs >= 0), np.mean(bootstrap_diffs <= 0))

    return observed_diff, ci_lower, ci_upper, p_value

def calculate_baseline_performance(model_df, human_data):
    """Calculate baseline performance metrics for key baselines only.

    Args:
        model_df: DataFrame with model predictions
        human_data: Dictionary of human mean responses

    Returns:
        dict: Contains MAE for:
            - 'always_50': Model that always predicts 50%
            - 'normal_human': Normal distribution using human mean and std
    """
    baselines = {}

    if not human_data:
        return baselines

    # Convert human data to list
    human_questions = list(human_data.keys())
    human_values = list(human_data.values())

    # Calculate human mean and std for the normal baseline
    human_mean = np.mean(human_values)
    human_std = np.std(human_values)

    # Baseline 1: Always predict 50%
    mae_always_50 = []
    for human_val in human_values:
        mae_always_50.append(abs(0.5 - human_val))

    # Calculate bootstrapped confidence intervals for MAE
    mae_mean, mae_ci_lower, mae_ci_upper = bootstrap_mae(mae_always_50)

    baselines['always_50'] = {
        'mae': mae_mean,
        'std': np.std(mae_always_50),
        'mae_ci_lower': mae_ci_lower,
        'mae_ci_upper': mae_ci_upper,
        'description': 'Always 50%'
    }

    # Baseline 2: Normal distribution using human mean and std
    mae_normal_human = []
    np.random.seed(43)  # For reproducibility
    for human_val in human_values:
        # Generate prediction from normal distribution with human parameters
        # Convert human mean/std from 0-1 scale to 0-100 scale for sampling
        normal_value = np.random.normal(human_mean * 100, human_std * 100)
        # Clip to [0, 100] range and convert back to 0-1 scale
        normal_value = np.clip(normal_value, 0, 100) / 100.0
        mae = abs(normal_value - human_val)
        mae_normal_human.append(mae)

    # Calculate bootstrapped confidence intervals
    mae_mean, mae_ci_lower, mae_ci_upper = bootstrap_mae(mae_normal_human)

    baselines['normal_human'] = {
        'mae': mae_mean,
        'std': np.std(mae_normal_human),
        'mae_ci_lower': mae_ci_lower,
        'mae_ci_upper': mae_ci_upper,
        'human_mean': human_mean * 100,  # Store in percentage scale for display
        'human_std': human_std * 100,     # Store in percentage scale for display
        'description': f'N({human_mean*100:.0f}, {human_std*100:.0f})'
    }

    return baselines

def compare_with_human_data(model_df, human_data, baselines, calculate_bootstrap=True):
    """Compare model outputs with human survey data and baselines.

    Args:
        model_df: DataFrame with model predictions
        human_data: Dictionary of human survey responses (means)
        baselines: Dictionary of baseline results
        calculate_bootstrap: Whether to calculate bootstrapped confidence intervals

    Returns:
        dict: Comparison metrics including MAE with bootstrapped CIs and comparisons to baselines
    """
    comparisons = {}

    if not human_data:
        return comparisons

    # Get human values for baseline calculations
    human_values = list(human_data.values())

    # Match questions and calculate MAE for each model
    for model_name in ['gpt', 'gemini', 'claude']:
        mae_values = []
        matched_questions = []
        paired_human_values = []  # Store paired human values for difference calculation
        model_predictions = []  # Store model predictions for baseline comparisons

        for idx, row in model_df.iterrows():
            question = row['question']
            # Try to find matching human data
            human_value = None

            for human_q, human_val in human_data.items():
                if question in human_q or human_q in question:
                    human_value = human_val
                    break

            if human_value is not None:
                # Get model confidence (use weighted for GPT/Gemini, regular for Claude)
                if model_name in ['claude']:
                    model_value = row[f'{model_name}_confidence']
                else:
                    model_value = row[f'{model_name}_weighted_confidence']
                    # Fall back to regular confidence if weighted is None
                    if pd.isna(model_value):
                        model_value = row[f'{model_name}_confidence']

                if not pd.isna(model_value):
                    # Convert model confidence to 0-1 scale
                    model_value = model_value / 100.0

                    # Calculate MAE
                    mae = abs(model_value - human_value)
                    mae_values.append(mae)

                    matched_questions.append(question)
                    paired_human_values.append(human_value)
                    model_predictions.append(model_value)

        if mae_values:
            # Basic statistics
            comparisons[model_name] = {
                'mae': np.mean(mae_values),
                'std': np.std(mae_values),
                'n_matched': len(mae_values),
                'mae_values': mae_values,
                'questions': matched_questions
            }

            if calculate_bootstrap:
                # Calculate bootstrapped confidence intervals for MAE
                mae_mean, mae_ci_lower, mae_ci_upper = bootstrap_mae(mae_values)
                comparisons[model_name]['mae_bootstrap'] = mae_mean
                comparisons[model_name]['mae_ci_lower'] = mae_ci_lower
                comparisons[model_name]['mae_ci_upper'] = mae_ci_upper

                # Compare to Always 50% baseline
                baseline_50_mae_values = [abs(0.5 - hv) for hv in paired_human_values]
                baseline_50_mae = np.mean(baseline_50_mae_values)

                # Calculate difference from 50% baseline
                diff_mean_50, diff_ci_lower_50, diff_ci_upper_50, p_value_50 = bootstrap_mae_difference(
                    mae_values, baseline_50_mae_values)

                comparisons[model_name]['vs_always_50'] = {
                    'mae_diff': diff_mean_50,
                    'mae_diff_ci_lower': diff_ci_lower_50,
                    'mae_diff_ci_upper': diff_ci_upper_50,
                    'p_value': p_value_50,
                    'baseline_mae': np.mean(baseline_50_mae_values)
                }

                # Compare to Normal(human μ, σ) baseline
                # Generate predictions from normal distribution for the same questions
                np.random.seed(43)  # Same seed as baseline calculation for consistency
                human_mean = np.mean(human_values)
                human_std = np.std(human_values)

                normal_human_mae_values = []
                for hv in paired_human_values:
                    normal_value = np.random.normal(human_mean * 100, human_std * 100)
                    normal_value = np.clip(normal_value, 0, 100) / 100.0
                    mae_normal = abs(normal_value - hv)
                    normal_human_mae_values.append(mae_normal)

                # Calculate difference from Normal(human) baseline
                diff_mean_normal, diff_ci_lower_normal, diff_ci_upper_normal, p_value_normal = bootstrap_mae_difference(
                    mae_values, normal_human_mae_values)

                comparisons[model_name]['vs_normal_human'] = {
                    'mae_diff': diff_mean_normal,
                    'mae_diff_ci_lower': diff_ci_lower_normal,
                    'mae_diff_ci_upper': diff_ci_upper_normal,
                    'p_value': p_value_normal,
                    'baseline_mae': np.mean(normal_human_mae_values)
                }

    # Also calculate correlation with human data
    for model_name in ['gpt', 'gemini', 'claude']:
        if model_name in comparisons:
            model_values = []
            human_values = []

            for idx, row in model_df.iterrows():
                question = row['question']
                human_value = None

                for human_q, human_val in human_data.items():
                    if question in human_q or human_q in question:
                        human_value = human_val
                        break

                if human_value is not None:
                    if model_name in ['claude', 'random']:
                        model_value = row[f'{model_name}_confidence']
                    else:
                        model_value = row[f'{model_name}_weighted_confidence']
                        if pd.isna(model_value):
                            model_value = row[f'{model_name}_confidence']

                    if not pd.isna(model_value):
                        model_values.append(model_value / 100.0)
                        human_values.append(human_value)

            if len(model_values) >= 2:
                corr, p_value = pearsonr(model_values, human_values)
                comparisons[model_name]['correlation'] = corr
                comparisons[model_name]['p_value'] = p_value

    return comparisons

def generate_latex_tables(comparisons, baselines):
    """Generate LaTeX tables for the MAE results.

    Args:
        comparisons: Dictionary with model comparisons
        baselines: Dictionary with baseline results

    Returns:
        str: LaTeX table code
    """
    latex_tables = []

    # Table 1: Main MAE Results with CI (formatted like requested)
    table1 = r"""\begin{table}[H]
\centering
\begin{tabular}{lcc}
\hline
\textbf{Model} & \textbf{MAE} & \textbf{95\% CI} \\
\hline
"""

    # Add baseline results first (Random and Equanimity style)
    baseline_names_map = {
        'always_50': 'Random (Always 50%)',
        'normal_human': 'Equanimity (Human Normal)'
    }

    for baseline_name in ['always_50', 'normal_human']:
        if baseline_name in baselines:
            b = baselines[baseline_name]
            mae = b['mae']
            ci_lower = b['mae_ci_lower']
            ci_upper = b['mae_ci_upper']
            name = baseline_names_map.get(baseline_name, b['description'])
            table1 += f"{name} & {mae:.3f} & [{ci_lower:.3f}, {ci_upper:.3f}] \\\\\n"

    table1 += r"\hline" + "\n"

    # Add model results
    model_names = {'gpt': 'GPT-4.1', 'gemini': 'Gemini-2.0-flash', 'claude': 'Claude Opus 4.1'}
    for model_key in ['gpt', 'claude', 'gemini']:  # Reorder as in example
        if model_key in comparisons:
            m = comparisons[model_key]
            mae = m['mae']
            ci_lower = m['mae_ci_lower']
            ci_upper = m['mae_ci_upper']
            name = model_names[model_key]
            table1 += f"{name} & {mae:.3f} & [{ci_lower:.3f}, {ci_upper:.3f}] \\\\\n"

    table1 += r"""\hline
\end{tabular}
\caption{Mean Absolute Error (MAE) of Models and Baselines. Lower MAE values indicate better alignment with human judgments.}
\label{tab:mae_results}
\end{table}
\FloatBarrier"""

    latex_tables.append(table1)

    # Table 2: MAE Differences from Baseline with p-values
    table2 = r"""\begin{table}[H]
\centering
\begin{tabular}{lccc}
\hline
\textbf{Model} & \textbf{MAE Difference} & \textbf{95\% CI} & \textbf{p-value} \\
 & \textbf{from Random} & & \\
\hline
"""

    # Add model comparisons vs Random (Always 50%)
    for model_key in ['gpt', 'claude', 'gemini']:
        if model_key in comparisons:
            m = comparisons[model_key]
            name = model_names[model_key]

            # vs Always 50%
            if 'vs_always_50' in m:
                diff_50 = m['vs_always_50']['mae_diff']
                ci_lower_50 = m['vs_always_50']['mae_diff_ci_lower']
                ci_upper_50 = m['vs_always_50']['mae_diff_ci_upper']
                p_value_50 = m['vs_always_50'].get('p_value', None)

                # Format p-value
                if p_value_50 is not None:
                    if p_value_50 < 0.001:
                        p_str = "< 0.001"
                    else:
                        p_str = f"{p_value_50:.3f}"
                else:
                    p_str = "--"

                table2 += f"{name} & {diff_50:+.3f} & [{ci_lower_50:+.3f}, {ci_upper_50:+.3f}] & {p_str} \\\\\n"
            else:
                table2 += f"{name} & -- & -- & -- \\\\\n"

    table2 += r"""\hline
\end{tabular}
\caption{MAE difference between each model and Random baseline (always predicting 50\%). Positive differences indicate worse performance than baseline. P-values test whether the difference is significantly different from zero.}
\label{tab:mae_vs_random}
\end{table}
\FloatBarrier"""

    latex_tables.append(table2)

    # Table 3: MAE Differences from Human Normal baseline with p-values
    table3 = r"""\begin{table}[H]
\centering
\begin{tabular}{lccc}
\hline
\textbf{Model} & \textbf{MAE Difference} & \textbf{95\% CI} & \textbf{p-value} \\
 & \textbf{from Equanimity} & & \\
\hline
"""

    # Add model comparisons vs Normal(human)
    for model_key in ['gpt', 'claude', 'gemini']:
        if model_key in comparisons:
            m = comparisons[model_key]
            name = model_names[model_key]

            # vs Normal(human)
            if 'vs_normal_human' in m:
                diff_norm = m['vs_normal_human']['mae_diff']
                ci_lower_norm = m['vs_normal_human']['mae_diff_ci_lower']
                ci_upper_norm = m['vs_normal_human']['mae_diff_ci_upper']
                p_value_norm = m['vs_normal_human'].get('p_value', None)

                # Format p-value
                if p_value_norm is not None:
                    if p_value_norm < 0.001:
                        p_str = "< 0.001"
                    else:
                        p_str = f"{p_value_norm:.3f}"
                else:
                    p_str = "--"

                table3 += f"{name} & {diff_norm:+.3f} & [{ci_lower_norm:+.3f}, {ci_upper_norm:+.3f}] & {p_str} \\\\\n"
            else:
                table3 += f"{name} & -- & -- & -- \\\\\n"

    table3 += r"""\hline
\end{tabular}
\caption{MAE difference between each model and Equanimity baseline (normal distribution with human mean and std). Positive differences indicate worse performance than baseline. P-values test whether the difference is significantly different from zero.}
\label{tab:mae_vs_equanimity}
\end{table}
\FloatBarrier"""

    latex_tables.append(table3)

    # Table 4: Summary Statistics
    table4 = r"""\begin{table}[H]
\centering
\begin{tabular}{lcccc}
\hline
\textbf{Model} & \textbf{MAE} & \textbf{MAE SD} & \textbf{Correlation} & \textbf{p-value} \\
\hline
"""

    # Add model statistics
    for model_key in ['gpt', 'claude', 'gemini']:
        if model_key in comparisons:
            m = comparisons[model_key]
            name = model_names[model_key]
            mae = m['mae']
            mae_std = m.get('mae_std', m.get('std', 0))

            if 'correlation' in m:
                corr = m['correlation']
                p_val = m['p_value']
                if p_val < 0.001:
                    p_str = "< 0.001"
                else:
                    p_str = f"{p_val:.4f}"
            else:
                corr = None
                p_str = "--"

            # Format row
            table4 += f"{name} & {mae:.3f} & {mae_std:.3f} & "

            if corr is not None:
                table4 += f"{corr:.3f} & {p_str} \\\\\n"
            else:
                table4 += f"-- & {p_str} \\\\\n"

    table4 += r"""\hline
\end{tabular}
\caption{Summary statistics for model performance. Correlation measures Pearson correlation between model predictions and human assessments. P-value tests significance of correlation.}
\label{tab:summary_stats}
\end{table}
\FloatBarrier"""

    latex_tables.append(table4)

    return "\n\n".join(latex_tables)

def calculate_human_response_statistics(human_means, full_responses):
    """Calculate statistics about human response distribution.

    Args:
        human_means: Dictionary of question -> mean response (0-1 scale)
        full_responses: Dictionary of question -> list of all responses (0-1 scale)

    Returns:
        dict: Statistics including overall mean, std, and per-question statistics
    """
    statistics = {}

    # Overall statistics across all questions
    all_means = list(human_means.values())
    statistics['overall_mean'] = np.mean(all_means)
    statistics['overall_std'] = np.std(all_means)
    statistics['overall_min'] = np.min(all_means)
    statistics['overall_max'] = np.max(all_means)
    statistics['overall_median'] = np.median(all_means)

    # Statistics about response variability within questions
    within_question_stds = []
    response_counts = []

    for question, responses in full_responses.items():
        if len(responses) > 1:
            within_question_stds.append(np.std(responses))
            response_counts.append(len(responses))

    statistics['mean_within_question_std'] = np.mean(within_question_stds)
    statistics['std_within_question_std'] = np.std(within_question_stds)
    statistics['mean_response_count'] = np.mean(response_counts)
    statistics['total_responses'] = sum(response_counts)

    # Distribution percentiles
    statistics['percentiles'] = {
        '5th': np.percentile(all_means, 5),
        '25th': np.percentile(all_means, 25),
        '50th': np.percentile(all_means, 50),
        '75th': np.percentile(all_means, 75),
        '95th': np.percentile(all_means, 95)
    }

    return statistics

def create_mae_heatmap(model_df, human_data, output_path):
    """Create a heatmap showing MAE for each model on each question.

    Args:
        model_df: DataFrame with model predictions
        human_data: Dictionary of human survey responses
        output_path: Path to save the heatmap

    Returns:
        pd.DataFrame: DataFrame containing the MAE values
    """
    # Prepare data for heatmap
    models = ['gpt', 'gemini', 'claude']
    model_names = ['GPT-4.1', 'Gemini', 'Claude']

    mae_matrix = []
    matched_questions = []

    for idx, row in model_df.iterrows():
        question = row['question']

        # Find matching human data
        human_value = None
        for human_q, human_val in human_data.items():
            if question in human_q or human_q in question:
                human_value = human_val
                break

        if human_value is not None:
            mae_row = []
            for model in models:
                if model in ['claude']:
                    model_value = row[f'{model}_confidence']
                else:
                    model_value = row[f'{model}_weighted_confidence']
                    if pd.isna(model_value):
                        model_value = row[f'{model}_confidence']

                if not pd.isna(model_value):
                    # Convert to 0-1 scale and calculate MAE
                    model_value = model_value / 100.0
                    mae = abs(model_value - human_value)
                    mae_row.append(mae)
                else:
                    mae_row.append(np.nan)

            if not all(pd.isna(mae_row)):
                mae_matrix.append(mae_row)
                # Truncate question for display
                truncated_q = question[:50] + '...' if len(question) > 50 else question
                matched_questions.append(truncated_q)

    # Create DataFrame
    mae_df = pd.DataFrame(mae_matrix, columns=model_names, index=matched_questions)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, max(12, len(matched_questions) * 0.3)))

    # Create heatmap
    sns.heatmap(mae_df, annot=True, fmt='.3f', cmap='YlOrRd',
                cbar_kws={'label': 'Mean Absolute Error'},
                linewidths=0.5, linecolor='gray',
                vmin=0, vmax=0.5,  # MAE typically ranges 0-0.5
                ax=ax)

    ax.set_title('Model MAE per Question', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Question', fontsize=12)

    # Rotate y-axis labels for better readability
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0, fontsize=8)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved MAE heatmap to {output_path}")

    # Also save as CSV for further analysis
    csv_path = output_path.replace('.png', '.csv')
    mae_df.to_csv(csv_path)
    print(f"Saved MAE data to {csv_path}")

    return mae_df

def create_per_question_error_plot(model_df, human_data, output_path):
    """Create a scatter plot with jitter showing per-question raw error for each model.

    Args:
        model_df: DataFrame with model predictions
        human_data: Dictionary of human survey responses
        output_path: Path to save the plot

    Returns:
        pd.DataFrame: DataFrame containing the raw error values
    """
    # Prepare data
    models = ['gpt', 'gemini', 'claude']
    model_names = {'gpt': 'GPT-4.1', 'gemini': 'Gemini', 'claude': 'Claude Opus 4.1'}

    # Store errors for each model
    errors_by_model = {model: [] for model in models}
    question_indices = []

    for idx, row in model_df.iterrows():
        question = row['question']

        # Find matching human data
        human_value = None
        for human_q, human_val in human_data.items():
            if question in human_q or human_q in question:
                human_value = human_val
                break

        if human_value is not None:
            question_indices.append(idx + 1)  # 1-indexed for display

            for model in models:
                if model in ['claude']:
                    model_value = row[f'{model}_confidence']
                else:
                    model_value = row[f'{model}_weighted_confidence']
                    if pd.isna(model_value):
                        model_value = row[f'{model}_confidence']

                if not pd.isna(model_value):
                    # Convert to 0-1 scale and calculate raw error (model - human)
                    model_value = model_value / 100.0
                    error = model_value - human_value  # Raw error, not absolute
                    errors_by_model[model].append(error)
                else:
                    errors_by_model[model].append(np.nan)

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # X positions for each model
    x_positions = [0, 1, 2]

    # Add jitter to x-coordinates
    np.random.seed(42)  # For reproducible jitter

    # Plot scattered points and calculate statistics for each model
    for i, model in enumerate(models):
        # Filter out NaN values
        valid_errors = [e for e in errors_by_model[model] if not np.isnan(e)]

        # Create jittered x-coordinates using normal distribution
        x_jittered = np.random.normal(x_positions[i], 0.08, size=len(valid_errors))

        # Plot points
        ax.scatter(x_jittered, valid_errors, alpha=0.4, s=30)

        # Calculate and plot mean as a black dot
        mean_error = np.mean(valid_errors)
        ax.scatter(x_positions[i], mean_error, color='black', s=80, zorder=5)

        # Calculate and plot 95% confidence interval
        ci_lower = np.percentile(valid_errors, 2.5)
        ci_upper = np.percentile(valid_errors, 97.5)

        # Draw CI as a vertical line with caps
        ax.plot([x_positions[i], x_positions[i]], [ci_lower, ci_upper],
                'k-', linewidth=1.5, alpha=0.5)
        ax.plot([x_positions[i] - 0.05, x_positions[i] + 0.05], [ci_lower, ci_lower],
                'k-', linewidth=1.5, alpha=0.5)
        ax.plot([x_positions[i] - 0.05, x_positions[i] + 0.05], [ci_upper, ci_upper],
                'k-', linewidth=1.5, alpha=0.5)

    # Add horizontal line at zero
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels([model_names[m] for m in models])
    ax.set_ylabel('Error (Model - Human)', fontsize=12)

    # Set symmetric y-limits around zero
    max_abs_error = max([abs(e) for errors in errors_by_model.values()
                         for e in errors if not np.isnan(e)])
    ax.set_ylim(-max_abs_error * 1.1, max_abs_error * 1.1)

    # Remove grid
    ax.grid(False)

    # Show all spines for complete box
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved per-question error plot to {output_path}")

    # Create DataFrame for CSV export
    error_df = pd.DataFrame(errors_by_model)
    error_df.index = question_indices
    error_df.index.name = 'Question'

    # Save as CSV
    csv_path = output_path.replace('.png', '.csv')
    error_df.to_csv(csv_path)
    print(f"Saved error data to {csv_path}")

    return error_df


def create_visualizations(df, correlations, human_comparisons=None):
    """Create visualization plots for the analysis."""
    # Create figure with more subplots to include human comparison
    if human_comparisons:
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: GPT vs Gemini relative probabilities
    if 'gpt_relative_prob' in df.columns and 'gemini_relative_prob' in df.columns:
        axes[0, 0].scatter(df['gpt_relative_prob'], df['gemini_relative_prob'], alpha=0.6)
        axes[0, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[0, 0].set_xlabel('GPT-4.1 Relative Probability')
        axes[0, 0].set_ylabel('Gemini Relative Probability')
        axes[0, 0].set_title(f'GPT vs Gemini (ρ={correlations.get("gpt_gemini_relative_prob", {}).get("correlation", 0):.3f})')

    # Plot 2: GPT weighted confidence distribution
    if 'gpt_weighted_confidence' in df.columns:
        axes[0, 1].hist(df['gpt_weighted_confidence'].dropna(), bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Weighted Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('GPT-4.1 Weighted Confidence Distribution')
        axes[0, 1].axvline(df['gpt_weighted_confidence'].mean(), color='red', linestyle='--', label=f'Mean: {df["gpt_weighted_confidence"].mean():.1f}')
        axes[0, 1].legend()

    # Plot 3: Gemini weighted confidence distribution
    if 'gemini_weighted_confidence' in df.columns:
        axes[0, 2].hist(df['gemini_weighted_confidence'].dropna(), bins=20, edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Weighted Confidence')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Gemini Weighted Confidence Distribution')
        axes[0, 2].axvline(df['gemini_weighted_confidence'].mean(), color='red', linestyle='--', label=f'Mean: {df["gemini_weighted_confidence"].mean():.1f}')
        axes[0, 2].legend()

    # Plot 4: Claude confidence distribution
    if 'claude_confidence' in df.columns:
        axes[1, 0].hist(df['claude_confidence'].dropna(), bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Claude Opus 4.1 Confidence Distribution')
        axes[1, 0].axvline(df['claude_confidence'].mean(), color='red', linestyle='--', label=f'Mean: {df["claude_confidence"].mean():.1f}')
        axes[1, 0].legend()

    # Plot 5: Model agreement heatmap
    models = ['GPT-4.1', 'Gemini', 'Claude']
    agreement_matrix = np.zeros((3, 3))

    # Calculate agreement based on binary responses
    for i, model1 in enumerate(['gpt', 'gemini', 'claude']):
        for j, model2 in enumerate(['gpt', 'gemini', 'claude']):
            if i == j:
                agreement_matrix[i, j] = 1.0
            else:
                col1 = f'{model1}_response'
                col2 = f'{model2}_response'
                if col1 in df.columns and col2 in df.columns:
                    agreement = (df[col1] == df[col2]).mean()
                    agreement_matrix[i, j] = agreement

    im = axes[1, 1].imshow(agreement_matrix, cmap='coolwarm', vmin=0, vmax=1)
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_xticklabels(models)
    axes[1, 1].set_yticklabels(models)
    axes[1, 1].set_title('Model Agreement on Binary Responses')

    # Add values to heatmap
    for i in range(3):
        for j in range(3):
            axes[1, 1].text(j, i, f'{agreement_matrix[i, j]:.2f}',
                          ha='center', va='center', color='white' if agreement_matrix[i, j] < 0.5 else 'black')

    # Plot 6: Response distribution by model
    response_counts = pd.DataFrame({
        'GPT-4.1': df['gpt_response'].value_counts(),
        'Gemini': df['gemini_response'].value_counts(),
        'Claude': df['claude_response'].value_counts()
    }).fillna(0)

    response_counts.T.plot(kind='bar', ax=axes[1, 2])
    axes[1, 2].set_xlabel('Model')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Response Distribution by Model')
    axes[1, 2].legend(title='Response')
    axes[1, 2].tick_params(axis='x', rotation=45)

    # If human comparisons exist, add additional plots
    if human_comparisons:
        # Extract model comparisons and baselines if they're in a nested structure
        if 'models' in human_comparisons:
            model_comparisons = human_comparisons['models']
            baseline_comparisons = human_comparisons.get('baselines', {})
        else:
            model_comparisons = human_comparisons
            baseline_comparisons = {}

        # Plot 7: MAE comparison bar chart (including baselines)
        if any(model in model_comparisons for model in ['gpt', 'gemini', 'claude']) or baseline_comparisons:
            mae_data = []
            std_data = []
            labels = []
            colors = []

            # Add model results
            for model in ['gpt', 'gemini', 'claude']:
                if model in model_comparisons:
                    mae_data.append(model_comparisons[model]['mae'])
                    # Use bootstrapped CI if available, otherwise use std
                    if 'mae_ci_lower' in model_comparisons[model] and 'mae_ci_upper' in model_comparisons[model]:
                        ci_lower = model_comparisons[model]['mae'] - model_comparisons[model]['mae_ci_lower']
                        ci_upper = model_comparisons[model]['mae_ci_upper'] - model_comparisons[model]['mae']
                        std_data.append([[ci_lower], [ci_upper]])
                    else:
                        std_data.append(model_comparisons[model]['std'])
                    labels.append(model.upper() if model != 'claude' else 'Claude')
                    colors.append('#1f77b4' if model == 'gpt' else '#2ca02c' if model == 'gemini' else '#d62728')

            # Add baseline results (only the two we're keeping)
            if 'always_50' in baseline_comparisons:
                mae_data.append(baseline_comparisons['always_50']['mae'])
                if 'mae_ci_lower' in baseline_comparisons['always_50'] and 'mae_ci_upper' in baseline_comparisons['always_50']:
                    ci_lower = baseline_comparisons['always_50']['mae'] - baseline_comparisons['always_50']['mae_ci_lower']
                    ci_upper = baseline_comparisons['always_50']['mae_ci_upper'] - baseline_comparisons['always_50']['mae']
                    std_data.append([[ci_lower], [ci_upper]])
                else:
                    std_data.append(baseline_comparisons['always_50'].get('mae_std', baseline_comparisons['always_50'].get('std', 0)))
                labels.append('Always 50%')
                colors.append('#808080')

            if 'normal_human' in baseline_comparisons:
                mae_data.append(baseline_comparisons['normal_human']['mae'])
                if 'mae_ci_lower' in baseline_comparisons['normal_human'] and 'mae_ci_upper' in baseline_comparisons['normal_human']:
                    ci_lower = baseline_comparisons['normal_human']['mae'] - baseline_comparisons['normal_human']['mae_ci_lower']
                    ci_upper = baseline_comparisons['normal_human']['mae_ci_upper'] - baseline_comparisons['normal_human']['mae']
                    std_data.append([[ci_lower], [ci_upper]])
                else:
                    std_data.append(baseline_comparisons['normal_human'].get('mae_std', baseline_comparisons['normal_human'].get('std', 0)))
                # Show the human mean and std in the label
                human_mean = baseline_comparisons['normal_human'].get('human_mean', 62)
                human_std = baseline_comparisons['normal_human'].get('human_std', 17)
                labels.append(f'N({human_mean:.0f},{human_std:.0f})')
                colors.append('#17becf')

            x = np.arange(len(labels))
            # Convert std_data to proper format for error bars
            yerr_formatted = []
            for item in std_data:
                if isinstance(item, list) and len(item) == 2:
                    # It's already in [[lower], [upper]] format, extract values
                    yerr_formatted.append([item[0][0], item[1][0]])
                else:
                    # It's a scalar std, use symmetric error bars
                    yerr_formatted.append([item, item])

            # Transpose to get (2, n) shape required by matplotlib
            if yerr_formatted:
                yerr_formatted = np.array(yerr_formatted).T
            else:
                yerr_formatted = None

            bars = axes[2, 0].bar(x, mae_data, yerr=yerr_formatted, capsize=5, alpha=0.7, color=colors)
            axes[2, 0].set_xlabel('Model')
            axes[2, 0].set_ylabel('Mean Absolute Error')
            axes[2, 0].set_title('MAE vs Human Assessments (Lower is Better)')
            axes[2, 0].set_xticks(x)
            axes[2, 0].set_xticklabels(labels, rotation=45, ha='right')
            axes[2, 0].grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for i, mae in enumerate(mae_data):
                # Get the upper error bar value for positioning text
                if yerr_formatted is not None:
                    upper_err = yerr_formatted[1, i] if yerr_formatted.shape[0] == 2 else yerr_formatted[i]
                else:
                    upper_err = 0
                axes[2, 0].text(i, mae + upper_err + 0.01, f'{mae:.3f}', ha='center')

        # Plot 8: Model vs Human confidence scatter plots
        if 'gpt' in model_comparisons and model_comparisons['gpt'].get('correlation') is not None:
            # Create correlation summary text plot
            axes[2, 1].axis('off')
            summary_text = "Model-Human Correlations:\n\n"
            for model in ['gpt', 'gemini', 'claude']:
                if model in model_comparisons and 'correlation' in model_comparisons[model]:
                    model_name = model.upper() if model != 'claude' else 'Claude'
                    corr = model_comparisons[model]['correlation']
                    p_val = model_comparisons[model]['p_value']
                    n_matched = model_comparisons[model]['n_matched']
                    summary_text += f"{model_name}:\n"
                    summary_text += f"  Correlation: {corr:.3f}\n"
                    summary_text += f"  P-value: {p_val:.4f}\n"
                    summary_text += f"  N matched: {n_matched}\n\n"

            axes[2, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center', fontfamily='monospace')
            axes[2, 1].set_title('Model-Human Correlations')

        # Plot 9: Confidence comparison boxplots
        confidence_data = []
        labels = []

        for model in ['gpt', 'gemini', 'claude']:
            if model == 'claude':
                col = f'{model}_confidence'
            else:
                col = f'{model}_weighted_confidence'

            if col in df.columns:
                data = df[col].dropna()
                if len(data) > 0:
                    confidence_data.append(data)
                    labels.append(model.upper() if model != 'claude' else 'Claude')

        if confidence_data:
            bp = axes[2, 2].boxplot(confidence_data, tick_labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
                patch.set_facecolor(color)
            axes[2, 2].set_ylabel('Confidence')
            axes[2, 2].set_title('Confidence Distribution Comparison')
            axes[2, 2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison_plots.png", dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {OUTPUT_DIR}/model_comparison_plots.png")

    # Also save individual high-quality plots for the most important comparisons
    if human_comparisons:
        # Extract model comparisons and baselines if they're in a nested structure
        if 'models' in human_comparisons:
            model_comparisons = human_comparisons['models']
            baseline_comparisons = human_comparisons.get('baselines', {})
        else:
            model_comparisons = human_comparisons
            baseline_comparisons = {}

        # Save MAE comparison separately
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        if any(model in model_comparisons for model in ['gpt', 'gemini', 'claude']) or baseline_comparisons:
            mae_data = []
            std_data = []
            labels = []
            colors = []

            # Add model results
            for model in ['gpt', 'gemini', 'claude']:
                if model in model_comparisons:
                    mae_data.append(model_comparisons[model]['mae'])
                    # Use bootstrapped CI if available, otherwise use std
                    if 'mae_ci_lower' in model_comparisons[model] and 'mae_ci_upper' in model_comparisons[model]:
                        ci_lower = model_comparisons[model]['mae'] - model_comparisons[model]['mae_ci_lower']
                        ci_upper = model_comparisons[model]['mae_ci_upper'] - model_comparisons[model]['mae']
                        std_data.append([[ci_lower], [ci_upper]])
                    else:
                        std_data.append(model_comparisons[model]['std'])
                    labels.append(model.upper() if model != 'claude' else 'Claude')
                    colors.append('#1f77b4' if model == 'gpt' else '#2ca02c' if model == 'gemini' else '#d62728')

            # Add baseline results (only the two we're keeping)
            if 'always_50' in baseline_comparisons:
                mae_data.append(baseline_comparisons['always_50']['mae'])
                std_data.append(baseline_comparisons['always_50'].get('mae_std', baseline_comparisons['always_50'].get('std', 0)))
                labels.append('Always 50%\n(Baseline)')
                colors.append('#808080')

            if 'normal_human' in baseline_comparisons:
                mae_data.append(baseline_comparisons['normal_human']['mae'])
                std_data.append(baseline_comparisons['normal_human'].get('mae_std', baseline_comparisons['normal_human'].get('std', 0)))
                human_mean = baseline_comparisons['normal_human'].get('human_mean', 62)
                human_std = baseline_comparisons['normal_human'].get('human_std', 17)
                labels.append(f'N({human_mean:.0f},{human_std:.0f})\n(Baseline)')
                colors.append('#17becf')

            x = np.arange(len(labels))
            # Convert std_data to proper format for error bars
            yerr_formatted = []
            for item in std_data:
                if isinstance(item, list) and len(item) == 2:
                    # It's already in [[lower], [upper]] format, extract values
                    yerr_formatted.append([item[0][0], item[1][0]])
                else:
                    # It's a scalar std, use symmetric error bars
                    yerr_formatted.append([item, item])

            # Transpose to get (2, n) shape required by matplotlib
            if yerr_formatted:
                yerr_formatted = np.array(yerr_formatted).T
            else:
                yerr_formatted = None

            bars = ax2.bar(x, mae_data, yerr=yerr_formatted, capsize=10, alpha=0.7, color=colors)
            ax2.set_xlabel('Model', fontsize=12)
            ax2.set_ylabel('Mean Absolute Error', fontsize=12)
            ax2.set_title('Mean Absolute Error vs Human Assessments (Lower is Better)', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.grid(axis='y', alpha=0.3)

            # Add horizontal line at baseline levels for reference
            if 'always_50' in baseline_comparisons:
                ax2.axhline(y=baseline_comparisons['always_50']['mae'], color='gray', linestyle='--', alpha=0.3)

            # Add value labels on bars
            for i, (bar, mae) in enumerate(zip(bars, mae_data)):
                height = bar.get_height()
                # Get the upper error bar value for positioning text
                if yerr_formatted is not None:
                    upper_err = yerr_formatted[1, i] if yerr_formatted.shape[0] == 2 else yerr_formatted[i]
                else:
                    upper_err = 0
                ax2.text(bar.get_x() + bar.get_width()/2., height + upper_err + 0.005,
                        f'{mae:.3f}', ha='center', va='bottom', fontsize=11)

            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/mae_comparison.png", dpi=300, bbox_inches='tight')
            print(f"Saved MAE comparison to {OUTPUT_DIR}/mae_comparison.png")

def main():
    """Main execution function."""
    print("="*60)
    print("CLOSED-SOURCE LLM EVALUATION ON ORDINARY MEANING QUESTIONS")
    print("(Extended to 100 Questions - Parts 1 & 2)")
    print("="*60)

    # Check API keys
    if not all([OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY]):
        print("ERROR: Missing API keys. Please set all required API keys in config.py")
        return

    # Load questions
    print("\nLoading ordinary meaning questions...")
    questions = load_ordinary_meaning_questions()
    print(f"Loaded {len(questions)} questions")

    # Check if we should load from saved results instead of cache/API
    saved_results_file = f"{OUTPUT_DIR}/closed_source_evaluation_results.csv"
    if os.path.exists(saved_results_file):
        print(f"\nLoading existing results from {saved_results_file}...")
        results_df = pd.read_csv(saved_results_file)
        print(f"Loaded {len(results_df)} results from file")
    else:
        # Check cache status
        if USE_CACHE and os.path.exists(CACHE_FILE):
            print(f"\nCache mode: ENABLED (using cached results from {CACHE_FILE})")
            print("To force fresh API calls, set USE_CACHE = False in the script")
        else:
            print(f"\nCache mode: DISABLED (will make fresh API calls)")
            # Estimate time and cost
            total_api_calls = len(questions) * 6  # 2 calls per model per question
            estimated_time = (len(questions) * (GPT_DELAY * 2 + GEMINI_DELAY * 2 + CLAUDE_DELAY * 2)) / 60
            print(f"\nEstimated processing time: {estimated_time:.1f} minutes")
            print(f"Total API calls: {total_api_calls}")

            # Confirm before proceeding
            response = input("\nProceed with evaluation? (yes/no): ")
            if response.lower() != 'yes':
                print("Evaluation cancelled.")
                return

        # Evaluate all models
        print("\nStarting evaluation...")
        results_df = evaluate_all_models(questions)

    # Save results
    results_df.to_csv(f"{OUTPUT_DIR}/closed_source_evaluation_results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/closed_source_evaluation_results.csv")

    # Calculate correlations
    print("\nCalculating correlations between models...")
    correlations = calculate_correlations(results_df)

    # Print correlation results
    print("\nModel Correlations:")
    print("-" * 40)
    for key, values in correlations.items():
        print(f"{key}:")
        print(f"  Correlation: {values['correlation']:.3f}")
        print(f"  P-value: {values['p_value']:.4f}")

    # Save correlation results
    with open(f"{OUTPUT_DIR}/correlations.json", 'w') as f:
        json.dump(correlations, f, indent=2)

    # Load and compare with human data
    print("\nLoading human survey data...")
    human_data, full_responses = load_human_survey_data(return_full_data=True)
    comparisons = {}
    baselines = {}
    human_stats = {}

    if human_data:
        print(f"Loaded human data for {len(human_data)} questions")

        # Calculate human response statistics
        human_stats = calculate_human_response_statistics(human_data, full_responses)

        # Print human response statistics
        print("\nHuman Response Statistics:")
        print("-" * 40)
        print(f"Overall mean response: {human_stats['overall_mean']:.3f} ({human_stats['overall_mean']*100:.1f}%)")
        print(f"Overall standard deviation: {human_stats['overall_std']:.3f}")
        print(f"Response range: [{human_stats['overall_min']:.3f}, {human_stats['overall_max']:.3f}]")
        print(f"Median response: {human_stats['overall_median']:.3f}")
        print(f"\nPercentiles:")
        for perc_name, perc_val in human_stats['percentiles'].items():
            print(f"  {perc_name}: {perc_val:.3f}")
        print(f"\nWithin-question variability:")
        print(f"  Mean std within questions: {human_stats['mean_within_question_std']:.3f}")
        print(f"  Std of within-question std: {human_stats['std_within_question_std']:.3f}")
        print(f"  Mean responses per question: {human_stats['mean_response_count']:.1f}")
        print(f"  Total responses: {human_stats['total_responses']:.0f}")

        # Calculate baselines first
        baselines = calculate_baseline_performance(results_df, human_data)

        # Compare models with human data and baselines
        comparisons = compare_with_human_data(results_df, human_data, baselines)

        # Print human comparison results
        if comparisons:
            print("\nModel-Human Comparisons:")
            print("-" * 40)
            for model, metrics in comparisons.items():
                model_display = model.upper() if model != 'claude' else 'Claude'
                print(f"{model_display}:")
                print(f"  MAE: {metrics['mae']:.3f} ± {metrics.get('mae_std', metrics.get('std', 0)):.3f}")
                if 'mae_ci_lower' in metrics and 'mae_ci_upper' in metrics:
                    print(f"  MAE 95% CI: [{metrics['mae_ci_lower']:.3f}, {metrics['mae_ci_upper']:.3f}]")

                # Print MAE comparisons to baselines
                if 'vs_always_50' in metrics:
                    diff_50 = metrics['vs_always_50']['mae_diff']
                    ci_lower_50 = metrics['vs_always_50']['mae_diff_ci_lower']
                    ci_upper_50 = metrics['vs_always_50']['mae_diff_ci_upper']
                    p_value_50 = metrics['vs_always_50'].get('p_value', None)
                    p_str_50 = f", p={p_value_50:.3f}" if p_value_50 is not None else ""
                    print(f"  MAE vs. Always 50%: {diff_50:+.3f} [{ci_lower_50:+.3f}, {ci_upper_50:+.3f}]{p_str_50}")

                if 'vs_normal_human' in metrics:
                    diff_norm = metrics['vs_normal_human']['mae_diff']
                    ci_lower_norm = metrics['vs_normal_human']['mae_diff_ci_lower']
                    ci_upper_norm = metrics['vs_normal_human']['mae_diff_ci_upper']
                    p_value_norm = metrics['vs_normal_human'].get('p_value', None)
                    p_str_norm = f", p={p_value_norm:.3f}" if p_value_norm is not None else ""
                    print(f"  MAE vs. N(human): {diff_norm:+.3f} [{ci_lower_norm:+.3f}, {ci_upper_norm:+.3f}]{p_str_norm}")

                print(f"  Matched questions: {metrics['n_matched']}")
                if 'correlation' in metrics:
                    print(f"  Correlation: {metrics['correlation']:.3f} (p={metrics['p_value']:.4f})")

        # Print baseline results
        if baselines:
            print("\nBaseline Performance:")
            print("-" * 40)
            for baseline_name, metrics in baselines.items():
                print(f"{metrics['description']}:")
                print(f"  MAE: {metrics['mae']:.3f} ± {metrics.get('mae_std', metrics.get('std', 0)):.3f}")
                if 'mae_ci_lower' in metrics and 'mae_ci_upper' in metrics:
                    print(f"  MAE 95% CI: [{metrics['mae_ci_lower']:.3f}, {metrics['mae_ci_upper']:.3f}]")

                if 'human_mean' in metrics and 'human_std' in metrics:
                    print(f"  Parameters: mean={metrics['human_mean']:.1f}%, std={metrics['human_std']:.1f}%")

        # Generate and save LaTeX tables
        if comparisons and baselines:
            latex_tables = generate_latex_tables(comparisons, baselines)
            latex_file = f"{OUTPUT_DIR}/mae_results_tables.tex"
            with open(latex_file, 'w') as f:
                f.write(latex_tables)
            print(f"\nLaTeX tables saved to {latex_file}")

            # Also print the LaTeX tables to console for easy copying
            print("\n" + "="*60)
            print("LATEX TABLES FOR ARTICLE")
            print("="*60)
            print(latex_tables)

        # Create MAE heatmap
        print("\nCreating MAE heatmap...")
        mae_heatmap_path = f"{OUTPUT_DIR}/mae_heatmap.png"
        mae_df = create_mae_heatmap(results_df, human_data, mae_heatmap_path)

        # Create per-question error plot
        print("\nCreating per-question error plot...")
        error_plot_path = f"{OUTPUT_DIR}/per_question_errors.png"
        error_df = create_per_question_error_plot(results_df, human_data, error_plot_path)

        # Save comparison data
        comparison_data = {
            'models': comparisons,
            'baselines': baselines,
            'human_statistics': human_stats
        }

        with open(f"{OUTPUT_DIR}/human_comparisons.json", 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)

    # Create visualizations
    print("\nCreating visualizations...")
    # Combine model comparisons and baselines for visualization
    human_comparisons = None
    if comparisons or baselines:
        human_comparisons = {'models': comparisons, 'baselines': baselines}
    create_visualizations(results_df, correlations, human_comparisons)

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    # Agreement rates
    print("\nBinary Response Agreement Rates:")
    print(f"  GPT-Gemini: {(results_df['gpt_response'] == results_df['gemini_response']).mean():.1%}")
    print(f"  GPT-Claude: {(results_df['gpt_response'] == results_df['claude_response']).mean():.1%}")
    print(f"  Gemini-Claude: {(results_df['gemini_response'] == results_df['claude_response']).mean():.1%}")

    # Average confidence levels
    print("\nAverage Confidence Levels:")
    print(f"  GPT-4.1 (weighted): {results_df['gpt_weighted_confidence'].mean():.1f}")
    print(f"  Gemini (weighted): {results_df['gemini_weighted_confidence'].mean():.1f}")
    print(f"  Claude (verbalized): {results_df['claude_confidence'].mean():.1f}")

    print("\nEvaluation complete!")

if __name__ == "__main__":
    main()