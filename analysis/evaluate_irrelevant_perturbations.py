"""
Evaluate Legal Scenario Perturbations with Irrelevant Statements
This script evaluates how GPT, Claude, and Gemini models respond to legal scenarios
that have been perturbed with irrelevant factual statements.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
import argparse
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Import model-specific modules
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

# Import configuration
from config import OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY

# Configuration
# Use paths relative to the project root (parent of analysis directory)

# Get the project root directory (parent of analysis directory)
PROJECT_ROOT = Path(__file__).parent.parent
PERTURBATIONS_FILE = str(PROJECT_ROOT / "data" / "perturbations_irrelevant.json")
RESULTS_DIR = PROJECT_ROOT / "results" / "irrelevant_perturbations"
COMBINED_RESULTS_FILE = str(RESULTS_DIR / "combined_model_results.xlsx")
CHECKPOINT_FILE = str(RESULTS_DIR / "checkpoint.pkl")
PROGRESS_FILE = str(RESULTS_DIR / "progress.json")

# Model configurations - Using the same models as in other perturbation analyses
MODELS = {
    "gpt": {
        "name": "gpt-4.1-2025-04-14",  # GPT-4.1
        "temperature": 0.7,
        "max_tokens": 500
    },
    "claude": {
        "name": "claude-opus-4-1-20250805",  # Claude Opus 4.1
        "temperature": 0.7,
        "max_tokens": 500
    },
    "gemini": {
        "name": "gemini-2.5-pro",  # Gemini 2.5 Pro
        "temperature": 0.7,
        "max_tokens": 500
    }
}

# Processing configuration
BATCH_SIZE = 10
DELAY_BETWEEN_REQUESTS = 0.1  # Reduced by 90% from 1.0 to 0.1 seconds
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5
MAX_RETRY_DELAY = 60

# Initialize API clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Configure Gemini with permissive safety settings to prevent blocking
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]
gemini_model = genai.GenerativeModel(MODELS["gemini"]["name"], safety_settings=safety_settings)


def ensure_directory(path: str):
    """Ensure directory exists, handling Windows paths with spaces."""
    # Convert to Path object for better path handling
    path_obj = Path(path)
    # Create the directory if it doesn't exist
    path_obj.mkdir(parents=True, exist_ok=True)


def save_checkpoint(checkpoint_data: Dict, checkpoint_file: str = CHECKPOINT_FILE):
    """Save checkpoint data to file for resuming interrupted runs.

    Args:
        checkpoint_data: Dictionary containing:
            - completed: Set of (model, scenario, perturbation_id) tuples
            - all_results: List of all results collected so far
            - last_model: Last model being processed
            - last_scenario: Last scenario being processed
            - timestamp: When checkpoint was created
    """
    ensure_directory(os.path.dirname(checkpoint_file))

    # Convert sets to lists for JSON serialization
    checkpoint_data_serializable = checkpoint_data.copy()
    checkpoint_data_serializable['completed'] = list(checkpoint_data['completed'])
    checkpoint_data_serializable['timestamp'] = datetime.now().isoformat()

    # Save as pickle (handles complex data structures better)
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    # Also save a human-readable progress file
    progress_data = {
        'timestamp': checkpoint_data_serializable['timestamp'],
        'total_completed': len(checkpoint_data['completed']),
        'last_model': checkpoint_data.get('last_model', 'N/A'),
        'last_scenario': checkpoint_data.get('last_scenario', 'N/A'),
        'models_progress': {}
    }

    # Calculate progress per model
    for model in ['gpt', 'claude', 'gemini']:
        model_items = [item for item in checkpoint_data['completed'] if item[0] == model]
        progress_data['models_progress'][model] = len(model_items)

    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress_data, f, indent=2)

    print(f"  [Checkpoint saved: {len(checkpoint_data['completed'])} evaluations completed]")


def load_checkpoint(checkpoint_file: str = CHECKPOINT_FILE) -> Optional[Dict]:
    """Load checkpoint data if it exists.

    Returns:
        Dictionary with checkpoint data or None if no checkpoint exists
    """
    if not os.path.exists(checkpoint_file):
        return None

    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)

        # Ensure 'completed' is a set
        if isinstance(checkpoint_data.get('completed'), list):
            checkpoint_data['completed'] = set(checkpoint_data['completed'])

        print(f"Loaded checkpoint with {len(checkpoint_data['completed'])} completed evaluations")
        print(f"Last checkpoint: {checkpoint_data.get('timestamp', 'Unknown')}")

        # Show progress summary
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
                print("Progress by model:")
                for model, count in progress.get('models_progress', {}).items():
                    print(f"  - {model}: {count} evaluations")

        return checkpoint_data
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None


def should_skip_evaluation(model: str, scenario: str, perturbation_id: str,
                          completed_set: Set[Tuple[str, str, str]]) -> bool:
    """Check if an evaluation should be skipped based on checkpoint.

    Args:
        model: Model name
        scenario: Scenario name
        perturbation_id: Perturbation ID (or 'original')
        completed_set: Set of completed (model, scenario, perturbation_id) tuples

    Returns:
        True if this evaluation was already completed and should be skipped
    """
    return (model, scenario, perturbation_id) in completed_set


def load_perturbations(file_path: str) -> List[Dict]:
    """Load perturbations from JSON file."""
    print(f"Loading perturbations from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} scenarios with perturbations")
    return data


def extract_final_number(response_text: str) -> Optional[float]:
    """Extract the final number from a response, handling thinking model outputs.

    Thinking models (Claude and Gemini) often provide detailed reasoning followed
    by a final answer. This function handles various formats including:
    - Numbers after *** markers
    - Numbers after ### markers
    - Standalone numbers on their own line
    - Last number in the text
    """
    import re

    if not response_text:
        return None

    # First, check for numbers between *** or ### markers (common in thinking models)
    # Look for patterns like "***\n20\n***" or "###\n20\n###"
    marker_pattern = r'(?:\*{3,}|#{3,})\s*(\d+(?:\.\d+)?)\s*(?:\*{3,}|#{3,})'
    marker_match = re.search(marker_pattern, response_text, re.MULTILINE | re.DOTALL)
    if marker_match:
        try:
            return float(marker_match.group(1))
        except (ValueError, IndexError):
            pass

    # Check for a number that appears after the last *** or ### in the text
    # This handles cases where the number is after markers but not between them
    lines = response_text.split('\n')
    after_marker = False
    for line in reversed(lines):
        line = line.strip()
        if '***' in line or '###' in line:
            after_marker = True
        elif after_marker and line:
            # Try to extract a number from this line
            number_match = re.match(r'^(\d+(?:\.\d+)?)$', line)
            if number_match:
                try:
                    return float(number_match.group(1))
                except (ValueError, IndexError):
                    pass

    # Look for a line that contains only a number (possibly with whitespace)
    # This is common when models output a number on its own line at the end
    for line in reversed(lines):
        line = line.strip()
        if re.match(r'^(\d+(?:\.\d+)?)$', line):
            try:
                return float(line)
            except:
                pass

    # Look for standalone numbers (integers or floats) in the text
    # This pattern matches numbers that are not part of larger words
    number_pattern = r'\b(\d+(?:\.\d+)?)\b'

    # Find all numbers in the response
    numbers = re.findall(number_pattern, response_text)

    if numbers:
        # Return the last number found, which should be the final answer
        try:
            return float(numbers[-1])
        except (ValueError, IndexError):
            pass

    # Final fallback: try to extract digits only (original method)
    # This is least preferred as it can concatenate separate numbers
    try:
        digits_only = ''.join(filter(str.isdigit, response_text))
        if digits_only and len(digits_only) <= 3:  # Only use this for short numbers
            return float(digits_only)
    except:
        pass

    return None


def retry_with_exponential_backoff(func, max_retries=MAX_RETRIES,
                                  initial_delay=INITIAL_RETRY_DELAY,
                                  max_delay=MAX_RETRY_DELAY):
    """Retry a function with exponential backoff on API errors."""
    import random
    num_retries = 0
    delay = initial_delay

    while True:
        try:
            return func()
        except Exception as e:
            num_retries += 1
            if num_retries > max_retries:
                raise Exception(f"Maximum retries ({max_retries}) exceeded: {str(e)}")

            # Add jitter
            jitter = random.uniform(0.8, 1.2)
            sleep_time = min(delay * jitter, max_delay)

            print(f"  API error: {str(e)[:100]}... Retrying in {sleep_time:.1f}s (retry {num_retries}/{max_retries})")
            time.sleep(sleep_time)

            # Exponential backoff
            delay = min(delay * 1.5, max_delay)


def evaluate_with_gpt(prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> Dict:
    """Evaluate prompt with GPT-4."""
    def api_call():
        response = openai_client.chat.completions.create(
            model=MODELS["gpt"]["name"],
            messages=[
                {"role": "system", "content": "You are a legal expert analyzing contract and insurance policy language."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return {
            "response": response.choices[0].message.content,
            "model": MODELS["gpt"]["name"],
            "finish_reason": response.choices[0].finish_reason
        }

    return retry_with_exponential_backoff(api_call)


def evaluate_with_claude(prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> Dict:
    """Evaluate prompt with Claude."""
    def api_call():
        response = anthropic_client.messages.create(
            model=MODELS["claude"]["name"],
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return {
            "response": response.content[0].text,
            "model": MODELS["claude"]["name"],
            "finish_reason": response.stop_reason
        }

    return retry_with_exponential_backoff(api_call)


def evaluate_with_gemini(prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> Dict:
    """Evaluate prompt with Gemini.

    Note: max_tokens parameter is ignored for Gemini due to a bug where setting
    max_output_tokens causes Gemini 2.5 Pro to return empty responses with MAX_TOKENS finish reason.
    """
    def api_call():
        # Don't set max_output_tokens for Gemini - it causes issues
        generation_config = genai.GenerationConfig(
            temperature=temperature
        )
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Get response text - simplified since we shouldn't hit MAX_TOKENS anymore
        response_text = ""
        if hasattr(response, 'text'):
            response_text = response.text
        elif response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            response_text = response.candidates[0].content.parts[0].text

        finish_reason_name = "UNKNOWN"
        if response.candidates and hasattr(response.candidates[0], 'finish_reason'):
            finish_reason_name = response.candidates[0].finish_reason.name

        return {
            "response": response_text,
            "model": MODELS["gemini"]["name"],
            "finish_reason": finish_reason_name
        }

    return retry_with_exponential_backoff(api_call)


def process_scenario_perturbations(scenario_data: Dict, model_name: str,
                                  completed_set: Set[Tuple[str, str, str]] = None,
                                  checkpoint_callback=None) -> List[Dict]:
    """Process all perturbations for a single scenario with a specific model.

    Args:
        scenario_data: Scenario data dictionary
        model_name: Name of the model to use
        completed_set: Set of already completed (model, scenario, perturbation_id) tuples
        checkpoint_callback: Optional callback function to save checkpoints

    Returns:
        List of results dictionaries
    """
    if completed_set is None:
        completed_set = set()

    results = []
    scenario_name = scenario_data["scenario_name"]
    total_perturbations = len(scenario_data.get('perturbations_with_irrelevant', []))

    print(f"\nProcessing {scenario_name} with {model_name}...")

    # Count how many evaluations we'll actually do (excluding skipped)
    evaluations_to_skip = 0
    if should_skip_evaluation(model_name, scenario_name, "original", completed_set):
        evaluations_to_skip += 1
    for pert in scenario_data.get('perturbations_with_irrelevant', []):
        if should_skip_evaluation(model_name, scenario_name, pert['perturbation_id'], completed_set):
            evaluations_to_skip += 1

    actual_evaluations = (1 + total_perturbations) - evaluations_to_skip
    print(f"  Total evaluations for this scenario: {actual_evaluations} new + {evaluations_to_skip} skipped = {1 + total_perturbations} total")

    # Evaluate original scenario first
    original_prompt = f"{scenario_data['original_main']}\n\n{scenario_data['response_format']}"

    if model_name == "gpt":
        eval_func = evaluate_with_gpt
    elif model_name == "claude":
        eval_func = evaluate_with_claude
    else:  # gemini
        eval_func = evaluate_with_gemini

    # Check if we should skip the original evaluation
    if should_skip_evaluation(model_name, scenario_name, "original", completed_set):
        print(f"  Skipping original scenario (already completed)")
    else:
        # Get original response
        print("  Evaluating original scenario...")
        original_result = eval_func(original_prompt)
        time.sleep(DELAY_BETWEEN_REQUESTS)

        # Get original confidence
        confidence_prompt = f"{scenario_data['original_main']}\n\n{scenario_data['confidence_format']}"
        confidence_result = eval_func(confidence_prompt)
        time.sleep(DELAY_BETWEEN_REQUESTS)

        # Parse confidence using the new extraction function
        confidence = extract_final_number(confidence_result['response'])

        result_dict = {
            "scenario": scenario_name,
            "model": model_name,
            "perturbation_id": "original",
            "irrelevant_statement": None,
            "position_index": None,
            "position_description": None,
            "response": original_result['response'],
            "confidence": confidence,
            "is_original": True,
            "response_prompt": original_prompt,
            "confidence_prompt": confidence_prompt,
            "confidence_raw_response": confidence_result['response']
        }
        results.append(result_dict)

        # Mark as completed and save checkpoint if callback provided
        completed_set.add((model_name, scenario_name, "original"))
        if checkpoint_callback:
            checkpoint_callback()

    # Process each perturbation
    perturbations = scenario_data.get('perturbations_with_irrelevant', [])
    for idx, pert in enumerate(perturbations, 1):
        pert_id = pert['perturbation_id']

        # Check if we should skip this perturbation
        if should_skip_evaluation(model_name, scenario_name, pert_id, completed_set):
            print(f"  Skipping perturbation {idx}/{len(perturbations)} (ID: {pert_id}) - already completed")
            continue

        print(f"  Processing perturbation {idx}/{len(perturbations)} (ID: {pert_id})...")

        # Get response for perturbed text
        perturbed_prompt = f"{pert['perturbed_text']}\n\n{scenario_data['response_format']}"
        perturbed_result = eval_func(perturbed_prompt)
        time.sleep(DELAY_BETWEEN_REQUESTS)

        # Get confidence for perturbed text
        confidence_prompt = f"{pert['perturbed_text']}\n\n{scenario_data['confidence_format']}"
        confidence_result = eval_func(confidence_prompt)
        time.sleep(DELAY_BETWEEN_REQUESTS)

        # Parse confidence using the updated extraction function
        confidence = extract_final_number(confidence_result['response'])

        result_dict = {
            "scenario": scenario_name,
            "model": model_name,
            "perturbation_id": pert_id,
            "irrelevant_statement": pert['irrelevant_statement'],
            "position_index": pert['position_index'],
            "position_description": pert['position_description'],
            "response": perturbed_result['response'],
            "confidence": confidence,
            "is_original": False,
            "response_prompt": perturbed_prompt,
            "confidence_prompt": confidence_prompt,
            "confidence_raw_response": confidence_result['response']
        }
        results.append(result_dict)

        # Mark as completed and save checkpoint if callback provided
        completed_set.add((model_name, scenario_name, pert_id))
        if checkpoint_callback:
            checkpoint_callback()

    return results


def analyze_results(results_df: pd.DataFrame) -> Dict:
    """Analyze results to compute consistency metrics with enhanced statistics."""
    analysis = {}

    for scenario in results_df['scenario'].unique():
        scenario_data = results_df[results_df['scenario'] == scenario]
        analysis[scenario] = {}

        for model in results_df['model'].unique():
            model_data = scenario_data[scenario_data['model'] == model]

            # Check if we have any data for this model/scenario combination
            if len(model_data) == 0:
                print(f"  Warning: No data found for model {model} in scenario {scenario}")
                continue

            # Get original response
            original_data = model_data[model_data['is_original'] == True]

            # Handle missing original by using the most common response as reference
            if len(original_data) == 0:
                print(f"  Warning: No original response found for {model} in {scenario}")

                # If we have perturbed data, use the most common response as reference
                perturbed = model_data[model_data['is_original'] == False]
                if len(perturbed) > 0:
                    print(f"    Using most common response as reference for analysis")
                    # Find most common response
                    most_common_response = perturbed['response'].mode()[0] if len(perturbed['response'].mode()) > 0 else perturbed['response'].iloc[0]
                    # Calculate mean confidence for reference
                    reference_confidence = perturbed['confidence'].mean()

                    # Create a synthetic original for analysis
                    original = pd.Series({
                        'response': most_common_response,
                        'confidence': reference_confidence,
                        'response_prompt': 'N/A - Original missing',
                        'confidence_prompt': 'N/A - Original missing',
                        'confidence_raw_response': 'N/A - Original missing'
                    })
                else:
                    print(f"    Skipping - no data available for analysis")
                    continue
            else:
                original = original_data.iloc[0]
                perturbed = model_data[model_data['is_original'] == False]

            # Get all confidence values (including original if available)
            all_confidences = model_data['confidence'].dropna()

            # Skip if no confidence data
            if len(all_confidences) == 0:
                print(f"  Warning: No confidence data found for {model} in {scenario}")
                continue

            # Calculate consistency (how often the model gives the same answer)
            if 'perturbed' not in locals() or len(perturbed) == 0:
                perturbed = model_data[model_data['is_original'] == False]

            if len(perturbed) > 0:
                consistency = (perturbed['response'] == original['response']).mean()
            else:
                consistency = 1.0  # Only original, so 100% consistent

            # Calculate enhanced confidence statistics
            confidence_stats = {
                "original_confidence": original['confidence'] if pd.notna(original['confidence']) else None,
                "mean_all_confidence": all_confidences.mean(),
                "std_all_confidence": all_confidences.std(),
                "median_all_confidence": all_confidences.median(),
                "ci_lower_95": np.percentile(all_confidences, 2.5),
                "ci_upper_95": np.percentile(all_confidences, 97.5),
                "min_confidence": all_confidences.min(),
                "max_confidence": all_confidences.max(),
                "n_samples": len(all_confidences)
            }

            # Add perturbed-specific stats if we have perturbed data
            if len(perturbed) > 0:
                perturbed_confidences = perturbed['confidence'].dropna()
                if len(perturbed_confidences) > 0:
                    confidence_stats.update({
                        "mean_perturbed_confidence": perturbed_confidences.mean(),
                        "std_perturbed_confidence": perturbed_confidences.std(),
                        "median_perturbed_confidence": perturbed_confidences.median(),
                        "perturbed_ci_lower_95": np.percentile(perturbed_confidences, 2.5),
                        "perturbed_ci_upper_95": np.percentile(perturbed_confidences, 97.5)
                    })

            # Position-based analysis (by position index)
            position_consistency = {}
            if len(perturbed) > 0:
                for position_idx in perturbed['position_index'].unique():
                    if pd.notna(position_idx):  # Skip None values
                        pos_data = perturbed[perturbed['position_index'] == position_idx]
                        position_desc = pos_data['position_description'].iloc[0] if len(pos_data) > 0 else str(position_idx)
                        position_consistency[f"{int(position_idx)}_{position_desc}"] = (pos_data['response'] == original['response']).mean()

            # Store all confidence values for violin plot
            confidence_values = all_confidences.tolist()

            analysis[scenario][model] = {
                "consistency": consistency,
                "confidence_stats": confidence_stats,
                "position_consistency": position_consistency,
                "num_perturbations": len(perturbed) if 'perturbed' in locals() else 0,
                "num_total_samples": len(model_data),
                "original_response": original['response'],
                "original_response_prompt": original.get('response_prompt', 'N/A'),
                "original_confidence_prompt": original.get('confidence_prompt', 'N/A'),
                "original_confidence_raw_response": original.get('confidence_raw_response', 'N/A'),
                "confidence_values": confidence_values  # For violin plots
            }

    return analysis


def save_results(all_results: List[Dict], analysis: Dict, output_dir: str):
    """Save results to files."""
    ensure_directory(output_dir)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save raw results as CSV (always works)
    raw_file = f"{output_dir}/raw_results.csv"
    results_df.to_csv(raw_file, index=False)
    print(f"Saved raw results to {raw_file}")

    # Prepare summary data with enhanced statistics
    summary_data = []
    for scenario, scenario_analysis in analysis.items():
        for model, model_analysis in scenario_analysis.items():
            conf_stats = model_analysis["confidence_stats"]
            summary_data.append({
                "scenario": scenario,
                "model": model,
                "consistency": model_analysis["consistency"],
                "original_confidence": conf_stats.get("original_confidence"),
                "mean_all_confidence": conf_stats.get("mean_all_confidence"),
                "std_all_confidence": conf_stats.get("std_all_confidence"),
                "median_all_confidence": conf_stats.get("median_all_confidence"),
                "ci_lower_95": conf_stats.get("ci_lower_95"),
                "ci_upper_95": conf_stats.get("ci_upper_95"),
                "n_samples": conf_stats.get("n_samples"),
                "mean_perturbed_confidence": conf_stats.get("mean_perturbed_confidence"),
                "std_perturbed_confidence": conf_stats.get("std_perturbed_confidence"),
                "original_response": model_analysis["original_response"],
                "num_perturbations": model_analysis.get("num_perturbations", 0),
                "num_total_samples": model_analysis.get("num_total_samples", 0)
            })

    summary_df = pd.DataFrame(summary_data)

    # Save summary as CSV
    summary_csv = f"{output_dir}/summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}")

    # Prepare position analysis data
    position_data = []
    for scenario, scenario_analysis in analysis.items():
        for model, model_analysis in scenario_analysis.items():
            for position, consistency in model_analysis["position_consistency"].items():
                position_data.append({
                    "scenario": scenario,
                    "model": model,
                    "position": position,
                    "consistency": consistency
                })

    # Try to save Excel file with multiple sheets
    excel_file = f"{output_dir}/results_analysis.xlsx"
    excel_saved = False

    # Try xlsxwriter first
    try:
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            # Raw results
            results_df.to_excel(writer, sheet_name='Raw Results', index=False)
            # Summary
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            # Position analysis
            if position_data:
                position_df = pd.DataFrame(position_data)
                position_pivot = position_df.pivot_table(
                    index=['scenario', 'model'],
                    columns='position',
                    values='consistency'
                )
                position_pivot.to_excel(writer, sheet_name='Position Analysis')
        excel_saved = True
        print(f"Saved analysis to {excel_file}")
    except ImportError:
        # Try openpyxl as fallback
        try:
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # Raw results
                results_df.to_excel(writer, sheet_name='Raw Results', index=False)
                # Summary
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                # Position analysis
                if position_data:
                    position_df = pd.DataFrame(position_data)
                    position_pivot = position_df.pivot_table(
                        index=['scenario', 'model'],
                        columns='position',
                        values='consistency'
                    )
                    position_pivot.to_excel(writer, sheet_name='Position Analysis')
            excel_saved = True
            print(f"Saved analysis to {excel_file} (using openpyxl)")
        except ImportError:
            print(f"Warning: Could not save Excel file (neither xlsxwriter nor openpyxl installed)")
            print(f"Results saved as CSV files instead")

    # Save analysis as JSON
    json_file = f"{output_dir}/analysis.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to {json_file}")

    # Create summary report
    report_file = f"{output_dir}/summary_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("IRRELEVANT STATEMENT PERTURBATION ANALYSIS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for scenario, scenario_analysis in analysis.items():
            f.write(f"\n{scenario}\n")
            f.write("-" * 40 + "\n")

            for model, model_analysis in scenario_analysis.items():
                conf_stats = model_analysis['confidence_stats']
                f.write(f"\n{model}:\n")
                f.write(f"  Consistency: {model_analysis['consistency']:.2%}\n")
                f.write(f"  Original Response: {model_analysis['original_response']}\n")
                f.write(f"  Number of Samples: {conf_stats.get('n_samples', 'N/A')}\n")
                f.write(f"\n  Confidence Statistics:\n")
                f.write(f"    Original: {conf_stats.get('original_confidence', 'N/A')}\n")
                f.write(f"    Mean (all): {conf_stats.get('mean_all_confidence', 0):.1f}\n")
                f.write(f"    Std Dev (all): {conf_stats.get('std_all_confidence', 0):.1f}\n")
                f.write(f"    Median (all): {conf_stats.get('median_all_confidence', 0):.1f}\n")
                f.write(f"    95% CI: [{conf_stats.get('ci_lower_95', 0):.1f}, {conf_stats.get('ci_upper_95', 0):.1f}]\n")

                if 'mean_perturbed_confidence' in conf_stats:
                    f.write(f"    Mean (perturbed only): {conf_stats.get('mean_perturbed_confidence', 0):.1f}\n")
                    f.write(f"    Std Dev (perturbed only): {conf_stats.get('std_perturbed_confidence', 0):.1f}\n")

                f.write("\n  Position Consistency:\n")
                for position, consistency in model_analysis['position_consistency'].items():
                    f.write(f"    {position}: {consistency:.2%}\n")

                f.write("\n  Prompts Used:\n")
                f.write(f"    Response Prompt:\n")
                f.write(f"    {model_analysis['original_response_prompt'][:200]}...\n" if len(model_analysis['original_response_prompt']) > 200 else f"    {model_analysis['original_response_prompt']}\n")
                f.write(f"    \n    Confidence Prompt:\n")
                f.write(f"    {model_analysis['original_confidence_prompt'][:200]}...\n" if len(model_analysis['original_confidence_prompt']) > 200 else f"    {model_analysis['original_confidence_prompt']}\n")

    print(f"Saved summary report to {report_file}")

    # Create detailed prompts file
    prompts_file = f"{output_dir}/detailed_prompts.txt"
    with open(prompts_file, 'w', encoding='utf-8') as f:
        f.write("DETAILED PROMPTS USED IN EVALUATION\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Group by unique prompts to show variety
        seen_prompts = set()

        for idx, row in results_df.iterrows():
            prompt_key = (row['scenario'], row['perturbation_id'])
            if prompt_key not in seen_prompts:
                seen_prompts.add(prompt_key)
                f.write(f"\nScenario: {row['scenario']}\n")
                f.write(f"Perturbation ID: {row['perturbation_id']}\n")
                if row['irrelevant_statement']:
                    f.write(f"Irrelevant Statement: {row['irrelevant_statement']}\n")
                f.write(f"Model: {row['model']}\n")
                f.write("-" * 40 + "\n")

                f.write("\nRESPONSE PROMPT:\n")
                f.write(row['response_prompt'])
                f.write("\n\nCONFIDENCE PROMPT:\n")
                f.write(row['confidence_prompt'])
                f.write("\n\nModel Response: " + row['response'])
                f.write("\nModel Confidence: " + str(row['confidence']))
                f.write("\nRaw Confidence Response: " + row['confidence_raw_response'])
                f.write("\n" + "=" * 60 + "\n")

                # Only show first 5 examples per scenario to keep file manageable
                if prompt_key[1] == 5:
                    f.write(f"\n[Showing first 5 perturbations for {row['scenario']}. Full data in raw_results.csv]\n")
                    break

    print(f"Saved detailed prompts to {prompts_file}")


def create_violin_plots(analysis: Dict, output_dir: str):
    """Create violin plots for confidence score distributions across models and scenarios.

    This creates a stacked visualization matching the format of three_model_stacked_visualization.png:
    - 3 vertically stacked subplots (one per model: GPT, Claude, Gemini)
    - Each subplot shows scenarios as numbered points on x-axis
    - Violin plots with jittered points, mean, and 95% CI error bars
    - Consistent color scheme per scenario
    """

    print("\nCreating three-model stacked visualization...")

    # Prepare data structure for plotting
    model_data = {
        'gpt': {'scenarios': [], 'confidence_lists': []},
        'claude': {'scenarios': [], 'confidence_lists': []},
        'gemini': {'scenarios': [], 'confidence_lists': []}
    }

    # Get sorted list of scenarios for consistent ordering
    scenarios = sorted(analysis.keys())

    # Collect data for each model
    for scenario_idx, scenario in enumerate(scenarios):
        scenario_analysis = analysis[scenario]

        for model in ['gpt', 'claude', 'gemini']:
            if model in scenario_analysis:
                model_analysis = scenario_analysis[model]
                if 'confidence_values' in model_analysis and model_analysis['confidence_values']:
                    model_data[model]['scenarios'].append(scenario)
                    model_data[model]['confidence_lists'].append(model_analysis['confidence_values'])

    # Create figure with 3 subplots stacked vertically (matching the original format)
    # Using the exact figure size from create_three_model_stacked_visualization.py
    fig, axes = plt.subplots(3, 1, figsize=(14, 16.8))  # Same as original

    # Define colors for each scenario (same as original)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Model display names and order (matching original)
    model_names = {
        'gpt': 'GPT-4.1',  # Corrected to match actual model used
        'claude': 'Claude Opus 4.1',
        'gemini': 'Gemini 2.5 Pro'
    }
    model_order = ['gpt', 'claude', 'gemini']

    # Process each model
    for model_idx, model_key in enumerate(model_order):
        ax = axes[model_idx]
        data = model_data[model_key]

        if not data['scenarios']:
            # No data for this model
            ax.text(0.5, 0.5, f'No data available for {model_names[model_key]}',
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_xlim(0, len(scenarios) + 1)
            ax.set_ylim(0, 100)
        else:
            # Plot each scenario for this model
            plot_position = 0
            plotted_scenarios = []

            for scenario_idx, scenario in enumerate(scenarios):
                # Check if this model has data for this scenario
                if scenario in data['scenarios']:
                    data_idx = data['scenarios'].index(scenario)
                    confidence_values = data['confidence_lists'][data_idx]

                    if len(confidence_values) > 0:
                        plot_position += 1
                        plotted_scenarios.append(scenario_idx)

                        # Calculate statistics
                        mean_conf = np.mean(confidence_values)
                        lower_percentile = np.percentile(confidence_values, 2.5)
                        upper_percentile = np.percentile(confidence_values, 97.5)

                        # Add violin plot (matching original style)
                        violin_parts = ax.violinplot([confidence_values], [plot_position],
                                                    widths=0.3, showmeans=False, showmedians=False, showextrema=False)
                        for pc in violin_parts['bodies']:
                            pc.set_facecolor(colors[scenario_idx % len(colors)])
                            pc.set_edgecolor('none')
                            pc.set_alpha(0.3)

                        # Add jittered points (matching original)
                        np.random.seed(42 + scenario_idx)  # For reproducibility
                        x_jittered = np.random.normal(plot_position, 0.08, size=len(confidence_values))
                        ax.scatter(x_jittered, confidence_values, alpha=0.4, s=30,
                                  color=colors[scenario_idx % len(colors)])

                        # Add mean point (black dot, matching original)
                        ax.scatter(plot_position, mean_conf, color='black', s=80, zorder=5)

                        # Add error bars for 95% CI (matching original style)
                        ax.plot([plot_position, plot_position], [lower_percentile, upper_percentile],
                               color='black', linewidth=2, zorder=4)

                        # Add caps to the error bars (matching original)
                        cap_width = 0.1
                        ax.plot([plot_position - cap_width, plot_position + cap_width],
                               [lower_percentile, lower_percentile], color='black', linewidth=2, zorder=4)
                        ax.plot([plot_position - cap_width, plot_position + cap_width],
                               [upper_percentile, upper_percentile], color='black', linewidth=2, zorder=4)

            # Add a horizontal line at 50 for reference (neutral confidence)
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)

            # Set the x-ticks and labels - only show scenario numbers
            if plot_position > 0:
                ax.set_xticks(range(1, plot_position + 1))
                ax.set_xticklabels([f"{idx+1}" for idx in plotted_scenarios], fontsize=14)

        # Format y-axis
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylabel('Confidence (0-100)', fontsize=16)
        ax.set_ylim(0, 100)

        # Add title above each subplot (matching original)
        ax.set_title(model_names[model_key], fontsize=18, fontweight='bold', pad=10)

        # Add x-axis label only to the bottom panel (matching original)
        if model_idx == 2:  # Bottom panel
            ax.set_xlabel('Prompt Number', fontsize=16)

    # Adjust layout (matching original - no overall title, no legend)
    plt.tight_layout()

    # Save figure with same naming convention
    output_file = f"{output_dir}/three_model_stacked_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved visualization to {output_file}")
    plt.close()

    print(f"Saved three-model stacked visualization to {output_file}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Legal Scenario Perturbations with Irrelevant Statements"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        default=False,
        help="Run in test mode with limited rows (default: False)"
    )
    parser.add_argument(
        "--full-mode",
        action="store_true",
        help="Run on all data (overrides test mode)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of rows to process in test mode (default: 100)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["gpt", "claude", "gemini"],
        default=["gpt", "claude", "gemini"],
        help="Models to evaluate (default: all)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available (default: False)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Force start from beginning, ignoring any checkpoint (default: False)"
    )
    parser.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Clear existing checkpoint before starting (default: False)"
    )
    parser.add_argument(
        "--load-existing",
        action="store_true",
        default=True,
        help="Load existing results and analysis from files instead of running new evaluations (default: True)"
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force running new evaluations even if results exist (overrides --load-existing)"
    )
    parser.add_argument(
        "--regenerate-plots",
        action="store_true",
        help="Only regenerate plots from existing analysis.json file"
    )
    return parser.parse_args()


def main():
    """Main function to evaluate perturbations across all models."""
    args = parse_arguments()

    # Check if we should just regenerate plots
    if args.regenerate_plots:
        print("Regenerating plots from existing analysis...")
        print("=" * 60)

        analysis_file = str(RESULTS_DIR / "analysis.json")
        if not os.path.exists(analysis_file):
            print(f"Error: Analysis file not found at {analysis_file}")
            print("Please run the evaluation first to generate the analysis.")
            return

        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis = json.load(f)

        create_violin_plots(analysis, str(RESULTS_DIR))
        print("\nPlots regenerated successfully!")
        print(f"Check: {RESULTS_DIR}/three_model_stacked_visualization.png")
        return

    # Check if we should load existing results (default behavior)
    if args.load_existing and not args.force_rerun:
        print("Loading existing results and analysis...")
        print("=" * 60)

        # Check if results files exist
        raw_results_file = str(RESULTS_DIR / "raw_results.csv")
        analysis_file = str(RESULTS_DIR / "analysis.json")

        if os.path.exists(raw_results_file) and os.path.exists(analysis_file):
            print(f"Found existing results at: {RESULTS_DIR}")

            # Load the raw results
            results_df = pd.read_csv(raw_results_file)
            print(f"Loaded {len(results_df)} evaluation results from raw_results.csv")

            # Load the analysis
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
            print(f"Loaded analysis for {len(analysis)} scenarios")

            # Show summary
            print("\n" + "=" * 60)
            print("SUMMARY OF LOADED RESULTS")
            print("-" * 60)

            # Count results by model and scenario
            for model in ['gpt', 'claude', 'gemini']:
                model_results = results_df[results_df['model'] == model]
                if len(model_results) > 0:
                    scenarios = model_results['scenario'].unique()
                    print(f"{model.upper()}: {len(model_results)} evaluations across {len(scenarios)} scenarios")

            print("\nScenarios analyzed:")
            for idx, scenario in enumerate(sorted(analysis.keys()), 1):
                print(f"  {idx}. {scenario}")

            # Regenerate visualization with the new format
            print("\n" + "=" * 60)
            print("Generating three-model stacked visualization...")
            create_violin_plots(analysis, str(RESULTS_DIR))

            print("\n" + "=" * 60)
            print("LOADING COMPLETE")
            print(f"Results loaded from: {RESULTS_DIR}")
            print(f"New visualization saved as: three_model_stacked_visualization.png")
            print("\nTo force re-running evaluations, use: --force-rerun")
            print("To only regenerate plots, use: --regenerate-plots")
            print("=" * 60)
            return
        else:
            print("No existing results found. Running new evaluation...")
            print("(Use --force-rerun to override this check)")
            print()

    # If we get here, run the full evaluation
    # Determine if we're in test mode
    test_mode = args.test_mode and not args.full_mode

    print("Starting Irrelevant Statement Perturbation Evaluation")
    print("=" * 60)

    # Handle checkpoint clearing
    if args.clear_checkpoint and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        print("Cleared existing checkpoint")

    # Load checkpoint if resuming
    checkpoint_data = None
    completed_set = set()
    all_results = []

    if args.resume and not args.no_resume:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            completed_set = checkpoint_data.get('completed', set())
            all_results = checkpoint_data.get('all_results', [])
            print(f"Resuming from checkpoint with {len(all_results)} existing results")
        else:
            print("No checkpoint found, starting from beginning")
    elif args.no_resume:
        print("Ignoring checkpoint (--no-resume specified)")
    else:
        # Check if checkpoint exists and prompt user
        if os.path.exists(CHECKPOINT_FILE):
            print("\n⚠️  Checkpoint file detected!")
            print("Use --resume to continue from checkpoint, or --no-resume to start fresh")
            print("Use --clear-checkpoint to remove the checkpoint file")
            response = input("Do you want to resume from checkpoint? (y/n): ").strip().lower()
            if response == 'y':
                checkpoint_data = load_checkpoint()
                if checkpoint_data:
                    completed_set = checkpoint_data.get('completed', set())
                    all_results = checkpoint_data.get('all_results', [])
                    print(f"Resuming from checkpoint with {len(all_results)} existing results")

    if test_mode:
        print(f"RUNNING IN TEST MODE - Processing only {args.limit} total evaluations")
        print(f"To run on all data, use: python {__file__} --full-mode")
    else:
        print("RUNNING IN FULL MODE - Processing all data")

    print(f"Models to evaluate: {', '.join(args.models)}")
    print("=" * 60)

    # Load perturbations
    perturbations = load_perturbations(PERTURBATIONS_FILE)

    if test_mode:
        # In test mode, distribute evaluations evenly across models
        total_limit = args.limit
        num_models = len(args.models)
        evaluations_per_model = total_limit // num_models
        remaining_evaluations = total_limit % num_models  # Handle remainder

        print(f"Distributing {total_limit} evaluations across {num_models} models:")
        print(f"  - Each model gets approximately {evaluations_per_model} evaluations")
        if remaining_evaluations > 0:
            print(f"  - First {remaining_evaluations} model(s) get 1 extra evaluation")

    # Calculate total evaluations per scenario (1 original + number of perturbations per scenario)
    evaluations_per_scenario = {}
    for scenario_data in perturbations:
        # Each scenario has 1 original + N perturbations
        evaluations_per_scenario[scenario_data["scenario_name"]] = 1 + len(scenario_data.get('perturbations_with_irrelevant', []))

    # Define checkpoint callback
    def save_checkpoint_callback():
        """Callback to save checkpoint after each evaluation."""
        checkpoint_data = {
            'completed': completed_set,
            'all_results': all_results,
            'last_model': current_model,
            'last_scenario': current_scenario,
            'timestamp': datetime.now().isoformat()
        }
        save_checkpoint(checkpoint_data)

    # Process each model
    current_model = None
    current_scenario = None

    for model_idx, model_name in enumerate(args.models):
        current_model = model_name
        print(f"\n{'=' * 60}")
        print(f"Processing with {model_name.upper()}")
        print(f"{'=' * 60}")

        # Calculate limit for this model in test mode
        if test_mode:
            # Give extra evaluation to first models if there's a remainder
            model_limit = evaluations_per_model + (1 if model_idx < remaining_evaluations else 0)
            print(f"Model evaluation limit: {model_limit}")
            model_evaluation_count = 0
        else:
            model_limit = float('inf')
            model_evaluation_count = 0

        for scenario_idx, scenario_data in enumerate(perturbations):
            current_scenario = scenario_data["scenario_name"]

            if test_mode and model_evaluation_count >= model_limit:
                print(f"\nReached model limit of {model_limit} evaluations for {model_name}. Moving to next model.")
                break

            # Calculate how many evaluations this scenario will add
            scenario_evaluations = evaluations_per_scenario[scenario_data["scenario_name"]]

            # Check if processing this scenario would exceed the limit
            if test_mode and model_evaluation_count + scenario_evaluations > model_limit:
                # Process only partial perturbations to stay within limit
                remaining = model_limit - model_evaluation_count
                if remaining <= 0:
                    break

                print(f"\nProcessing only {remaining} evaluations from {scenario_data['scenario_name']} to stay within model limit")

                # Process scenario with limited perturbations
                limited_scenario_data = scenario_data.copy()
                if remaining > 1:
                    # Include original and some perturbations
                    limited_scenario_data['perturbations_with_irrelevant'] = scenario_data.get('perturbations_with_irrelevant', [])[:remaining-1]
                else:
                    # Only include original
                    limited_scenario_data['perturbations_with_irrelevant'] = []

                scenario_results = process_scenario_perturbations(
                    limited_scenario_data, model_name,
                    completed_set=completed_set,
                    checkpoint_callback=save_checkpoint_callback
                )
                all_results.extend(scenario_results)
                model_evaluation_count += len(scenario_results)
            else:
                # Process full scenario
                total_so_far = len(all_results)
                print(f"\n[Progress: {total_so_far}/{args.limit if test_mode else 'all'} total evaluations completed]")
                print(f"Processing scenario {scenario_idx + 1}/{len(perturbations)}: {scenario_data['scenario_name']}")

                scenario_results = process_scenario_perturbations(
                    scenario_data, model_name,
                    completed_set=completed_set,
                    checkpoint_callback=save_checkpoint_callback
                )
                all_results.extend(scenario_results)
                model_evaluation_count += len(scenario_results)

                print(f"  Completed {len(scenario_results)} evaluations (Model total: {model_evaluation_count}, Overall total: {len(all_results)})")

    # Combine current results with checkpoint data for complete analysis
    print("\n" + "=" * 60)
    print("Analyzing results...")

    # If we have a checkpoint, load it to get complete data
    complete_results = all_results.copy()

    # Check if we have checkpoint data with results to merge
    if checkpoint_data and 'all_results' in checkpoint_data:
        # Add checkpoint results that aren't in current results
        current_ids = set()
        for r in all_results:
            current_ids.add((r['model'], r['scenario'], r['perturbation_id']))

        for r in checkpoint_data['all_results']:
            result_id = (r['model'], r['scenario'], r['perturbation_id'])
            if result_id not in current_ids:
                complete_results.append(r)

        merged_count = len(complete_results) - len(all_results)
        if merged_count > 0:
            print(f"  Combined {len(all_results)} new results with {merged_count} checkpoint results")
            print(f"  Total results for analysis: {len(complete_results)}")

    results_df = pd.DataFrame(complete_results)
    analysis = analyze_results(results_df)

    # Save all results (including checkpoint data for completeness)
    save_results(complete_results, analysis, str(RESULTS_DIR))

    # Create violin plots
    create_violin_plots(analysis, str(RESULTS_DIR))

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("-" * 60)
    if test_mode:
        print(f"Mode: TEST MODE (limit: {args.limit})")
    else:
        print("Mode: FULL MODE")
    print(f"Models evaluated: {', '.join(args.models)}")
    print(f"Total scenarios processed: {len(set(r['scenario'] for r in all_results))}/{len(perturbations)}")
    print(f"Total evaluations completed: {len(all_results)}")
    print(f"Results saved to: {str(RESULTS_DIR)}")

    # Clean up checkpoint files after successful completion
    if not test_mode and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        print("Cleaned up checkpoint files after successful completion")

    if test_mode:
        print(f"\nTo run full evaluation, use: python {__file__} --full-mode")
    print("=" * 60)


if __name__ == "__main__":
    main()