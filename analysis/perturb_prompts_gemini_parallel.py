#!/usr/bin/env python3
"""
Gemini 2.5 Pro Perturbation Analysis - Parallel Processing Version

Uses concurrent requests with intelligent rate limiting for fast processing.
Processes ~200-400 perturbations/minute instead of ~1-2/minute.
"""

import os
import time
import json
import pandas as pd
import numpy as np
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import google.generativeai as genai
from config import GEMINI_API_KEY

# Configuration
PERTURBATIONS_FILE = "../data/perturbations.json"
OUTPUT_EXCEL = "../results/gemini_perturbation_results.xlsx"
CHECKPOINT_FILE = "../results/gemini_checkpoint.xlsx"

# Model configuration
MODEL_NAME = "gemini-2.5-pro"

# Parallel processing configuration
MAX_WORKERS = 20  # Number of concurrent threads
REQUESTS_PER_SECOND = 2.3  # ~140 RPM with buffer
START_FROM_PROMPT = 2  # Start from prompt 3 (affiliates question)
CHECKPOINT_EVERY = 50  # Save checkpoint every N perturbations

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# Global state for rate limiting
rate_lock = Lock()
request_times = []


def rate_limit_wait():
    """Intelligent rate limiting - wait only if needed."""
    global request_times

    with rate_lock:
        now = time.time()
        # Remove requests older than 1 second
        request_times = [t for t in request_times if now - t < 1.0]

        # If at limit, wait
        if len(request_times) >= REQUESTS_PER_SECOND:
            sleep_time = 1.0 - (now - request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
            # Clean again after wait
            now = time.time()
            request_times = [t for t in request_times if now - t < 1.0]

        # Record this request
        request_times.append(time.time())


def call_gemini_with_retry(prompt, use_logprobs=True, max_retries=5):
    """Call Gemini API with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            rate_limit_wait()

            generation_config = {
                "temperature": 0.0,
                "max_output_tokens": 500,
            }
            if use_logprobs:
                generation_config["response_logprobs"] = True

            response = model.generate_content(prompt, generation_config=generation_config)

            text = response.text.strip() if response.text else ""
            logprobs = None

            if use_logprobs and hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'logprobs_result'):
                    logprobs = candidate.logprobs_result

            return text, logprobs

        except Exception as e:
            error_str = str(e)

            # Handle rate limits
            if '429' in error_str or 'quota' in error_str.lower():
                wait_time = (2 ** attempt) * 30  # Exponential: 30, 60, 120, 240, 480 seconds
                print(f"  [Thread] Rate limit, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            # Other errors
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                print(f"  [Thread] Error after {max_retries} attempts: {error_str[:100]}")
                return None, None

    return None, None


def extract_token_probabilities(logprobs_data, target_tokens):
    """Extract probabilities for specific target tokens."""
    token_1_prob = 0.0
    token_2_prob = 0.0

    if not logprobs_data:
        return token_1_prob, token_2_prob

    try:
        if hasattr(logprobs_data, 'top_candidates') and logprobs_data.top_candidates:
            first_position = logprobs_data.top_candidates[0]

            for candidate in first_position.candidates:
                token_text = candidate.token
                log_prob = candidate.log_probability

                if token_text == target_tokens[0]:
                    token_1_prob = np.exp(log_prob)
                elif token_text == target_tokens[1]:
                    token_2_prob = np.exp(log_prob)
    except Exception as e:
        pass

    return token_1_prob, token_2_prob


def extract_confidence_value(response_text):
    """Extract numerical confidence value."""
    match = re.search(r'\b(\d+)\b', response_text)
    if match:
        return int(match.group(1))
    return None


def calculate_weighted_confidence(logprobs_data):
    """Calculate weighted confidence from logprobs."""
    if not logprobs_data:
        return None

    try:
        if not hasattr(logprobs_data, 'top_candidates') or len(logprobs_data.top_candidates) < 1:
            return None

        positions = logprobs_data.top_candidates
        first_pos = positions[0] if len(positions) > 0 else None
        second_pos = positions[1] if len(positions) > 1 else None
        third_pos = positions[2] if len(positions) > 2 else None

        if not first_pos:
            return None

        value_probs = {}

        for first_cand in first_pos.candidates[:19]:
            first_token = first_cand.token.strip()
            first_log_prob = first_cand.log_probability
            first_prob = np.exp(first_log_prob)

            if first_token.isdigit():
                if len(first_token) == 1:
                    first_digit = int(first_token)

                    if second_pos and 1 <= first_digit <= 9:
                        for second_cand in second_pos.candidates[:19]:
                            second_token = second_cand.token.strip()
                            second_log_prob = second_cand.log_probability
                            second_prob = np.exp(second_log_prob)

                            if second_token.isdigit() and len(second_token) == 1:
                                second_digit = int(second_token)
                                value = first_digit * 10 + second_digit

                                if value == 10 and third_pos:
                                    for third_cand in third_pos.candidates[:19]:
                                        third_token = third_cand.token.strip()
                                        if third_token == '0':
                                            third_log_prob = third_cand.log_probability
                                            third_prob = np.exp(third_log_prob)
                                            combined_prob = first_prob * second_prob * third_prob
                                            value_probs[100] = value_probs.get(100, 0) + combined_prob
                                else:
                                    combined_prob = first_prob * second_prob
                                    value_probs[value] = value_probs.get(value, 0) + combined_prob
                    else:
                        value_probs[first_digit] = value_probs.get(first_digit, 0) + first_prob
                else:
                    value = int(first_token)
                    if 0 <= value <= 100:
                        value_probs[value] = value_probs.get(value, 0) + first_prob

        if value_probs:
            total_prob = sum(value_probs.values())
            if total_prob > 0:
                return sum(v * p for v, p in value_probs.items()) / total_prob

    except Exception as e:
        pass

    return None


def process_single_perturbation(args):
    """Process a single perturbation (both binary and confidence)."""
    prompt_idx, rephrase_idx, orig_main, rephrased_main, orig_format, confidence_format, target_tokens = args

    # Binary request
    binary_prompt = f"{rephrased_main} {orig_format}"
    binary_response, binary_logprobs = call_gemini_with_retry(binary_prompt, use_logprobs=True)

    if binary_response is None:
        return None

    # Confidence request
    confidence_prompt = f"{rephrased_main} {confidence_format}"
    confidence_response, confidence_logprobs = call_gemini_with_retry(confidence_prompt, use_logprobs=True)

    if confidence_response is None:
        return None

    # Extract metrics
    token_1_prob, token_2_prob = extract_token_probabilities(binary_logprobs, target_tokens)
    odds_ratio = token_1_prob / token_2_prob if token_2_prob > 0 else float('inf')

    confidence_value = extract_confidence_value(confidence_response)
    weighted_confidence = calculate_weighted_confidence(confidence_logprobs)

    return {
        "Model": MODEL_NAME,
        "Original Main Part": orig_main,
        "Response Format": orig_format,
        "Confidence Format": confidence_format,
        "Rephrased Main Part": rephrased_main,
        "Full Rephrased Prompt": binary_prompt,
        "Full Confidence Prompt": confidence_prompt,
        "Model Response": binary_response,
        "Model Confidence Response": confidence_response,
        "Log Probabilities": str(binary_logprobs) if binary_logprobs else "N/A",
        "Token_1_Prob": token_1_prob,
        "Token_2_Prob": token_2_prob,
        "Odds_Ratio": odds_ratio,
        "Confidence Value": confidence_value,
        "Weighted Confidence": weighted_confidence
    }


def load_perturbations(file_path):
    """Load perturbations from JSON."""
    print(f"Loading perturbations from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompt_parts_list = []
    rephrasings_list = []

    for item in data:
        prompt_parts = (
            item['original_main'],
            item['response_format'],
            tuple(item['target_tokens']),
            item['confidence_format']
        )
        prompt_parts_list.append(prompt_parts)
        rephrasings_list.append(item['rephrasings'])

    print(f"Loaded {len(prompt_parts_list)} prompts")
    return prompt_parts_list, rephrasings_list


def load_existing_results(output_file):
    """Load already-processed perturbations."""
    processed = set()
    if os.path.exists(output_file):
        try:
            df = pd.read_excel(output_file)
            for _, row in df.iterrows():
                key = (row['Original Main Part'], row['Rephrased Main Part'])
                processed.add(key)
            print(f"Found {len(processed)} already processed")
        except Exception as e:
            print(f"Could not load existing results: {e}")
    return processed


def save_checkpoint(results, output_file):
    """Save checkpoint to avoid losing progress."""
    if not results:
        return

    try:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if os.path.exists(output_file):
            existing_df = pd.read_excel(output_file)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_excel(output_file, index=False)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("GEMINI 2.5 PRO - PARALLEL PROCESSING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Concurrent workers: {MAX_WORKERS}")
    print(f"  - Target rate: ~{REQUESTS_PER_SECOND * 60:.0f} RPM")
    print(f"  - Starting from prompt: {START_FROM_PROMPT + 1}")

    if not GEMINI_API_KEY:
        print("\n✗ ERROR: GEMINI_API_KEY not found!")
        return

    # Load data
    prompt_parts_list, rephrasings_list = load_perturbations(PERTURBATIONS_FILE)
    processed_set = load_existing_results(OUTPUT_EXCEL)

    # Build task list
    tasks = []
    for prompt_idx in range(START_FROM_PROMPT, len(prompt_parts_list)):
        orig_main, orig_format, target_tokens, confidence_format = prompt_parts_list[prompt_idx]
        rephrasings = rephrasings_list[prompt_idx]

        for rephrase_idx, rephrased_main in enumerate(rephrasings):
            if (orig_main, rephrased_main) in processed_set:
                continue

            tasks.append((
                prompt_idx,
                rephrase_idx,
                orig_main,
                rephrased_main,
                orig_format,
                confidence_format,
                target_tokens
            ))

    total_tasks = len(tasks)
    print(f"\nTotal perturbations to process: {total_tasks}")
    print(f"Estimated time: {total_tasks / (REQUESTS_PER_SECOND * 60 / 2):.1f} minutes")

    response = input("\nProceed? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return

    # Process in parallel
    print(f"\nProcessing with {MAX_WORKERS} workers...\n")
    start_time = time.time()
    results = []
    completed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_single_perturbation, task): task for task in tasks}

        for future in as_completed(future_to_task):
            result = future.result()

            if result:
                results.append(result)
                completed += 1
            else:
                failed += 1

            # Progress update
            if (completed + failed) % 10 == 0:
                elapsed = time.time() - start_time
                rate = completed / (elapsed / 60) if elapsed > 0 else 0
                remaining = total_tasks - (completed + failed)
                eta = remaining / rate if rate > 0 else 0

                print(f"Progress: {completed}/{total_tasks} | "
                      f"Rate: {rate:.1f}/min | "
                      f"Failed: {failed} | "
                      f"ETA: {eta:.1f}min")

            # Checkpoint
            if len(results) >= CHECKPOINT_EVERY:
                save_checkpoint(results, OUTPUT_EXCEL)
                results = []

    # Final save
    if results:
        save_checkpoint(results, OUTPUT_EXCEL)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ Complete!")
    print(f"  - Processed: {completed}")
    print(f"  - Failed: {failed}")
    print(f"  - Time: {elapsed/60:.1f} minutes")
    print(f"  - Rate: {completed/(elapsed/60):.1f} perturbations/minute")
    print(f"  - Results: {OUTPUT_EXCEL}")


if __name__ == "__main__":
    main()