print("Claude Opus 4.1 Perturbation Analysis - Version 1.0.0")

import os
import time
import json
import pandas as pd
import numpy as np
import re
from datetime import datetime
from anthropic import Anthropic
from anthropic._exceptions import RateLimitError, APIError, APIStatusError
from config import ANTHROPIC_API_KEY

# Configuration
PERTURBATIONS_FILE = "data/perturbations.json"
OUTPUT_EXCEL = "results/claude_opus_perturbation_results.xlsx"

# Model configuration
MODEL_NAME = "claude-opus-4-1-20250805"  # Claude Opus 4.1 model

# Processing configuration
BATCH_SIZE = 100  # Process in batches to manage rate limits
DELAY_BETWEEN_REQUESTS = 1.0  # Delay in seconds between API calls
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 10
MAX_RETRY_DELAY = 60
REASONING_MODEL_RUNS = 10  # Number of runs for binary format to approximate logprobs

# Initialize Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY)

def load_perturbations(file_path):
    """Load existing perturbations from JSON file."""
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
    
    print(f"Loaded {len(prompt_parts_list)} prompts with perturbations")
    return prompt_parts_list, rephrasings_list

def load_existing_results(output_file):
    """Load existing results to avoid reprocessing."""
    processed = set()
    if os.path.exists(output_file):
        print(f"Loading existing results from {output_file}...")
        try:
            df = pd.read_excel(output_file)
            for _, row in df.iterrows():
                key = (row['Original Main Part'], row['Rephrased Main Part'])
                processed.add(key)
            print(f"Found {len(processed)} already processed perturbations")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    return processed

def retry_with_exponential_backoff(func, max_retries=MAX_RETRIES, 
                                  initial_delay=INITIAL_RETRY_DELAY, 
                                  max_delay=MAX_RETRY_DELAY):
    """Retry a function with exponential backoff on API errors."""
    num_retries = 0
    delay = initial_delay
    
    while True:
        try:
            return func()
        except (RateLimitError, APIError, APIStatusError) as e:
            num_retries += 1
            if num_retries > max_retries:
                raise Exception(f"Maximum retries ({max_retries}) exceeded: {str(e)}")
            
            # Add jitter to prevent synchronized retries
            import random
            jitter = random.uniform(0.8, 1.2)
            sleep_time = min(delay * jitter, max_delay)
            
            print(f"  API error: {str(e)[:100]}... Retrying in {sleep_time:.1f}s (retry {num_retries}/{max_retries})")
            time.sleep(sleep_time)
            
            # Exponential backoff
            delay = min(delay * 1.5, max_delay)

def process_with_claude(prompt):
    """Process a single prompt with Claude API."""
    def api_call():
        return client.messages.create(
            model=MODEL_NAME,
            max_tokens=500,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}]
        )
    
    try:
        response = retry_with_exponential_backoff(api_call)
        return response.content[0].text.strip() if response.content else ""
    except Exception as e:
        print(f"  Error processing prompt: {e}")
        return None

def extract_confidence_value(response_text):
    """Extract numerical confidence value from response."""
    try:
        match = re.search(r'\b(\d+)\b', response_text)
        if match:
            value = int(match.group(1))
            if 0 <= value <= 100:
                return value
    except:
        pass
    return None

def approximate_logprobs(prompt, target_tokens, num_runs=REASONING_MODEL_RUNS):
    """
    Approximate log probabilities by running the model multiple times.
    This is for the binary format where we want Yes/No type answers.
    """
    token_counts = {token: 0 for token in target_tokens}
    responses = []
    
    for run in range(num_runs):
        response = process_with_claude(prompt)
        if response:
            responses.append(response)
            # Check which target token appears in the response
            for token in target_tokens:
                if token in response:
                    token_counts[token] += 1
                    break  # Count only the first matching token
        
        # Small delay between runs to avoid rate limits
        if run < num_runs - 1:
            time.sleep(0.5)
    
    # Calculate approximate probabilities
    total_valid_responses = sum(token_counts.values())
    if total_valid_responses == 0:
        # If no target tokens found, use uniform distribution
        probs = {token: 1.0 / len(target_tokens) for token in target_tokens}
    else:
        probs = {token: count / num_runs for token, count in token_counts.items()}
    
    # Return the most common response
    most_common_response = max(set(responses), key=responses.count) if responses else ""
    
    return probs, most_common_response

def process_perturbations(prompt_parts_list, rephrasings_list, processed_set, 
                         skip_binary_approximation=True):
    """
    Process all perturbations with Claude.
    
    Args:
        skip_binary_approximation: If True, skip the expensive logprob approximation
                                  for binary format (similar to GPT-5 treatment)
    """
    results = []
    total_processed = 0
    total_skipped = 0
    
    for prompt_idx, (prompt_parts, rephrasings) in enumerate(zip(prompt_parts_list, rephrasings_list)):
        orig_main, orig_format, target_tokens, confidence_format = prompt_parts
        
        print(f"\nProcessing prompt {prompt_idx + 1}/{len(prompt_parts_list)}")
        print(f"Original: {orig_main[:100]}...")
        
        for rephrase_idx, rephrased_main in enumerate(rephrasings):
            # Check if already processed
            if (orig_main, rephrased_main) in processed_set:
                total_skipped += 1
                continue
            
            # Process binary format
            binary_prompt = f"{rephrased_main} {orig_format}"
            
            if skip_binary_approximation:
                # Just get a single response without approximating logprobs
                binary_response = process_with_claude(binary_prompt)
                if binary_response is None:
                    print(f"  Failed to process binary prompt {rephrase_idx + 1}")
                    continue
                
                # Set probabilities to indicate this wasn't computed
                token_1_prob = 0.0
                token_2_prob = 0.0
                odds_ratio = 0.0
            else:
                # Approximate logprobs through multiple runs (expensive!)
                probs, binary_response = approximate_logprobs(binary_prompt, target_tokens)
                if not binary_response:
                    print(f"  Failed to process binary prompt {rephrase_idx + 1}")
                    continue
                
                token_1_prob = probs.get(target_tokens[0], 0.0)
                token_2_prob = probs.get(target_tokens[1], 0.0)
                odds_ratio = token_1_prob / token_2_prob if token_2_prob > 0 else float('inf')
            
            # Process confidence format
            confidence_prompt = f"{rephrased_main} {confidence_format}"
            confidence_response = process_with_claude(confidence_prompt)
            
            if confidence_response is None:
                print(f"  Failed to process confidence prompt {rephrase_idx + 1}")
                confidence_response = ""
                confidence_value = None
            else:
                confidence_value = extract_confidence_value(confidence_response)
            
            # Store result
            result = {
                "Model": MODEL_NAME,
                "Original Main Part": orig_main,
                "Response Format": orig_format,
                "Confidence Format": confidence_format,
                "Rephrased Main Part": rephrased_main,
                "Full Rephrased Prompt": binary_prompt,
                "Full Confidence Prompt": confidence_prompt,
                "Model Response": binary_response,
                "Model Confidence Response": confidence_response,
                "Log Probabilities": "N/A (Claude does not support logprobs)",
                "Token_1_Prob": token_1_prob,
                "Token_2_Prob": token_2_prob,
                "Odds_Ratio": odds_ratio,
                "Confidence Value": confidence_value,
                "Weighted Confidence": confidence_value  # For Claude, no weighted confidence
            }
            
            results.append(result)
            total_processed += 1
            
            # Progress update
            if total_processed % 10 == 0:
                print(f"  Processed {total_processed} perturbations...")
            
            # Rate limiting
            time.sleep(DELAY_BETWEEN_REQUESTS)
            
            # Save checkpoint periodically
            if total_processed % BATCH_SIZE == 0:
                save_results(results, OUTPUT_EXCEL, append=True)
                print(f"  Saved checkpoint with {len(results)} results")
                results = []  # Clear results after saving
    
    # Save any remaining results
    if results:
        save_results(results, OUTPUT_EXCEL, append=True)
    
    print(f"\nProcessing complete!")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped (already processed): {total_skipped}")
    
    return total_processed

def save_results(results, output_file, append=False):
    """Save results to Excel file."""
    if not results:
        return
    
    df = pd.DataFrame(results)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if append and os.path.exists(output_file):
        # Append to existing file
        try:
            existing_df = pd.read_excel(output_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_excel(output_file, index=False)
            print(f"Appended {len(df)} results to {output_file}")
        except Exception as e:
            print(f"Error appending to existing file: {e}")
            # Save to a new file instead
            backup_file = output_file.replace(".xlsx", f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            df.to_excel(backup_file, index=False)
            print(f"Saved {len(df)} results to backup file: {backup_file}")
    else:
        # Create new file
        df.to_excel(output_file, index=False)
        print(f"Saved {len(df)} results to {output_file}")

def estimate_cost(num_perturbations, skip_binary_approximation=True):
    """Estimate the cost of processing perturbations."""
    # Claude Opus pricing (approximate - check current pricing)
    # Input: $15 per million tokens
    # Output: $75 per million tokens
    
    avg_input_tokens_per_request = 150  # Rough estimate
    avg_output_tokens_per_request = 20  # Rough estimate
    
    if skip_binary_approximation:
        # 2 requests per perturbation (binary + confidence)
        total_requests = num_perturbations * 2
    else:
        # Multiple runs for binary + 1 for confidence
        total_requests = num_perturbations * (REASONING_MODEL_RUNS + 1)
    
    total_input_tokens = total_requests * avg_input_tokens_per_request
    total_output_tokens = total_requests * avg_output_tokens_per_request
    
    input_cost = (total_input_tokens / 1_000_000) * 15.00
    output_cost = (total_output_tokens / 1_000_000) * 75.00
    total_cost = input_cost + output_cost
    
    return total_cost, total_requests

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("CLAUDE OPUS 4.1 PERTURBATION ANALYSIS")
    print("="*60)
    
    # Check for API key
    if not ANTHROPIC_API_KEY:
        print("\nERROR: ANTHROPIC_API_KEY not found!")
        print("Please set ANTHROPIC_API_KEY in your config.py or .env file")
        return
    
    # Configuration for this run
    SKIP_BINARY_APPROXIMATION = True  # Set to False to approximate logprobs (expensive!)
    
    if SKIP_BINARY_APPROXIMATION:
        print("\nNOTE: Skipping logprob approximation for binary format")
        print("      (Similar to GPT-5 treatment in the original analysis)")
        print("      Only verbalized confidence will be used for analysis.")
    else:
        print(f"\nWARNING: Logprob approximation enabled!")
        print(f"         Each binary prompt will be run {REASONING_MODEL_RUNS} times")
        print(f"         This will significantly increase processing time and cost!")
    
    # Load perturbations
    prompt_parts_list, rephrasings_list = load_perturbations(PERTURBATIONS_FILE)
    
    # Calculate total perturbations
    total_perturbations = sum(len(rephrasings) for rephrasings in rephrasings_list)
    print(f"\nTotal perturbations to process: {total_perturbations}")
    
    # Load existing results
    processed_set = load_existing_results(OUTPUT_EXCEL)
    remaining = total_perturbations - len(processed_set)
    print(f"Remaining to process: {remaining}")
    
    if remaining == 0:
        print("\nAll perturbations have already been processed!")
        return
    
    # Estimate time and cost
    if SKIP_BINARY_APPROXIMATION:
        estimated_time = remaining * (DELAY_BETWEEN_REQUESTS * 2 + 3)  # 2 requests per perturbation
    else:
        estimated_time = remaining * (DELAY_BETWEEN_REQUESTS * (REASONING_MODEL_RUNS + 1) + 10)
    
    print(f"\nEstimated processing time: {estimated_time/60:.1f} minutes")
    
    estimated_cost, total_requests = estimate_cost(remaining, SKIP_BINARY_APPROXIMATION)
    print(f"Estimated API calls: {total_requests}")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    
    # Confirm before proceeding
    response = input("\nProceed with processing? (yes/no): ")
    if response.lower() != 'yes':
        print("Processing cancelled.")
        return
    
    # Process perturbations
    start_time = time.time()
    total_processed = process_perturbations(
        prompt_parts_list, 
        rephrasings_list, 
        processed_set,
        skip_binary_approximation=SKIP_BINARY_APPROXIMATION
    )
    elapsed_time = time.time() - start_time
    
    if total_processed > 0:
        print(f"\nTotal processing time: {elapsed_time/60:.1f} minutes")
        print(f"Average time per perturbation: {elapsed_time/total_processed:.2f} seconds")
    print(f"\nResults saved to: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()