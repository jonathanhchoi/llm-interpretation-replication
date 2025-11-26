print("Claude Opus 4.1 Batch Perturbation Analysis - Confidence Only Mode - Version 2.1.0")

import os
import time
import json
import pandas as pd
import numpy as np
import re
import argparse
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from anthropic import Anthropic
from anthropic._exceptions import RateLimitError, APIError, APIStatusError
from config import ANTHROPIC_API_KEY

# Configuration
PERTURBATIONS_FILE = "data/perturbations.json"
OUTPUT_EXCEL = "results/claude_opus_batch_perturbation_results.xlsx"
BATCH_RESULTS_DIR = "results/claude_batch_results"

# Model configuration - Updated to Claude Opus 4.1
MODEL_NAME = "claude-opus-4-1-20250805"  # Claude Opus 4.1 model

# Batch processing configuration
MAX_BATCH_SIZE = 10000  # Claude's max batch size
DEFAULT_TEST_SIZE = 100  # Default number of cases for testing
BATCH_CHECK_INTERVAL = 30  # Seconds between checking batch status
MAX_BATCH_WAIT_TIME = 3600 * 24  # Maximum 24 hours to wait for batch completion

# Initialize Anthropic client
client = Anthropic(api_key=ANTHROPIC_API_KEY)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Claude Opus 4.1 Batch Perturbation Analysis')
    parser.add_argument(
        '--test',
        action='store_true',
        help='Enable test mode to process only a subset of cases'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=DEFAULT_TEST_SIZE,
        help=f'Number of cases to process in test mode (default: {DEFAULT_TEST_SIZE})'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        default=True,
        help='Skip already processed perturbations (default: True)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_EXCEL,
        help=f'Output Excel file path (default: {OUTPUT_EXCEL})'
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Automatically proceed without confirmation'
    )
    return parser.parse_args()

def load_perturbations(file_path: str, limit: Optional[int] = None) -> Tuple[List, List]:
    """Load perturbations from JSON file with optional limit."""
    print(f"Loading perturbations from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if limit:
        print(f"Limiting to first {limit} prompts for testing")
        data = data[:limit]

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

    total_perturbations = sum(len(rephrasings) for rephrasings in rephrasings_list)
    print(f"Loaded {len(prompt_parts_list)} prompts with {total_perturbations} total perturbations")
    return prompt_parts_list, rephrasings_list

def load_existing_results(output_file: str) -> set:
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

def prepare_batch_requests(
    prompt_parts_list: List,
    rephrasings_list: List,
    processed_set: set,
    test_mode: bool = False,
    test_size: int = DEFAULT_TEST_SIZE
) -> List[Dict]:
    """Prepare batch API requests for all perturbations."""
    batch_requests = []
    request_id_map = {}
    total_count = 0

    for prompt_idx, (prompt_parts, rephrasings) in enumerate(zip(prompt_parts_list, rephrasings_list)):
        if test_mode and total_count >= test_size:
            break

        orig_main, orig_format, target_tokens, confidence_format = prompt_parts

        for rephrase_idx, rephrased_main in enumerate(rephrasings):
            if test_mode and total_count >= test_size:
                break

            # Check if already processed
            if (orig_main, rephrased_main) in processed_set:
                continue

            # Create unique ID for tracking (confidence only)
            confidence_request_id = f"confidence_{prompt_idx}_{rephrase_idx}_{uuid.uuid4().hex[:8]}"

            # Confidence format request only
            confidence_prompt = f"{rephrased_main} {confidence_format}"
            batch_requests.append({
                "custom_id": confidence_request_id,
                "params": {
                    "model": MODEL_NAME,
                    "max_tokens": 500,
                    "temperature": 1.0,
                    "messages": [{"role": "user", "content": confidence_prompt}]
                }
            })

            # Store mapping for result processing
            base_info = {
                "prompt_idx": prompt_idx,
                "rephrase_idx": rephrase_idx,
                "orig_main": orig_main,
                "orig_format": orig_format,
                "confidence_format": confidence_format,
                "rephrased_main": rephrased_main,
                "target_tokens": target_tokens
            }
            request_id_map[confidence_request_id] = {**base_info, "type": "confidence"}

            total_count += 1

    print(f"Prepared {len(batch_requests)} batch requests for {total_count} perturbations")
    return batch_requests, request_id_map

def create_batch_file(batch_requests: List[Dict], batch_dir: str) -> str:
    """Create a JSONL file for batch processing."""
    os.makedirs(batch_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_file = os.path.join(batch_dir, f"batch_input_{timestamp}.jsonl")

    with open(batch_file, 'w') as f:
        for request in batch_requests:
            f.write(json.dumps(request) + '\n')

    print(f"Created batch file: {batch_file}")
    return batch_file

def submit_batch(batch_file: str) -> str:
    """Submit batch to Claude API."""
    print("Submitting batch to Claude API...")

    # Read the JSONL file and parse requests
    requests = []
    with open(batch_file, 'r') as f:
        for line in f:
            requests.append(json.loads(line))

    # Submit the batch
    batch_response = client.beta.messages.batches.create(
        requests=requests
    )

    batch_id = batch_response.id
    print(f"Batch submitted successfully. Batch ID: {batch_id}")
    return batch_id

def wait_for_batch_completion(batch_id: str, max_wait_time: int = MAX_BATCH_WAIT_TIME) -> Dict:
    """Wait for batch to complete and return results."""
    print(f"Waiting for batch {batch_id} to complete...")
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        try:
            batch_status = client.beta.messages.batches.retrieve(batch_id)

            # Get counts from request_counts attributes
            processing = getattr(batch_status.request_counts, 'processing', 0)
            succeeded = getattr(batch_status.request_counts, 'succeeded', 0)
            errored = getattr(batch_status.request_counts, 'errored', 0)
            total = processing + succeeded + errored

            print(f"Batch status: {batch_status.processing_status} "
                  f"(Processing: {processing}, Succeeded: {succeeded}, Failed: {errored}, Total: {total})")

            if batch_status.processing_status == "ended":
                if succeeded > 0:
                    print(f"Batch completed successfully! "
                          f"Succeeded: {succeeded}, "
                          f"Failed: {errored}")

                    # Retrieve results
                    results = client.beta.messages.batches.results(batch_id)
                    return {"status": "success", "results": results}
                else:
                    return {"status": "failed", "message": "All requests failed"}

            elif batch_status.processing_status == "failed":
                return {"status": "failed", "message": "Batch processing failed"}

            # Wait before checking again
            time.sleep(BATCH_CHECK_INTERVAL)

        except Exception as e:
            print(f"Error checking batch status: {e}")
            time.sleep(BATCH_CHECK_INTERVAL)

    return {"status": "timeout", "message": f"Batch did not complete within {max_wait_time} seconds"}

def process_batch_results(batch_results: Dict, request_id_map: Dict) -> List[Dict]:
    """Process batch results into structured format."""
    processed_results = {}

    for result in batch_results:
        custom_id = result.custom_id
        request_info = request_id_map.get(custom_id)

        if not request_info:
            print(f"Warning: Unknown request ID: {custom_id}")
            continue

        # Extract response text
        if result.result and result.result.type == "succeeded":
            response_text = result.result.message.content[0].text.strip()
        else:
            response_text = ""
            print(f"Warning: Failed request {custom_id}")

        # Create or update result entry
        result_key = (request_info["orig_main"], request_info["rephrased_main"])

        if result_key not in processed_results:
            processed_results[result_key] = {
                "Model": MODEL_NAME,
                "Original Main Part": request_info["orig_main"],
                "Response Format": request_info["orig_format"],
                "Confidence Format": request_info["confidence_format"],
                "Rephrased Main Part": request_info["rephrased_main"],
                "Target Tokens": request_info["target_tokens"],
                "Model Confidence Response": response_text,
                "Full Confidence Prompt": f"{request_info['rephrased_main']} {request_info['confidence_format']}"
            }

            # Extract confidence value
            confidence_value = extract_confidence_value(response_text)
            processed_results[result_key]["Confidence Value"] = confidence_value
            processed_results[result_key]["Weighted Confidence"] = confidence_value

            # Add N/A for binary format fields since we're not using that method
            processed_results[result_key]["Model Response"] = "N/A (Confidence-only mode)"
            processed_results[result_key]["Full Rephrased Prompt"] = "N/A (Confidence-only mode)"

    # Convert to list and add default values
    results_list = []
    for result in processed_results.values():
        # Add default values for fields not computed in batch mode
        result["Log Probabilities"] = "N/A (Batch processing - logprobs not available)"
        result["Token_1_Prob"] = 0.0
        result["Token_2_Prob"] = 0.0
        result["Odds_Ratio"] = 0.0
        results_list.append(result)

    return results_list

def extract_confidence_value(response_text: str) -> Optional[int]:
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

def save_results(results: List[Dict], output_file: str, append: bool = False):
    """Save results to Excel file."""
    if not results:
        return

    df = pd.DataFrame(results)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if append and os.path.exists(output_file):
        try:
            existing_df = pd.read_excel(output_file)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_excel(output_file, index=False)
            print(f"Appended {len(df)} results to {output_file}")
        except Exception as e:
            print(f"Error appending to existing file: {e}")
            backup_file = output_file.replace(".xlsx", f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            df.to_excel(backup_file, index=False)
            print(f"Saved {len(df)} results to backup file: {backup_file}")
    else:
        df.to_excel(output_file, index=False)
        print(f"Saved {len(df)} results to {output_file}")

def estimate_cost(num_requests: int) -> Tuple[float, float]:
    """
    Estimate the cost of processing requests (confidence-only mode).
    Updated pricing as per https://claude.com/pricing (as of 2025):
    - Claude Opus 4.1:
      - Input: $15 per million tokens
      - Output: $75 per million tokens
    - Batch API: 50% discount
    Note: num_requests here refers to confidence requests only (no binary format requests)
    """
    # Pricing for Claude 3.5 Opus (per million tokens)
    INPUT_PRICE_PER_M = 15.00  # $15 per million input tokens
    OUTPUT_PRICE_PER_M = 75.00  # $75 per million output tokens
    BATCH_DISCOUNT = 0.5  # 50% discount for batch API

    # Estimate token usage
    avg_input_tokens_per_request = 150  # Rough estimate
    avg_output_tokens_per_request = 20  # Rough estimate

    total_input_tokens = num_requests * avg_input_tokens_per_request
    total_output_tokens = num_requests * avg_output_tokens_per_request

    # Calculate costs with batch discount
    input_cost = (total_input_tokens / 1_000_000) * INPUT_PRICE_PER_M * BATCH_DISCOUNT
    output_cost = (total_output_tokens / 1_000_000) * OUTPUT_PRICE_PER_M * BATCH_DISCOUNT
    total_cost = input_cost + output_cost

    # Also calculate standard API cost for comparison
    standard_input_cost = (total_input_tokens / 1_000_000) * INPUT_PRICE_PER_M
    standard_output_cost = (total_output_tokens / 1_000_000) * OUTPUT_PRICE_PER_M
    standard_total = standard_input_cost + standard_output_cost

    print(f"\nCost Estimation (Claude Opus 4.1 - Confidence Only Mode):")
    print(f"  Confidence Requests: {num_requests}")
    print(f"  Estimated tokens: {total_input_tokens:,} input, {total_output_tokens:,} output")
    print(f"  Batch API cost: ${total_cost:.2f} (50% discount)")
    print(f"  Standard API cost: ${standard_total:.2f}")
    print(f"  Savings: ${standard_total - total_cost:.2f}")

    return total_cost, standard_total

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("CLAUDE OPUS 4.1 BATCH PERTURBATION ANALYSIS")
    print("CONFIDENCE-ONLY MODE (NO LOGPROBS)")
    print("="*60)

    # Parse arguments
    args = parse_arguments()

    # Check for API key
    if not ANTHROPIC_API_KEY:
        print("\nERROR: ANTHROPIC_API_KEY not found!")
        print("Please set ANTHROPIC_API_KEY in your config.py or .env file")
        return

    # Set configuration based on arguments
    output_file = args.output
    test_mode = args.test  # Test mode is now disabled by default, enabled with --test
    test_size = args.test_size if test_mode else None

    if test_mode:
        print(f"\n*** TEST MODE ENABLED ***")
        print(f"Processing only {test_size} perturbations")
        output_file = output_file.replace(".xlsx", "_test.xlsx")
        print(f"Output will be saved to: {output_file}")
    else:
        print(f"\n*** FULL PROCESSING MODE ***")
        print(f"Processing ALL perturbations")
        print(f"Output will be saved to: {output_file}")

    # Load perturbations
    if test_mode:
        # For test mode, limit the number of prompts loaded
        prompt_limit = min(10, test_size)  # Load at most 10 prompts for testing
        prompt_parts_list, rephrasings_list = load_perturbations(PERTURBATIONS_FILE, limit=prompt_limit)
    else:
        prompt_parts_list, rephrasings_list = load_perturbations(PERTURBATIONS_FILE)

    # Load existing results if skipping
    processed_set = set()
    if args.skip_existing:
        processed_set = load_existing_results(output_file)

    # Prepare batch requests
    batch_requests, request_id_map = prepare_batch_requests(
        prompt_parts_list,
        rephrasings_list,
        processed_set,
        test_mode=test_mode,
        test_size=test_size
    )

    if not batch_requests:
        print("\nNo new perturbations to process!")
        return

    # Estimate cost
    estimate_cost(len(batch_requests))

    # Confirm before proceeding
    if not test_mode and not args.yes:
        response = input("\nProceed with batch processing? (yes/no): ")
        if response.lower() != 'yes':
            print("Processing cancelled.")
            return
    else:
        if args.yes:
            print("\nProceeding with batch processing (auto-confirm enabled)...")
        else:
            print("\nProceeding with test batch processing...")

    # Create batch file
    batch_file = create_batch_file(batch_requests, BATCH_RESULTS_DIR)

    # Submit batch
    try:
        batch_id = submit_batch(batch_file)
    except Exception as e:
        print(f"Error submitting batch: {e}")
        return

    # Wait for completion
    start_time = time.time()
    batch_result = wait_for_batch_completion(batch_id)

    if batch_result["status"] != "success":
        print(f"Batch processing failed: {batch_result['message']}")
        return

    elapsed_time = time.time() - start_time
    print(f"Batch completed in {elapsed_time/60:.1f} minutes")

    # Process results
    results = process_batch_results(batch_result["results"], request_id_map)

    # Save results
    save_results(results, output_file, append=args.skip_existing)

    print(f"\n" + "="*60)
    print(f"PROCESSING COMPLETE")
    print(f"Total results: {len(results)}")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Results saved to: {output_file}")
    print("="*60)

if __name__ == "__main__":
    main()