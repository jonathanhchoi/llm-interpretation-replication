print("Gemini 2.5 Pro Perturbation Analysis - Batch API Version 2.0.0")

import os
import time
import json
import pandas as pd
import numpy as np
import re
import argparse
import requests
from datetime import datetime

# Try to import Google AI SDK (new SDK with batch support)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    print("Warning: Google Gen AI SDK not installed")
    print("Install with: pip install google-genai")
    GENAI_AVAILABLE = False
    genai = None
    types = None

try:
    from config import GEMINI_API_KEY
except ImportError:
    print("Warning: Could not import GEMINI_API_KEY from config")
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

# Configuration - handle both running from analysis/ dir and parent dir
if os.path.basename(os.getcwd()) == 'analysis':
    # Running from within analysis directory
    PERTURBATIONS_FILE = "../data/perturbations.json"
    OUTPUT_EXCEL = "../results/gemini_batch_perturbation_results.xlsx"
    BATCH_INPUT_FILE = "../data/gemini_batch_requests.jsonl"
    BATCH_OUTPUT_FILE = "../data/gemini_batch_responses.jsonl"
    BATCH_ID_FILE = "../data/gemini_batch_id.txt"
else:
    # Running from parent directory
    PERTURBATIONS_FILE = "data/perturbations.json"
    OUTPUT_EXCEL = "results/gemini_batch_perturbation_results.xlsx"
    BATCH_INPUT_FILE = "data/gemini_batch_requests.jsonl"
    BATCH_OUTPUT_FILE = "data/gemini_batch_responses.jsonl"
    BATCH_ID_FILE = "data/gemini_batch_id.txt"

# Model configuration
MODEL_NAME = "gemini-2.5-pro"  # Gemini 2.5 Pro model with logprobs support

# Processing configuration
MAX_BATCH_SIZE = 10000  # Maximum requests per batch job
TEST_MODE = True  # Set via command line
TEST_SIZE = 100  # Number of prompts to test in test mode
START_FROM_PROMPT = 0  # 0-indexed starting prompt
DEBUG_MODE = False  # Show detailed debugging
POLL_INTERVAL = 30  # Seconds between batch status checks

# Initialize Gemini client if available
if GENAI_AVAILABLE:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gemini Batch API Perturbation Analysis')
    parser.add_argument('--no-test', dest='test', action='store_false', default=True,
                       help='Disable test mode: process all prompts instead of just 100')
    parser.add_argument('--test-size', type=int, default=100,
                       help='Number of prompts to process in test mode (default: 100)')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Start from prompt index (0-based, default: 0)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--prepare-only', action='store_true',
                       help='Only prepare batch file without submitting')
    parser.add_argument('--process-results', action='store_true',
                       help='Only process existing results file')
    parser.add_argument('--batch-id', type=str,
                       help='Batch ID to check status or retrieve results')
    parser.add_argument('--use-cached-results', action='store_true', default=True,
                       help='Use cached batch results if available (default: True)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Do not use cached results, force new batch submission')
    return parser.parse_args()

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

def prepare_batch_requests(prompt_parts_list, rephrasings_list, output_file, test_mode=False, test_size=100, start_from=0):
    """Prepare batch requests in JSONL format for Gemini Batch API."""
    requests = []
    metadata_store = {}  # Store metadata separately
    request_id = 0

    # Limit prompts if in test mode
    end_idx = len(prompt_parts_list)
    if test_mode:
        # Calculate how many prompts we need to get test_size perturbations
        total_count = 0
        prompts_needed = 0
        for i in range(start_from, len(prompt_parts_list)):
            total_count += len(rephrasings_list[i])
            prompts_needed = i + 1
            if total_count >= test_size:
                break
        end_idx = min(prompts_needed, len(prompt_parts_list))
        print(f"Test mode: Processing prompts {start_from} to {end_idx-1} to get ~{test_size} perturbations")

    perturbation_count = 0
    for prompt_idx in range(start_from, end_idx):
        prompt_parts = prompt_parts_list[prompt_idx]
        rephrasings = rephrasings_list[prompt_idx]
        orig_main, orig_format, target_tokens, confidence_format = prompt_parts

        for rephrase_idx, rephrased_main in enumerate(rephrasings):
            if test_mode and perturbation_count >= test_size:
                break

            # Create binary format request (for Yes/No response)
            binary_prompt = f"{rephrased_main} {orig_format}"
            binary_id = f"binary_{prompt_idx}_{rephrase_idx}"

            # Store metadata separately (not sent to API)
            metadata_store[binary_id] = {
                "type": "binary",
                "prompt_idx": prompt_idx,
                "rephrase_idx": rephrase_idx,
                "original_main": orig_main,
                "rephrased_main": rephrased_main,
                "target_tokens": list(target_tokens)
            }

            # Format for Gemini Batch API
            binary_request = {
                "custom_id": binary_id,
                "generateContentRequest": {
                    "model": f"models/{MODEL_NAME}",
                    "contents": [
                        {
                            "parts": [{"text": binary_prompt}],
                            "role": "user"
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 500,
                        "responseLogprobs": True,
                        "logprobs": 19
                    }
                }
            }
            requests.append(binary_request)
            request_id += 1

            # Create confidence format request
            confidence_prompt = f"{rephrased_main} {confidence_format}"
            confidence_id = f"confidence_{prompt_idx}_{rephrase_idx}"

            # Store metadata
            metadata_store[confidence_id] = {
                "type": "confidence",
                "prompt_idx": prompt_idx,
                "rephrase_idx": rephrase_idx,
                "original_main": orig_main,
                "rephrased_main": rephrased_main,
                "confidence_format": confidence_format
            }

            confidence_request = {
                "custom_id": confidence_id,
                "generateContentRequest": {
                    "model": f"models/{MODEL_NAME}",
                    "contents": [
                        {
                            "parts": [{"text": confidence_prompt}],
                            "role": "user"
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 500,
                        "responseLogprobs": True,
                        "logprobs": 19
                    }
                }
            }
            requests.append(confidence_request)
            request_id += 1
            perturbation_count += 1

        if test_mode and perturbation_count >= test_size:
            break

    # Write requests to JSONL file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')

    # Save metadata separately
    metadata_file = output_file.replace('.jsonl', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata_store, f, indent=2)

    print(f"\nPrepared {len(requests)} batch requests ({perturbation_count} perturbations)")
    print(f"Requests saved to: {output_file}")
    print(f"Metadata saved to: {metadata_file}")
    print(f"\nEach request includes:")
    print(f"  - Model: {MODEL_NAME}")
    print(f"  - Temperature: 0.0")
    print(f"  - Logprobs: Enabled (top 19)")
    print(f"  - Max tokens: 500")

    return len(requests)

def submit_batch_job(input_file):
    """Submit a batch job to Gemini API using the SDK."""
    print(f"\nSubmitting batch job from {input_file}...")

    try:
        if not GEMINI_API_KEY:
            print("\nERROR: GEMINI_API_KEY not found!")
            return None

        if not GENAI_AVAILABLE or client is None:
            print("\nERROR: Google Gen AI SDK not available!")
            print("Please install it with: pip install google-genai")
            return None

        # Read the JSONL file and convert to inline requests
        batch_requests = []
        with open(input_file, 'r') as f:
            for line in f:
                req = json.loads(line)
                # Convert from JSONL format to inline format expected by SDK
                batch_requests.append({
                    "contents": req["generateContentRequest"]["contents"],
                })

        print(f"Loaded {len(batch_requests)} requests from file")

        print("\n" + "="*60)
        print("BATCH API SUBMISSION")
        print("="*60)

        # Create batch job using the SDK
        print(f"\nSubmitting batch with {len(batch_requests)} requests...")

        # Use the batches API with new SDK
        batch_job = client.batches.create(
            model=f"models/{MODEL_NAME}",
            src=batch_requests,
            config={
                'display_name': f"perturbation-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            }
        )

        batch_name = batch_job.name
        print(f"\n✓ Batch job submitted successfully!")
        print(f"  Batch name: {batch_name}")
        print(f"  Status: {batch_job.state.name}")

        return batch_name

    except Exception as e:
        print(f"Error submitting batch job: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_batch_status(batch_name):
    """Check the status of a batch job."""
    try:
        print(f"\nChecking batch status for: {batch_name}")

        # Get batch job status using SDK
        batch_job = client.batches.get(name=batch_name)

        print(f"\nBatch Status:")
        print(f"  Name: {batch_job.name}")
        print(f"  State: {batch_job.state.name}")
        print(f"  Model: {batch_job.model}")

        if hasattr(batch_job, 'request_count'):
            print(f"  Total requests: {batch_job.request_count}")
        if hasattr(batch_job, 'completed_count'):
            print(f"  Completed: {batch_job.completed_count}")

        return batch_job.state.name

    except Exception as e:
        print(f"Error checking batch status: {e}")
        import traceback
        traceback.print_exc()
        return None

def wait_for_batch_completion(batch_name, poll_interval=30):
    """Wait for a batch job to complete."""
    print(f"\nWaiting for batch completion: {batch_name}")
    print(f"Polling every {poll_interval} seconds...")
    print("Typical turnaround time: Within 24 hours")

    start_time = time.time()

    completed_states = {
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
    }

    while True:
        state = check_batch_status(batch_name)

        if state == "JOB_STATE_SUCCEEDED":
            print("\n✓ Batch job completed successfully!")
            return True
        elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
            print(f"\n✗ Batch job ended with state: {state}")
            return False
        elif state in ["JOB_STATE_PENDING", "JOB_STATE_RUNNING"]:
            elapsed = time.time() - start_time
            print(f"  Still processing... (elapsed: {elapsed/60:.1f} minutes)", end="\r")
            time.sleep(poll_interval)
        else:
            print(f"\n? Unknown state: {state}")
            return False

def save_batch_id(batch_name):
    """Save batch ID to file for later retrieval."""
    try:
        os.makedirs(os.path.dirname(BATCH_ID_FILE), exist_ok=True)
        with open(BATCH_ID_FILE, 'w') as f:
            f.write(batch_name)
        print(f"Saved batch ID to: {BATCH_ID_FILE}")
        return True
    except Exception as e:
        print(f"Warning: Could not save batch ID: {e}")
        return False

def load_batch_id():
    """Load saved batch ID from file."""
    if os.path.exists(BATCH_ID_FILE):
        try:
            with open(BATCH_ID_FILE, 'r') as f:
                batch_id = f.read().strip()
                if batch_id:
                    return batch_id
        except Exception as e:
            print(f"Warning: Could not read batch ID file: {e}")
    return None

def clear_batch_id():
    """Clear saved batch ID after successful completion."""
    try:
        if os.path.exists(BATCH_ID_FILE):
            os.remove(BATCH_ID_FILE)
            return True
    except Exception as e:
        print(f"Warning: Could not clear batch ID: {e}")
    return False

def retrieve_batch_results(batch_name, output_file):
    """Retrieve results from a completed batch job."""
    print(f"\nRetrieving batch results for: {batch_name}")

    try:
        # Get batch job
        batch_job = client.batches.get(name=batch_name)

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            print(f"Batch not ready yet. Current state: {batch_job.state.name}")
            return False

        # Get results from the batch job
        print("Fetching results from batch job...")

        # Write results to JSONL file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Load the request metadata to get custom_ids in order
        metadata_file = BATCH_INPUT_FILE.replace('.jsonl', '_metadata.json')
        with open(BATCH_INPUT_FILE, 'r') as f:
            request_ids = [json.loads(line)['custom_id'] for line in f]

        count = 0
        with open(output_file, 'w') as f:
            # Access inlined responses
            if batch_job.dest and batch_job.dest.inlined_responses:
                for idx, response in enumerate(batch_job.dest.inlined_responses):
                    # Get custom_id from the original request (responses are in same order)
                    custom_id = request_ids[idx] if idx < len(request_ids) else f"result_{idx}"

                    # Extract the actual text and logprobs from the response object
                    response_text = ""
                    logprobs_result_dict = None

                    # The response object should have candidates attribute
                    if hasattr(response, 'candidates') and response.candidates:
                        first_candidate = response.candidates[0]

                        # Extract text from the candidate's content
                        if hasattr(first_candidate, 'content') and hasattr(first_candidate.content, 'parts'):
                            if first_candidate.content.parts:
                                response_text = first_candidate.content.parts[0].text

                        # Extract and convert logprobs_result to dict if available
                        if hasattr(first_candidate, 'logprobs_result') and first_candidate.logprobs_result:
                            lr = first_candidate.logprobs_result
                            logprobs_result_dict = {
                                "chosen_candidates": [
                                    {"token": c.token, "log_probability": c.log_probability}
                                    for c in lr.chosen_candidates
                                ] if hasattr(lr, 'chosen_candidates') and lr.chosen_candidates else [],
                                "top_candidates": [
                                    {
                                        "candidates": [
                                            {"token": cand.token, "log_probability": cand.log_probability}
                                            for cand in pos.candidates
                                        ]
                                    }
                                    for pos in lr.top_candidates
                                ] if hasattr(lr, 'top_candidates') and lr.top_candidates else []
                            }

                    # Convert response to JSONL format compatible with processing function
                    result_json = {
                        "custom_id": custom_id,
                        "response": {
                            "candidates": [{
                                "content": {
                                    "parts": [{"text": response_text}]
                                },
                                "logprobs_result": logprobs_result_dict
                            }]
                        }
                    }
                    f.write(json.dumps(result_json) + '\n')
                    count += 1

        print(f"\n✓ Retrieved {count} responses")
        print(f"  Saved to: {output_file}")
        return True

    except Exception as e:
        print(f"Error retrieving results: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_token_probabilities(logprobs_data, target_tokens):
    """Extract probabilities for specific target tokens from logprobs."""
    token_1_prob = 0.0
    token_2_prob = 0.0

    if not logprobs_data:
        return token_1_prob, token_2_prob

    try:
        # Debug: Show what we're looking for
        if DEBUG_MODE:
            print(f"      Looking for tokens: {target_tokens}")

        # Access logprobs_data as dict (after JSON serialization)
        if isinstance(logprobs_data, dict) and 'top_candidates' in logprobs_data and logprobs_data['top_candidates']:
            # Look at the first token position
            first_position = logprobs_data['top_candidates'][0]

            if DEBUG_MODE:
                print(f"      Top 5 tokens at position 0:")
                candidates_list = first_position.get('candidates', [])
                for i, candidate in enumerate(candidates_list[:5]):
                    token_text = candidate['token']
                    log_prob = candidate['log_probability']
                    prob = np.exp(log_prob)
                    print(f"        {i+1}. '{token_text}': logprob={log_prob:.4f}, prob={prob:.4f}")

            for candidate in first_position.get('candidates', []):
                token_text = candidate['token']
                log_prob = candidate['log_probability']

                if token_text == target_tokens[0]:
                    token_1_prob = np.exp(log_prob)
                elif token_text == target_tokens[1]:
                    token_2_prob = np.exp(log_prob)
    except Exception as e:
        print(f"      Error extracting token probabilities: {e}")
        import traceback
        traceback.print_exc()

    return token_1_prob, token_2_prob

def extract_confidence_value(response_text):
    """Extract numerical confidence value from response."""
    try:
        match = re.search(r'\b(\d+)\b', response_text)
        if match:
            return int(match.group(1))
    except:
        pass
    return None

def calculate_weighted_confidence(logprobs_data):
    """Calculate weighted confidence from logprobs for confidence responses."""
    if not logprobs_data:
        return None

    try:
        # Access logprobs_data as dict (after JSON serialization)
        if isinstance(logprobs_data, dict) and 'top_candidates' in logprobs_data and len(logprobs_data['top_candidates']) >= 1:
            top_candidates_list = logprobs_data['top_candidates']
            first_pos = top_candidates_list[0] if len(top_candidates_list) > 0 else None
            second_pos = top_candidates_list[1] if len(top_candidates_list) > 1 else None
            third_pos = top_candidates_list[2] if len(top_candidates_list) > 2 else None

            if DEBUG_MODE:
                print(f"      Calculating weighted confidence from token combinations:")
                print(f"        Positions available: {len(top_candidates_list)}")

            # Store probabilities for different number formations
            one_digit_probs = {}   # e.g., "5" alone
            two_digit_probs = {}   # e.g., "5" + "0" = 50
            three_digit_probs = {} # e.g., "1" + "0" + "0" = 100

            # Process all first position candidates
            for first_cand in first_pos.get('candidates', [])[:19]:
                first_token = first_cand['token'].strip()
                first_log_prob = first_cand['log_probability']
                first_prob = np.exp(first_log_prob)
                
                # Check if it's a single digit
                if first_token.isdigit() and len(first_token) == 1:
                    first_digit = int(first_token)
                    
                    # Track probability of this being a standalone single digit
                    one_digit_prob_for_this = first_prob
                    two_digit_prob_sum = 0.0
                    three_digit_prob_sum = 0.0

                    # Check for two-digit formations
                    if second_pos and 1 <= first_digit <= 9:
                        for second_cand in second_pos.get('candidates', [])[:19]:
                            second_token = second_cand['token'].strip()
                            second_log_prob = second_cand['log_probability']
                            second_prob = np.exp(second_log_prob)
                            
                            if second_token.isdigit() and len(second_token) == 1:
                                second_digit = int(second_token)
                                two_digit_value = first_digit * 10 + second_digit

                                # Check for three-digit formation (only 100 is valid)
                                if two_digit_value == 10 and third_pos:
                                    for third_cand in third_pos.get('candidates', [])[:19]:
                                        third_token = third_cand['token'].strip()
                                        third_log_prob = third_cand['log_probability']
                                        third_prob = np.exp(third_log_prob)
                                        
                                        if third_token == "0":
                                            # This forms 100
                                            combined_prob = first_prob * second_prob * third_prob
                                            if 100 not in three_digit_probs:
                                                three_digit_probs[100] = 0
                                            three_digit_probs[100] += combined_prob
                                            three_digit_prob_sum += combined_prob
                                
                                # Add two-digit number probability (minus what becomes 100)
                                if 10 <= two_digit_value <= 99:
                                    combined_prob = first_prob * second_prob
                                    
                                    # Subtract probability that continues to form 100
                                    if two_digit_value == 10 and third_pos:
                                        # Find probability of "0" in third position
                                        third_zero_prob = 0.0
                                        for third_cand in third_pos.get('candidates', [])[:19]:
                                            if third_cand['token'].strip() == "0":
                                                third_zero_prob = np.exp(third_cand['log_probability'])
                                                break
                                        combined_prob *= (1 - third_zero_prob)
                                    
                                    if two_digit_value not in two_digit_probs:
                                        two_digit_probs[two_digit_value] = 0
                                    two_digit_probs[two_digit_value] += combined_prob
                                    two_digit_prob_sum += combined_prob
                    
                    # Calculate remaining probability for single digit
                    # Subtract probability that continues to form longer numbers
                    if second_pos and 1 <= first_digit <= 9:
                        # Find total probability of second position being a digit
                        second_digit_prob = 0.0
                        for second_cand in second_pos.get('candidates', [])[:19]:
                            if second_cand['token'].strip().isdigit() and len(second_cand['token'].strip()) == 1:
                                second_digit_prob += np.exp(second_cand['log_probability'])
                        
                        # Single digit prob = first prob * (1 - prob of second being digit)
                        one_digit_prob_for_this *= (1 - second_digit_prob)
                    
                    # Add single digit probability
                    if 0 <= first_digit <= 9:
                        if first_digit not in one_digit_probs:
                            one_digit_probs[first_digit] = 0
                        one_digit_probs[first_digit] += one_digit_prob_for_this
                
                # Check if it's already a complete number token
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
                
                if DEBUG_MODE:
                    sorted_values = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
                    print(f"        Found {len(sorted_values)} confidence values:")
                    for val, prob in sorted_values[:10]:
                        print(f"          {val}: prob={prob:.6f}")
                    print(f"        Total probability mass (before normalization): {total_prob:.6f}")
                    print(f"        Weighted confidence: {weighted_avg:.2f}")
                
                return weighted_avg
        
    except Exception as e:
        print(f"      Error calculating weighted confidence: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def process_batch_results(batch_results_file, prompt_parts_list, rephrasings_list, output_excel):
    """Process the batch results and save to Excel."""
    print(f"\nProcessing batch results from {batch_results_file}...")

    # Read metadata file
    metadata_file = BATCH_INPUT_FILE.replace('.jsonl', '_metadata.json')
    if not os.path.exists(metadata_file):
        print(f"Warning: Metadata file not found: {metadata_file}")
        print("Will attempt to reconstruct from request IDs")
        input_metadata = {}
    else:
        with open(metadata_file, 'r') as f:
            input_metadata = json.load(f)
    
    # Read batch results
    results = []
    binary_responses = {}
    confidence_responses = {}

    with open(batch_results_file, 'r') as f:
        for line in f:
            try:
                result = json.loads(line)
                custom_id = result.get('custom_id')
                if not custom_id:
                    continue

                # Parse response
                response_data = result.get('response', {})
                if custom_id.startswith('binary_'):
                    binary_responses[custom_id] = response_data
                elif custom_id.startswith('confidence_'):
                    confidence_responses[custom_id] = response_data
            except json.JSONDecodeError:
                print(f"Error parsing result line: {line[:100]}...")
                continue

    print(f"Found {len(binary_responses)} binary responses and {len(confidence_responses)} confidence responses")

    # Process paired responses
    processed_count = 0
    for binary_id, binary_response in binary_responses.items():
        # Extract indices from ID
        parts = binary_id.split('_')
        if len(parts) >= 3:
            prompt_idx = int(parts[1])
            rephrase_idx = int(parts[2])

            # Find corresponding confidence response
            confidence_id = f"confidence_{prompt_idx}_{rephrase_idx}"
            if confidence_id not in confidence_responses:
                print(f"Warning: No confidence response for {binary_id}")
                continue

            confidence_response = confidence_responses[confidence_id]

            # Get metadata
            metadata = input_metadata.get(binary_id, {})
            orig_main = metadata.get('original_main', '')
            rephrased_main = metadata.get('rephrased_main', '')
            target_tokens = metadata.get('target_tokens', [])

            # Get prompt parts
            if prompt_idx < len(prompt_parts_list):
                prompt_parts = prompt_parts_list[prompt_idx]
                orig_format = prompt_parts[1]
                confidence_format = prompt_parts[3]
            else:
                orig_format = ''
                confidence_format = ''

            # Extract responses
            binary_text = ''
            binary_logprobs = None
            confidence_text = ''
            confidence_logprobs = None

            if isinstance(binary_response, dict):
                if 'candidates' in binary_response and binary_response['candidates']:
                    candidate = binary_response['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        binary_text = candidate['content']['parts'][0].get('text', '')
                    if 'logprobs_result' in candidate:
                        binary_logprobs = candidate['logprobs_result']

            if isinstance(confidence_response, dict):
                if 'candidates' in confidence_response and confidence_response['candidates']:
                    candidate = confidence_response['candidates'][0]
                    if 'content' in candidate and 'parts' in candidate['content']:
                        confidence_text = candidate['content']['parts'][0].get('text', '')
                    if 'logprobs_result' in candidate:
                        confidence_logprobs = candidate['logprobs_result']

            # Extract token probabilities
            token_1_prob, token_2_prob = extract_token_probabilities(binary_logprobs, target_tokens)
            odds_ratio = token_1_prob / token_2_prob if token_2_prob > 0 else float('inf')

            # Extract confidence values
            confidence_value = extract_confidence_value(confidence_text)
            weighted_confidence = calculate_weighted_confidence(confidence_logprobs)

            # Store result
            result = {
                "Model": MODEL_NAME,
                "Original Main Part": orig_main,
                "Response Format": orig_format,
                "Confidence Format": confidence_format,
                "Rephrased Main Part": rephrased_main,
                "Full Rephrased Prompt": f"{rephrased_main} {orig_format}",
                "Full Confidence Prompt": f"{rephrased_main} {confidence_format}",
                "Model Response": binary_text,
                "Model Confidence Response": confidence_text,
                "Log Probabilities": str(binary_logprobs) if binary_logprobs else "N/A",
                "Token_1_Prob": token_1_prob,
                "Token_2_Prob": token_2_prob,
                "Odds_Ratio": odds_ratio,
                "Confidence Value": confidence_value,
                "Weighted Confidence": weighted_confidence
            }

            results.append(result)
            processed_count += 1

            if processed_count % 100 == 0:
                print(f"Processed {processed_count} perturbations...")

    # Save results to Excel
    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_excel), exist_ok=True)
        df.to_excel(output_excel, index=False)
        print(f"\nSaved {len(results)} results to {output_excel}")

    return processed_count


def main():
    """Main execution function."""
    args = parse_arguments()

    # Update global settings from args
    global TEST_MODE, TEST_SIZE, START_FROM_PROMPT, DEBUG_MODE
    TEST_MODE = args.test
    TEST_SIZE = args.test_size
    START_FROM_PROMPT = args.start_from
    DEBUG_MODE = args.debug

    # Handle cache settings
    use_cache = args.use_cached_results and not args.no_cache

    print("\n" + "="*60)
    print("GEMINI 2.5 PRO BATCH API PERTURBATION ANALYSIS")
    print("="*60)

    if TEST_MODE:
        print(f"\n*** TEST MODE: Processing only {TEST_SIZE} perturbations ***")

    if START_FROM_PROMPT > 0:
        print(f"\nStarting from prompt {START_FROM_PROMPT + 1} (0-indexed: {START_FROM_PROMPT})")

    # Load perturbations (needed for all operations)
    prompt_parts_list, rephrasings_list = load_perturbations(PERTURBATIONS_FILE)

    # Handle process-results mode early (doesn't need API)
    if args.process_results:
        # Only process existing results
        if not os.path.exists(BATCH_OUTPUT_FILE):
            print(f"\nERROR: Results file not found: {BATCH_OUTPUT_FILE}")
            print("Please run a batch job first or check the file path.")
            return

        print("\nProcessing existing batch results...")
        processed = process_batch_results(BATCH_OUTPUT_FILE, prompt_parts_list, rephrasings_list, OUTPUT_EXCEL)
        print(f"\nProcessed {processed} perturbations")
        return

    # Check for API key and SDK (needed for all other operations)
    if True:
        if not GEMINI_API_KEY:
            print("\nERROR: GEMINI_API_KEY not found!")
            print("Please set GEMINI_API_KEY in your config.py or .env file")
            return

        if not GENAI_AVAILABLE or client is None:
            print("\nERROR: Google Gen AI SDK not installed!")
            print("Please install it with: pip install google-genai")
            print("\nNote: The old 'google-generativeai' package is deprecated.")
            print("You must use the new 'google-genai' SDK for batch API support.")
            return

    # Calculate total perturbations
    total_perturbations = sum(len(rephrasings) for rephrasings in rephrasings_list)
    print(f"\nTotal perturbations available: {total_perturbations}")

    # Check for saved batch ID
    saved_batch_id = load_batch_id() if use_cache and not args.batch_id and not args.prepare_only else None

    if saved_batch_id:
        print(f"\n{'='*60}")
        print("SAVED BATCH JOB DETECTED")
        print(f"{'='*60}")
        print(f"\nFound saved batch ID: {saved_batch_id}")
        print("Checking batch status and retrieving results if completed...")
        print(f"{'='*60}\n")

        # Check batch status
        state = check_batch_status(saved_batch_id)

        if state == "JOB_STATE_SUCCEEDED":
            print("\nBatch completed! Retrieving results...")
            if retrieve_batch_results(saved_batch_id, BATCH_OUTPUT_FILE):
                print("\nProcessing results...")
                processed = process_batch_results(BATCH_OUTPUT_FILE, prompt_parts_list, rephrasings_list, OUTPUT_EXCEL)
                print(f"\nProcessed {processed} perturbations")
                print(f"Results saved to: {OUTPUT_EXCEL}")
                # Clear batch ID after successful processing
                clear_batch_id()
                return
            else:
                print("\nFailed to retrieve results. You may need to check the batch manually.")
                return
        elif state in ["JOB_STATE_PENDING", "JOB_STATE_RUNNING"]:
            print(f"\nBatch is still running (State: {state})")
            print("Check again later or use --batch-id to wait for completion.")
            print(f"\nCommand: python {__file__} --batch-id {saved_batch_id}")
            return
        elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
            print(f"\nBatch job ended with state: {state}")
            print("Clearing saved batch ID. You may need to submit a new batch.")
            clear_batch_id()
            # Continue to allow new batch submission
        else:
            print(f"\nUnknown batch state: {state}")
            print("Continuing with normal flow...")
            # Continue to allow new batch submission

    if args.batch_id:
        # Check status or retrieve results for existing batch
        print(f"\nChecking batch: {args.batch_id}")
        state = check_batch_status(args.batch_id)

        if state == "JOB_STATE_SUCCEEDED":
            print("\nBatch completed! Retrieving results...")
            if retrieve_batch_results(args.batch_id, BATCH_OUTPUT_FILE):
                print("\nProcessing results...")
                processed = process_batch_results(BATCH_OUTPUT_FILE, prompt_parts_list, rephrasings_list, OUTPUT_EXCEL)
                print(f"\nProcessed {processed} perturbations")
                print(f"Results saved to: {OUTPUT_EXCEL}")
                # Clear batch ID if it matches the one we just processed
                saved_batch_id = load_batch_id()
                if saved_batch_id == args.batch_id:
                    clear_batch_id()
        elif state in ["JOB_STATE_PENDING", "JOB_STATE_RUNNING"]:
            print("\nBatch is still running. Check again later or wait for completion.")
            response = input("Wait for completion? (yes/no): ")
            if response.lower() == 'yes':
                if wait_for_batch_completion(args.batch_id, POLL_INTERVAL):
                    if retrieve_batch_results(args.batch_id, BATCH_OUTPUT_FILE):
                        print("\nProcessing results...")
                        processed = process_batch_results(BATCH_OUTPUT_FILE, prompt_parts_list, rephrasings_list, OUTPUT_EXCEL)
                        print(f"\nProcessed {processed} perturbations")
                        print(f"Results saved to: {OUTPUT_EXCEL}")
                        # Clear batch ID if it matches the one we just processed
                        saved_batch_id = load_batch_id()
                        if saved_batch_id == args.batch_id:
                            clear_batch_id()
        return

    # Prepare batch requests
    print("\nPreparing batch requests...")
    num_requests = prepare_batch_requests(
        prompt_parts_list,
        rephrasings_list,
        BATCH_INPUT_FILE,
        test_mode=TEST_MODE,
        test_size=TEST_SIZE,
        start_from=START_FROM_PROMPT
    )

    # Estimate cost for batch API
    # Batch API is 50% cheaper than standard API
    estimated_tokens = num_requests * 100  # Rough estimate per request
    estimated_cost = (estimated_tokens / 1_000_000) * 0.0375  # Half of standard price
    print(f"\nEstimated cost: ${estimated_cost:.2f} (batch pricing: 50% of standard)")
    print(f"Number of requests: {num_requests}")
    print(f"Expected turnaround: Within 24 hours")

    if args.prepare_only:
        print("\nBatch file prepared. Use --batch-id to submit and monitor.")
        return

    # Confirm before proceeding
    print("\n" + "="*60)
    response = input("Submit batch job? (yes/no): ")
    print("="*60)
    if response.lower() != 'yes':
        print("Batch submission cancelled.")
        return

    # Submit batch job
    batch_name = submit_batch_job(BATCH_INPUT_FILE)
    if not batch_name:
        print("\nBatch job submission failed.")
        return

    print(f"\nBatch job submitted successfully!")
    print(f"Batch name: {batch_name}")

    # Save batch ID for later retrieval
    save_batch_id(batch_name)

    print(f"\nYou can check status with: python {__file__} --batch-id {batch_name}")

    # Optionally wait for completion
    response = input("\nWait for batch to complete? (yes/no): ")
    if response.lower() == 'yes':
        if wait_for_batch_completion(batch_name, POLL_INTERVAL):
            if retrieve_batch_results(batch_name, BATCH_OUTPUT_FILE):
                print("\nProcessing results...")
                processed = process_batch_results(BATCH_OUTPUT_FILE, prompt_parts_list, rephrasings_list, OUTPUT_EXCEL)
                print(f"\nProcessed {processed} perturbations")
                print(f"\nResults saved to: {OUTPUT_EXCEL}")
                # Clear batch ID after successful completion
                clear_batch_id()

if __name__ == "__main__":
    main()