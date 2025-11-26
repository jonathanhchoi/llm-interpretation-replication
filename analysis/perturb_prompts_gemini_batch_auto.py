#!/usr/bin/env python3
"""
Gemini 2.5 Pro Perturbation Analysis - Parallel Processing Version
Uses concurrent requests with rate limiting to efficiently process perturbations.
Much faster than sequential processing while respecting API limits.
"""

import os
import time
import json
import pandas as pd
import numpy as np
import re
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import google.generativeai as genai
from config import GEMINI_API_KEY

# Configuration
PERTURBATIONS_FILE = "../data/perturbations.json"
OUTPUT_EXCEL = "../results/gemini_batch_perturbation_results.xlsx"
CHECKPOINT_FILE = "../results/gemini_checkpoint.xlsx"

# Model configuration
MODEL_NAME = "gemini-2.5-pro"

# Processing configuration using Batch API quotas
# Batch API has higher limits: 2000 RPM, 4M TPM, 50,000 RPD
MAX_WORKERS = 50  # Concurrent threads
REQUESTS_PER_MINUTE = 1800  # Target 1800 RPM (safety buffer from 2000 RPM)
START_FROM_PROMPT = 2  # Start from prompt 3 (affiliates question)

# Rate limiting
rate_limiter_lock = Lock()
request_times = []


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Gemini Batch API Perturbation Analysis')
    parser.add_argument('--prepare-only', action='store_true',
                       help='Only prepare batch file without submitting')
    parser.add_argument('--batch-name', type=str,
                       help='Check status or retrieve results for batch with this name')
    parser.add_argument('--process-results', type=str,
                       help='Process results from specified batch job file')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process only first 10 perturbations')
    parser.add_argument('--start-from', type=int, default=2,
                       help='Start from prompt index (0-based, default: 2 for affiliates)')
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


def prepare_batch_requests(prompt_parts_list, rephrasings_list, output_file, test_mode=False, start_from=0):
    """Prepare batch requests in JSONL format."""
    requests_list = []
    metadata = {}

    # Determine which prompts to process
    end_idx = len(prompt_parts_list)
    total_count = 0

    for prompt_idx in range(start_from, end_idx):
        prompt_parts = prompt_parts_list[prompt_idx]
        rephrasings = rephrasings_list[prompt_idx]
        orig_main, orig_format, target_tokens, confidence_format = prompt_parts

        for rephrase_idx, rephrased_main in enumerate(rephrasings):
            if test_mode and total_count >= 10:
                break

            # Binary format request
            binary_prompt = f"{rephrased_main} {orig_format}"
            binary_id = f"binary_{prompt_idx}_{rephrase_idx}"

            binary_request = {
                "custom_id": binary_id,
                "method": "POST",
                "url": f"/v1beta/models/{MODEL_NAME}:generateContent",
                "body": {
                    "contents": [{
                        "parts": [{"text": binary_prompt}],
                        "role": "user"
                    }],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 500,
                        "responseLogprobs": True,
                        "logprobs": 19
                    }
                }
            }
            requests_list.append(binary_request)

            # Confidence format request
            confidence_prompt = f"{rephrased_main} {confidence_format}"
            confidence_id = f"confidence_{prompt_idx}_{rephrase_idx}"

            confidence_request = {
                "custom_id": confidence_id,
                "method": "POST",
                "url": f"/v1beta/models/{MODEL_NAME}:generateContent",
                "body": {
                    "contents": [{
                        "parts": [{"text": confidence_prompt}],
                        "role": "user"
                    }],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 500,
                        "responseLogprobs": True,
                        "logprobs": 19
                    }
                }
            }
            requests_list.append(confidence_request)

            # Store metadata
            metadata[binary_id] = {
                "type": "binary",
                "prompt_idx": prompt_idx,
                "rephrase_idx": rephrase_idx,
                "original_main": orig_main,
                "rephrased_main": rephrased_main,
                "target_tokens": list(target_tokens)
            }

            metadata[confidence_id] = {
                "type": "confidence",
                "prompt_idx": prompt_idx,
                "rephrase_idx": rephrase_idx,
                "original_main": orig_main,
                "rephrased_main": rephrased_main
            }

            total_count += 1

        if test_mode and total_count >= 10:
            break

    # Write requests to JSONL
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for request in requests_list:
            f.write(json.dumps(request) + '\n')

    # Save metadata
    metadata_file = output_file.replace('.jsonl', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nPrepared {len(requests_list)} batch requests ({total_count} perturbations)")
    print(f"Saved to: {output_file}")
    print(f"Metadata: {metadata_file}")

    return len(requests_list), total_count


def upload_batch_file(file_path):
    """Upload batch requests file to Gemini."""
    print(f"\nUploading batch file: {file_path}")

    url = f"{BATCH_API_BASE}/files"
    headers = {"X-Goog-Api-Key": GEMINI_API_KEY}

    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'application/jsonl')}
        response = requests.post(url, headers=headers, files=files)

    print(f"Upload response status: {response.status_code}")
    print(f"Response content: {response.text[:500]}")

    if response.status_code in [200, 201]:
        try:
            file_info = response.json()
            print(f"Full response JSON: {json.dumps(file_info, indent=2)[:500]}")

            # Try different possible field names
            file_uri = (file_info.get('file', {}).get('uri') or
                       file_info.get('uri') or
                       file_info.get('name') or
                       file_info.get('file', {}).get('name'))

            print(f"✓ File uploaded successfully")
            print(f"  File URI: {file_uri}")
            return file_uri
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None
    else:
        print(f"✗ Upload failed: {response.status_code}")
        print(f"  Response: {response.text}")
        return None


def submit_batch_job(file_uri):
    """Submit a batch prediction job."""
    print(f"\nSubmitting batch job...")

    url = f"{BATCH_API_BASE}/batchPredictionJobs"
    headers = {
        "X-Goog-Api-Key": GEMINI_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "inputFileUri": file_uri,
        "model": f"models/{MODEL_NAME}",
        "config": {
            "temperature": 0.0,
            "maxOutputTokens": 500
        }
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code in [200, 201]:
        job_info = response.json()
        job_name = job_info.get('name')
        print(f"✓ Batch job submitted successfully")
        print(f"  Job name: {job_name}")
        return job_name
    else:
        print(f"✗ Submission failed: {response.status_code}")
        print(f"  Response: {response.text}")
        return None


def check_batch_status(job_name):
    """Check the status of a batch job."""
    url = f"{BATCH_API_BASE}/{job_name}"
    headers = {"X-Goog-Api-Key": GEMINI_API_KEY}

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        job_info = response.json()
        state = job_info.get('state', 'UNKNOWN')
        return state, job_info
    else:
        print(f"✗ Status check failed: {response.status_code}")
        return None, None


def wait_for_completion(job_name, poll_interval=60, max_wait=86400):
    """Wait for batch job to complete."""
    print(f"\nWaiting for batch job to complete...")
    print(f"Job: {job_name}")
    print(f"Polling every {poll_interval} seconds")

    start_time = time.time()
    last_state = None

    while True:
        elapsed = time.time() - start_time

        if elapsed > max_wait:
            print(f"\n✗ Max wait time ({max_wait}s) exceeded")
            return False

        state, job_info = check_batch_status(job_name)

        if state != last_state:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status: {state}")
            last_state = state

        if state == "JOB_STATE_SUCCEEDED":
            output_uri = job_info.get('outputFileUri')
            print(f"\n✓ Batch job completed successfully!")
            print(f"  Output URI: {output_uri}")
            return output_uri

        elif state == "JOB_STATE_FAILED":
            error = job_info.get('error', 'Unknown error')
            print(f"\n✗ Batch job failed: {error}")
            return False

        elif state in ["JOB_STATE_PENDING", "JOB_STATE_RUNNING"]:
            # Still processing
            print(".", end="", flush=True)
            time.sleep(poll_interval)

        else:
            print(f"\n✗ Unknown state: {state}")
            return False


def download_results(output_uri, local_path):
    """Download batch results."""
    print(f"\nDownloading results...")

    # Extract file ID from URI
    if '/files/' in output_uri:
        file_id = output_uri.split('/files/')[-1]
        url = f"{BATCH_API_BASE}/files/{file_id}/content"
    else:
        url = output_uri

    headers = {"X-Goog-Api-Key": GEMINI_API_KEY}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"✓ Results downloaded to: {local_path}")
        return True
    else:
        print(f"✗ Download failed: {response.status_code}")
        print(f"  Response: {response.text}")
        return False


def extract_token_probabilities(logprobs_result, target_tokens):
    """Extract probabilities for target tokens from logprobs."""
    token_1_prob = 0.0
    token_2_prob = 0.0

    if not logprobs_result:
        return token_1_prob, token_2_prob

    try:
        # Get first token position
        if 'topCandidates' in logprobs_result and logprobs_result['topCandidates']:
            first_position = logprobs_result['topCandidates'][0]

            if 'candidates' in first_position:
                for candidate in first_position['candidates']:
                    token = candidate.get('token', '')
                    log_prob = candidate.get('logProbability', -999)

                    if token == target_tokens[0]:
                        token_1_prob = np.exp(log_prob)
                    elif token == target_tokens[1]:
                        token_2_prob = np.exp(log_prob)

    except Exception as e:
        print(f"    Warning: Error extracting token probabilities: {e}")

    return token_1_prob, token_2_prob


def extract_confidence_value(text):
    """Extract numerical confidence value from response."""
    match = re.search(r'\b(\d+)\b', text)
    if match:
        return int(match.group(1))
    return None


def calculate_weighted_confidence(logprobs_result):
    """Calculate weighted confidence from logprobs."""
    if not logprobs_result:
        return None

    try:
        if 'topCandidates' not in logprobs_result:
            return None

        positions = logprobs_result['topCandidates']
        if len(positions) < 1:
            return None

        # Build probability distribution over confidence values (0-100)
        value_probs = {}

        # Get first position candidates
        first_pos = positions[0] if len(positions) > 0 else None
        second_pos = positions[1] if len(positions) > 1 else None
        third_pos = positions[2] if len(positions) > 2 else None

        if not first_pos or 'candidates' not in first_pos:
            return None

        for first_cand in first_pos['candidates'][:19]:
            first_token = first_cand.get('token', '').strip()
            first_log_prob = first_cand.get('logProbability', -999)
            first_prob = np.exp(first_log_prob)

            if first_token.isdigit():
                if len(first_token) == 1:
                    # Single digit, check for continuation
                    first_digit = int(first_token)

                    if second_pos and 'candidates' in second_pos and 1 <= first_digit <= 9:
                        # Check for two-digit numbers
                        for second_cand in second_pos['candidates'][:19]:
                            second_token = second_cand.get('token', '').strip()
                            second_log_prob = second_cand.get('logProbability', -999)
                            second_prob = np.exp(second_log_prob)

                            if second_token.isdigit() and len(second_token) == 1:
                                second_digit = int(second_token)
                                value = first_digit * 10 + second_digit

                                if value == 10 and third_pos and 'candidates' in third_pos:
                                    # Check for 100
                                    for third_cand in third_pos['candidates'][:19]:
                                        third_token = third_cand.get('token', '').strip()
                                        third_log_prob = third_cand.get('logProbability', -999)
                                        third_prob = np.exp(third_log_prob)

                                        if third_token == '0':
                                            combined_prob = first_prob * second_prob * third_prob
                                            value_probs[100] = value_probs.get(100, 0) + combined_prob
                                else:
                                    combined_prob = first_prob * second_prob
                                    value_probs[value] = value_probs.get(value, 0) + combined_prob
                    else:
                        # Single digit value
                        value_probs[first_digit] = value_probs.get(first_digit, 0) + first_prob

                else:
                    # Multi-digit token
                    value = int(first_token)
                    if 0 <= value <= 100:
                        value_probs[value] = value_probs.get(value, 0) + first_prob

        # Calculate weighted average
        if value_probs:
            total_prob = sum(value_probs.values())
            if total_prob > 0:
                weighted_avg = sum(v * p for v, p in value_probs.items()) / total_prob
                return weighted_avg

    except Exception as e:
        print(f"    Warning: Error calculating weighted confidence: {e}")

    return None


def process_batch_results(results_file, metadata_file, prompt_parts_list, output_excel):
    """Process batch results and save to Excel."""
    print(f"\nProcessing batch results...")

    # Load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Load results
    binary_results = {}
    confidence_results = {}

    with open(results_file, 'r') as f:
        for line in f:
            try:
                result = json.loads(line)
                custom_id = result.get('custom_id')

                if custom_id:
                    if custom_id.startswith('binary_'):
                        binary_results[custom_id] = result.get('response', {})
                    elif custom_id.startswith('confidence_'):
                        confidence_results[custom_id] = result.get('response', {})
            except:
                continue

    print(f"Found {len(binary_results)} binary and {len(confidence_results)} confidence responses")

    # Process paired results
    final_results = []

    for binary_id, binary_response in binary_results.items():
        # Get corresponding confidence result
        parts = binary_id.split('_')
        prompt_idx = int(parts[1])
        rephrase_idx = int(parts[2])
        confidence_id = f"confidence_{prompt_idx}_{rephrase_idx}"

        if confidence_id not in confidence_results:
            continue

        confidence_response = confidence_results[confidence_id]

        # Get metadata
        meta = metadata.get(binary_id, {})
        orig_main = meta.get('original_main', '')
        rephrased_main = meta.get('rephrased_main', '')
        target_tokens = tuple(meta.get('target_tokens', []))

        # Get prompt parts
        prompt_parts = prompt_parts_list[prompt_idx]
        orig_format = prompt_parts[1]
        confidence_format = prompt_parts[3]

        # Extract text and logprobs
        binary_text = ''
        binary_logprobs = None
        if 'candidates' in binary_response and binary_response['candidates']:
            cand = binary_response['candidates'][0]
            if 'content' in cand and 'parts' in cand['content']:
                binary_text = cand['content']['parts'][0].get('text', '')
            binary_logprobs = cand.get('logprobsResult')

        confidence_text = ''
        confidence_logprobs = None
        if 'candidates' in confidence_response and confidence_response['candidates']:
            cand = confidence_response['candidates'][0]
            if 'content' in cand and 'parts' in cand['content']:
                confidence_text = cand['content']['parts'][0].get('text', '')
            confidence_logprobs = cand.get('logprobsResult')

        # Calculate metrics
        token_1_prob, token_2_prob = extract_token_probabilities(binary_logprobs, target_tokens)
        odds_ratio = token_1_prob / token_2_prob if token_2_prob > 0 else float('inf')

        confidence_value = extract_confidence_value(confidence_text)
        weighted_confidence = calculate_weighted_confidence(confidence_logprobs)

        # Store result
        final_results.append({
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
        })

    # Save to Excel
    if final_results:
        df = pd.DataFrame(final_results)
        os.makedirs(os.path.dirname(output_excel), exist_ok=True)
        df.to_excel(output_excel, index=False)
        print(f"\n✓ Saved {len(final_results)} results to {output_excel}")

    return len(final_results)


def main():
    """Main execution."""
    args = parse_arguments()

    print("\n" + "="*70)
    print("GEMINI 2.5 PRO - AUTOMATED BATCH API PERTURBATION ANALYSIS")
    print("="*70)

    if not GEMINI_API_KEY:
        print("\n✗ ERROR: GEMINI_API_KEY not found!")
        return

    # Load perturbations
    prompt_parts_list, rephrasings_list = load_perturbations(PERTURBATIONS_FILE)

    # Handle process-results mode
    if args.process_results:
        print(f"\nProcessing existing results from: {args.process_results}")
        processed = process_batch_results(
            args.process_results,
            BATCH_METADATA_FILE,
            prompt_parts_list,
            OUTPUT_EXCEL
        )
        print(f"\n✓ Processing complete: {processed} perturbations")
        return

    # Handle batch status check
    if args.batch_name:
        print(f"\nChecking batch: {args.batch_name}")
        state, job_info = check_batch_status(args.batch_name)

        if state == "JOB_STATE_SUCCEEDED":
            output_uri = job_info.get('outputFileUri')
            results_file = f"../data/batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

            if download_results(output_uri, results_file):
                processed = process_batch_results(
                    results_file,
                    BATCH_METADATA_FILE,
                    prompt_parts_list,
                    OUTPUT_EXCEL
                )
                print(f"\n✓ Complete: {processed} perturbations processed")
        return

    # Prepare batch requests
    print(f"\nStarting from prompt {args.start_from + 1} (0-indexed: {args.start_from})")
    if args.test:
        print("*** TEST MODE: Processing only 10 perturbations ***")

    num_requests, num_perturbations = prepare_batch_requests(
        prompt_parts_list,
        rephrasings_list,
        BATCH_REQUESTS_FILE,
        test_mode=args.test,
        start_from=args.start_from
    )

    # Estimate cost (batch API is 50% cheaper)
    estimated_cost = (num_requests * 100 / 1_000_000) * 0.0375
    print(f"\nEstimated cost: ${estimated_cost:.2f} (batch discount applied)")
    print(f"Expected turnaround: Within 24 hours")

    if args.prepare_only:
        print("\n✓ Batch file prepared (use without --prepare-only to submit)")
        return

    # Confirm submission
    response = input("\nSubmit batch job? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return

    # Upload and submit
    file_uri = upload_batch_file(BATCH_REQUESTS_FILE)
    if not file_uri:
        print("\n✗ Failed to upload batch file")
        return

    job_name = submit_batch_job(file_uri)
    if not job_name:
        print("\n✗ Failed to submit batch job")
        return

    # Wait for completion
    print(f"\nBatch job submitted: {job_name}")
    response = input("Wait for completion? (yes/no): ")

    if response.lower() == 'yes':
        output_uri = wait_for_completion(job_name, POLL_INTERVAL, MAX_WAIT_TIME)

        if output_uri:
            results_file = f"../data/batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

            if download_results(output_uri, results_file):
                processed = process_batch_results(
                    results_file,
                    BATCH_METADATA_FILE,
                    prompt_parts_list,
                    OUTPUT_EXCEL
                )
                print(f"\n✓ Complete: {processed} perturbations processed")
                print(f"✓ Results: {OUTPUT_EXCEL}")
    else:
        print(f"\nCheck status later with:")
        print(f"  python {__file__} --batch-name {job_name}")


if __name__ == "__main__":
    main()