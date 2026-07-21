print("Version 4.8.0")

import os
import csv
import openai
import time
import random
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from anthropic._exceptions import OverloadedError, RateLimitError, APIError, APIStatusError
from config import ANTHROPIC_API_KEY, OPENAI_API_KEY
import pandas as pd
import numpy as np
import re
import json
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration option to load existing perturbations
LOAD_EXISTING_PERTURBATIONS = True  # Set to False to generate new perturbations
PERTURBATIONS_FILE = "gdrive/My Drive/Computational/llm_interpretation/perturbations.json"

# Batch API configuration
BATCH_INPUT_DIR = "gdrive/My Drive/Computational/llm_interpretation/batch_inputs"
BATCH_RESULTS_DIR = "gdrive/My Drive/Computational/llm_interpretation/batch_results"
BATCH_CHECK_INTERVAL = 60  # Check batch status every 60 seconds
PROCESS_BATCHES_IN_PARALLEL = True  # Set to True to process multiple model batches simultaneously
MAX_PARALLEL_BATCHES = 3  # Maximum number of batches to process in parallel
MAX_BATCH_SIZE = 50000  # Maximum number of requests per batch (OpenAI limit)

# Random subset configuration
PROCESS_RANDOM_SUBSET = False  # Set to True to process only a random subset of rows
SUBSET_SIZE = 20  # Number of rows to process when using random subset
RANDOM_SEED = 42  # Fixed random seed for reproducibility

# List of models to test
MODELS_TO_TEST = [
    "gpt-4.1-2025-04-14",
    "gpt-4o-2024-11-20", 
    "gpt-4.1-mini-2025-04-14",
    "o3-2025-04-16",
    "gpt-5"
]

# Reasoning models that don't support temperature, logprobs, or top_logprobs parameters
REASONING_MODELS = ["o4-mini-2025-04-16", "o3-2025-04-16", "gpt-5"]
REASONING_MODEL_RUNS = 10  # Number of times to run reasoning models for averaging
SKIP_REASONING_MODEL_LOGPROBS = True  # When True, skip hidden-logprob proxies for reasoning models (only use confidence method)

# Model pricing (per million tokens)
MODEL_PRICING = {
    "gpt-4.1-2025-04-14": {"input": 1.00, "output": 4.00},
    "gpt-4.1-mini-2025-04-14": {"input": 0.20, "output": 0.80},
    "gpt-4.1-nano-2025-04-14": {"input": 0.05, "output": 0.20},
    "gpt-4.5-preview-2025-02-27": {"input": 37.50, "output": 75.00},
    "gpt-4o-2024-08-06": {"input": 1.25, "output": 5.00},
    "gpt-4o-2024-11-20": {"input": 1.25, "output": 5.00},  # Assuming same as gpt-4o-2024-08-06
    "gpt-4o-mini-2024-07-18": {"input": 0.075, "output": 0.30},
    "o1-2024-12-17": {"input": 7.50, "output": 30.00},
    "o1-pro-2025-03-19": {"input": 75.00, "output": 300.00},
    "o3-pro-2025-06-10": {"input": 10.00, "output": 40.00},
    "o3-2025-04-16": {"input": 1.00, "output": 4.00},
    "o3-deep-research-2025-06-26": {"input": 5.00, "output": 20.00},
    "o4-mini-2025-04-16": {"input": 0.55, "output": 2.20}
}

# Initialize the Anthropic client and OpenAI API key
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
client = openai.OpenAI(api_key=OPENAI_API_KEY)


def calculate_confidence_topk_metrics(logprobs_obj, max_confidence=100):
    """Calculate weighted confidence and bound omitted top-k mass."""
    if not logprobs_obj or 'content' not in logprobs_obj or not logprobs_obj['content']:
        return None, None, None

    first_token = logprobs_obj['content'][0]
    if 'top_logprobs' not in first_token:
        return None, None, None

    valid_top_mass = 0.0
    weighted_sum = 0.0
    valid_values = 0

    for logprob in first_token['top_logprobs']:
        match = re.fullmatch(r'\s*(\d+)\s*', logprob.get('token', ''))
        if not match:
            continue

        token_value = int(match.group(1))
        if 0 <= token_value <= max_confidence:
            token_prob = np.exp(logprob['logprob'])
            valid_top_mass += token_prob
            weighted_sum += token_value * token_prob
            valid_values += 1

    if valid_values == 0 or valid_top_mass <= 0:
        return None, None, None

    weighted_confidence = weighted_sum / valid_top_mass
    omitted_or_invalid_mass = max(0.0, 1.0 - valid_top_mass)
    error_bound = max_confidence * omitted_or_invalid_mass
    return weighted_confidence, valid_top_mass, error_bound


# Function to implement exponential backoff retry logic
def retry_with_exponential_backoff(func, max_retries=10, initial_delay=60, 
                                  max_delay=300, backoff_factor=1.5):
    """
    Retry a function with exponential backoff when API errors occur.
    
    Args:
        func: The function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Factor by which the delay increases
        
    Returns:
        The result of the function call
    """
    num_retries = 0
    delay = initial_delay
    
    while True:
        try:
            return func()
        except (OverloadedError, RateLimitError, APIError, APIStatusError) as e:
            num_retries += 1
            if num_retries > max_retries:
                raise Exception(f"Maximum number of retries ({max_retries}) exceeded: {str(e)}")
            
            # Add some jitter to the delay to prevent all clients retrying simultaneously
            jitter = random.uniform(0.8, 1.2)
            sleep_time = min(delay * jitter, max_delay)
            
            print(f"API error: {str(e)}. Retrying in {sleep_time:.1f} seconds (retry {num_retries}/{max_retries})...")
            time.sleep(sleep_time)
            
            # Increase the delay for the next retry
            delay = min(delay * backoff_factor, max_delay)

# === Batch API Helper Functions ===
def create_random_subset(prompt_parts_list, rephrasings_list, subset_size, random_seed):
    """
    Create a random subset of perturbations for processing.
    Returns filtered prompt_parts_list and rephrasings_list, and the indices used.
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create a list of all perturbations with their indices
    all_perturbations = []
    for prompt_idx, (prompt_parts, rephrasings) in enumerate(zip(prompt_parts_list, rephrasings_list)):
        for rephrase_idx, rephrased_main in enumerate(rephrasings):
            all_perturbations.append((prompt_idx, rephrase_idx))
    
    # Calculate total number of perturbations
    total_perturbations = len(all_perturbations)
    
    # Sample subset
    if subset_size >= total_perturbations:
        print(f"Subset size ({subset_size}) >= total perturbations ({total_perturbations}). Processing all perturbations.")
        return prompt_parts_list, rephrasings_list, None, total_perturbations
    
    # Randomly select indices
    selected_indices = random.sample(all_perturbations, subset_size)
    
    # Create new lists with only selected perturbations
    subset_prompt_parts = []
    subset_rephrasings = []
    
    # Group selected indices by prompt_idx
    from collections import defaultdict
    selected_by_prompt = defaultdict(list)
    for prompt_idx, rephrase_idx in selected_indices:
        selected_by_prompt[prompt_idx].append(rephrase_idx)
    
    # Build subset lists
    for prompt_idx in sorted(selected_by_prompt.keys()):
        prompt_parts = prompt_parts_list[prompt_idx]
        original_rephrasings = rephrasings_list[prompt_idx]
        
        # Get only selected rephrasings for this prompt
        selected_rephrase_indices = sorted(selected_by_prompt[prompt_idx])
        selected_rephrasings = [original_rephrasings[idx] for idx in selected_rephrase_indices]
        
        subset_prompt_parts.append(prompt_parts)
        subset_rephrasings.append(selected_rephrasings)
    
    print(f"Selected {subset_size} random perturbations out of {total_perturbations} total ({subset_size/total_perturbations*100:.1f}%)")
    
    return subset_prompt_parts, subset_rephrasings, selected_indices, total_perturbations

def load_existing_results(output_excel):
    """
    Load existing results from the Excel file if it exists.
    Returns a set of tuples (model, original_main, rephrased_main) that have already been processed.
    """
    processed_perturbations = set()
    
    if os.path.exists(output_excel):
        print(f"Loading existing results from {output_excel}...")
        try:
            existing_df = pd.read_excel(output_excel)
            
            # Create a set of already processed perturbations
            for _, row in existing_df.iterrows():
                # Use model, original main part, and rephrased main part as unique identifier
                key = (
                    row['Model'],
                    row['Original Main Part'],
                    row['Rephrased Main Part']
                )
                processed_perturbations.add(key)
            
            print(f"Found {len(processed_perturbations)} already processed perturbations")
        except Exception as e:
            print(f"Error loading existing results: {str(e)}")
            print("Continuing without filtering...")
    
    return processed_perturbations

def create_batch_requests(model_name, prompt_parts_list, rephrasings_list, is_reasoning_model=False, processed_perturbations=None):
    """
    Create batch requests for all prompts and their rephrasings.
    Returns a list of request dictionaries and a mapping of custom_ids to prompt info.
    """
    requests = []
    id_mapping = {}
    request_counter = 0
    skipped_count = 0
    
    for prompt_idx, (prompt_parts, rephrasings) in enumerate(zip(prompt_parts_list, rephrasings_list)):
        orig_main, orig_format, target_tokens, confidence_format = prompt_parts
        
        for rephrase_idx, rephrased_main in enumerate(rephrasings):
            # Check if this perturbation has already been processed
            if processed_perturbations and (model_name, orig_main, rephrased_main) in processed_perturbations:
                skipped_count += 1
                continue
            
            # Create requests for both binary and confidence formats
            # Skip binary format for reasoning models if SKIP_REASONING_MODEL_LOGPROBS is True
            formats_to_process = ['confidence'] if (is_reasoning_model and SKIP_REASONING_MODEL_LOGPROBS) else ['binary', 'confidence']
            
            for format_type in formats_to_process:
                if format_type == 'binary':
                    full_prompt = f"{rephrased_main} {orig_format}"
                else:
                    full_prompt = f"{rephrased_main} {confidence_format}"
                
                # For reasoning models, create multiple requests only for a sampling-based binary-response proxy.
                num_requests = REASONING_MODEL_RUNS if (is_reasoning_model and format_type == 'binary' and not SKIP_REASONING_MODEL_LOGPROBS) else 1
                
                for run_idx in range(num_requests):
                    custom_id = f"req-{request_counter}"
                    
                    # Store mapping information
                    id_mapping[custom_id] = {
                        'prompt_idx': prompt_idx,
                        'rephrase_idx': rephrase_idx,
                        'format_type': format_type,
                        'run_idx': run_idx,
                        'original_main': orig_main,
                        'response_format': orig_format,
                        'confidence_format': confidence_format,
                        'rephrased_main': rephrased_main,
                        'full_prompt': full_prompt,
                        'target_tokens': target_tokens,
                        'model': model_name
                    }
                    
                    # Create the request
                    request_body = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": full_prompt}],
                        "response_format": {"type": "text"}
                    }
                    
                    # Set max tokens parameter based on model type
                    if is_reasoning_model:
                        request_body["max_completion_tokens"] = 2000
                    else:
                        request_body["max_tokens"] = 500
                        request_body["temperature"] = 0.0
                        request_body["logprobs"] = True
                        request_body["top_logprobs"] = 20
                    
                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": request_body
                    }
                    
                    requests.append(request)
                    request_counter += 1
    
    if skipped_count > 0:
        print(f"Skipped {skipped_count} already processed perturbations for {model_name}")
    
    return requests, id_mapping

def save_batch_input_file(requests, model_name):
    """Save batch requests to a JSONL file."""
    os.makedirs(BATCH_INPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{BATCH_INPUT_DIR}/batch_input_{model_name}_{timestamp}.jsonl"
    
    with open(filename, 'w') as f:
        for request in requests:
            f.write(json.dumps(request) + '\n')
    
    print(f"Created batch input file: {filename}")
    return filename

def upload_batch_file(file_path):
    """Upload the batch input file to OpenAI."""
    with open(file_path, "rb") as file:
        batch_input_file = client.files.create(
            file=file,
            purpose="batch"
        )
    print(f"Uploaded file ID: {batch_input_file.id}")
    return batch_input_file.id

def create_batch(input_file_id, model_name):
    """Create a batch job with the uploaded file."""
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Perturbation analysis for {model_name}",
            "model": model_name
        }
    )
    print(f"Created batch ID: {batch.id} for model {model_name}")
    return batch.id

def check_batch_status(batch_id):
    """Check the status of a batch job."""
    batch = client.batches.retrieve(batch_id)
    return batch.status, batch

def wait_for_batch_completion(batch_id, model_name):
    """Wait for batch to complete, checking periodically."""
    print(f"Waiting for batch to complete for {model_name}...")
    
    while True:
        status, batch = check_batch_status(batch_id)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Batch status for {model_name}: {status}")
        
        if status == "completed":
            print(f"Batch completed successfully for {model_name}!")
            return batch
        elif status in ["failed", "cancelled", "expired"]:
            print(f"Batch failed for {model_name} with status: {status}")
            if hasattr(batch, 'errors') and batch.errors:
                print(f"Errors: {batch.errors}")
            return None
        
        time.sleep(BATCH_CHECK_INTERVAL)

def retrieve_batch_results(output_file_id, model_name):
    """Download the results from a completed batch."""
    content = client.files.content(output_file_id)
    
    # Save raw results
    os.makedirs(BATCH_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{BATCH_RESULTS_DIR}/batch_results_{model_name}_{timestamp}.jsonl"
    
    with open(results_file, 'w') as f:
        f.write(content.content.decode('utf-8'))
    
    print(f"Saved raw results to: {results_file}")
    return content.content.decode('utf-8')

def estimate_tokens(text):
    """Estimate token count for a text string (rough approximation)."""
    # Rough estimation: ~4 characters per token on average
    return len(text) // 4

def process_batch_results(results_content, id_mapping, is_reasoning_model=False):
    """Process the JSONL results from the batch."""
    results_by_prompt = {}
    total_input_tokens = 0
    total_output_tokens = 0
    
    for line in results_content.strip().split('\n'):
        if not line:
            continue
            
        result = json.loads(line)
        custom_id = result['custom_id']
        
        if custom_id not in id_mapping:
            print(f"Warning: Unknown custom_id {custom_id}")
            continue
        
        mapping_info = id_mapping[custom_id]
        key = (mapping_info['prompt_idx'], mapping_info['rephrase_idx'])
        
        if key not in results_by_prompt:
            results_by_prompt[key] = {
                'mapping_info': mapping_info,
                'binary_results': [],
                'confidence_result': None
            }
        
        if 'response' in result and 'body' in result['response']:
            response_body = result['response']['body']
            
            # Extract token usage if available
            if 'usage' in response_body:
                usage = response_body['usage']
                total_input_tokens += usage.get('prompt_tokens', 0)
                total_output_tokens += usage.get('completion_tokens', 0)
            
            if mapping_info['format_type'] == 'binary':
                results_by_prompt[key]['binary_results'].append(response_body)
            else:
                results_by_prompt[key]['confidence_result'] = response_body
        else:
            error_msg = result.get('error', {}).get('message', 'Unknown error')
            print(f"Error for request {custom_id}: {error_msg}")
    
    return results_by_prompt, total_input_tokens, total_output_tokens

def extract_results_from_batch(results_by_prompt, model_name, is_reasoning_model=False):
    """Extract and format results from batch processing."""
    formatted_results = []
    
    for (prompt_idx, rephrase_idx), prompt_results in results_by_prompt.items():
        mapping_info = prompt_results['mapping_info']
        binary_results = prompt_results['binary_results']
        confidence_result = prompt_results['confidence_result']
        confidence_topk_mass = None
        confidence_error_bound = None
        
        # Check if we should skip this result due to missing binary results
        if not binary_results and not (is_reasoning_model and SKIP_REASONING_MODEL_LOGPROBS):
            print(f"Warning: No binary results for prompt {prompt_idx}, rephrasing {rephrase_idx}")
            continue
        
        if is_reasoning_model and not SKIP_REASONING_MODEL_LOGPROBS:
            # Process reasoning model results (average across runs)
            token_1_counts = 0
            token_2_counts = 0
            all_responses = []
            
            for response_body in binary_results:
                answer_text = response_body['choices'][0]['message']['content'].strip()
                all_responses.append(answer_text)
                
                if mapping_info['target_tokens'][0] in answer_text:
                    token_1_counts += 1
                elif mapping_info['target_tokens'][1] in answer_text:
                    token_2_counts += 1
            
            # Calculate probabilities
            num_runs = len(binary_results)
            token_1_prob = token_1_counts / num_runs if num_runs > 0 else 0
            token_2_prob = token_2_counts / num_runs if num_runs > 0 else 0
            odds_ratio = token_1_prob / token_2_prob if token_2_prob > 0 else float('inf')
            
            # Use most common response
            answer_text = max(set(all_responses), key=all_responses.count) if all_responses else ""
            
            # Process confidence
            confidence_value = None
            confidence_answer = ""
            if confidence_result:
                confidence_answer = confidence_result['choices'][0]['message']['content'].strip()
                try:
                    confidence_value = int(re.search(r'\b(\d+)\b', confidence_answer).group(1))
                except (AttributeError, ValueError):
                    pass
            
            weighted_confidence = confidence_value
            
        elif is_reasoning_model and SKIP_REASONING_MODEL_LOGPROBS:
            # For reasoning models when skipping logprobs, only process confidence
            answer_text = "N/A (skipped for reasoning model)"
            token_1_prob = 0.0
            token_2_prob = 0.0
            odds_ratio = 0.0
            
            # Process confidence
            confidence_value = None
            confidence_answer = ""
            weighted_confidence = None
            confidence_topk_mass = None
            confidence_error_bound = None
            
            if confidence_result:
                confidence_answer = confidence_result['choices'][0]['message']['content'].strip()
                try:
                    confidence_value = int(re.search(r'\b(\d+)\b', confidence_answer).group(1))
                    weighted_confidence = confidence_value  # For reasoning models, weighted confidence equals raw confidence
                except (AttributeError, ValueError):
                    pass
            
        else:
            # Process non-reasoning model results
            response_body = binary_results[0]  # Should only have one result
            answer_text = response_body['choices'][0]['message']['content'].strip()
            
            # Extract log probabilities
            token_1_prob = 0.0
            token_2_prob = 0.0
            
            if 'choices' in response_body and response_body['choices']:
                choice = response_body['choices'][0]
                if 'logprobs' in choice and choice['logprobs']:
                    logprobs_obj = choice['logprobs']
                    if 'content' in logprobs_obj and logprobs_obj['content']:
                        first_token = logprobs_obj['content'][0]
                        if 'top_logprobs' in first_token:
                            for logprob in first_token['top_logprobs']:
                                if logprob['token'] == mapping_info['target_tokens'][0]:
                                    token_1_prob = np.exp(logprob['logprob'])
                                elif logprob['token'] == mapping_info['target_tokens'][1]:
                                    token_2_prob = np.exp(logprob['logprob'])
            
            odds_ratio = token_1_prob / token_2_prob if token_2_prob > 0 else float('inf')
            
            # Process confidence
            confidence_value = None
            confidence_answer = ""
            weighted_confidence = None
            
            if confidence_result:
                confidence_answer = confidence_result['choices'][0]['message']['content'].strip()
                try:
                    confidence_value = int(re.search(r'\b(\d+)\b', confidence_answer).group(1))
                except (AttributeError, ValueError):
                    pass
                
                # Calculate weighted confidence and the worst-case omitted-mass bound.
                if 'choices' in confidence_result and confidence_result['choices']:
                    choice = confidence_result['choices'][0]
                    if 'logprobs' in choice and choice['logprobs']:
                        logprobs_obj = choice['logprobs']
                        metrics = calculate_confidence_topk_metrics(logprobs_obj)
                        weighted_confidence, confidence_topk_mass, confidence_error_bound = metrics
        
        # Create result dictionary
        result_dict = {
            "Model": model_name,
            "Original Main Part": mapping_info['original_main'],
            "Response Format": mapping_info['response_format'],
            "Confidence Format": mapping_info['confidence_format'],
            "Rephrased Main Part": mapping_info['rephrased_main'],
            "Full Rephrased Prompt": f"{mapping_info['rephrased_main']} {mapping_info['response_format']}",
            "Full Confidence Prompt": f"{mapping_info['rephrased_main']} {mapping_info['confidence_format']}",
            "Model Response": answer_text,
            "Model Confidence Response": confidence_answer,
            "Log Probabilities": "N/A for reasoning models" if is_reasoning_model else str(response_body.get('choices', [{}])[0].get('logprobs', {})),
            "Token_1_Prob": token_1_prob,
            "Token_2_Prob": token_2_prob,
            "Odds_Ratio": odds_ratio,
            "Confidence Value": confidence_value,
            "Weighted Confidence": weighted_confidence,
            "Confidence Top-K Mass": confidence_topk_mass,
            "Confidence Error Bound": confidence_error_bound
        }
        
        formatted_results.append(result_dict)
    
    return formatted_results

def process_model_batch(model_name, prompt_parts_list, rephrasings_list, processed_perturbations=None, subset_ratio=1.0):
    """
    Process a single model through the batch API.
    Returns the formatted results for this model and token usage statistics.
    """
    print(f"\n=== Starting batch processing for model: {model_name} ===")
    
    # Check if this is a reasoning model
    is_reasoning_model = model_name in REASONING_MODELS
    
    # Create batch requests
    print(f"Creating batch requests for {model_name}...")
    requests, id_mapping = create_batch_requests(
        model_name, 
        prompt_parts_list, 
        rephrasings_list, 
        is_reasoning_model,
        processed_perturbations
    )
    print(f"Created {len(requests)} requests for {model_name}")
    
    # If no requests were created (all were skipped), return empty results
    if len(requests) == 0:
        print(f"All perturbations for {model_name} have already been processed. Skipping batch processing.")
        return [], 0, 0
    
    # Check if we need to chunk the requests (OpenAI batch limit is 50,000)
    if len(requests) > MAX_BATCH_SIZE:
        print(f"Request count ({len(requests)}) exceeds batch limit ({MAX_BATCH_SIZE}). Splitting into chunks...")
        if is_reasoning_model:
            # Calculate approximate number of perturbations being processed
            if SKIP_REASONING_MODEL_LOGPROBS:
                approx_perturbations = len(requests)  # Only confidence requests
                print(f"Note: {model_name} is a reasoning model (confidence only, logprobs skipped)")
            else:
                approx_perturbations = len(requests) // (REASONING_MODEL_RUNS * 2 + 2)  # binary runs + confidence
                print(f"Note: {model_name} is a reasoning model with {REASONING_MODEL_RUNS} runs per perturbation")
            print(f"Processing approximately {approx_perturbations} perturbations")
        
        # Split requests into chunks
        request_chunks = []
        id_mapping_chunks = []
        
        for i in range(0, len(requests), MAX_BATCH_SIZE):
            chunk_requests = requests[i:i + MAX_BATCH_SIZE]
            chunk_id_mapping = {req['custom_id']: id_mapping[req['custom_id']] for req in chunk_requests}
            request_chunks.append(chunk_requests)
            id_mapping_chunks.append(chunk_id_mapping)
        
        print(f"Split into {len(request_chunks)} chunks")
        
        # Process each chunk
        all_model_results = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        for chunk_idx, (chunk_requests, chunk_id_mapping) in enumerate(zip(request_chunks, id_mapping_chunks)):
            print(f"\nProcessing chunk {chunk_idx + 1}/{len(request_chunks)} ({len(chunk_requests)} requests)...")
            
            # Save batch input file for this chunk
            input_file_path = save_batch_input_file(chunk_requests, f"{model_name}_chunk{chunk_idx + 1}")
            
            try:
                # Upload the file
                input_file_id = upload_batch_file(input_file_path)
                
                # Create the batch job
                batch_id = create_batch(input_file_id, model_name)
                
                # Wait for completion
                completed_batch = wait_for_batch_completion(batch_id, model_name)
                
                if completed_batch:
                    # Retrieve results
                    output_file_id = completed_batch.output_file_id
                    results_content = retrieve_batch_results(output_file_id, f"{model_name}_chunk{chunk_idx + 1}")
                    
                    # Process results and get token counts
                    results_by_prompt, input_tokens, output_tokens = process_batch_results(results_content, chunk_id_mapping, is_reasoning_model)
                    
                    # Extract formatted results
                    chunk_results = extract_results_from_batch(results_by_prompt, model_name, is_reasoning_model)
                    
                    print(f"Successfully processed {len(chunk_results)} results for chunk {chunk_idx + 1}")
                    
                    # Accumulate results and token counts
                    all_model_results.extend(chunk_results)
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                else:
                    print(f"Batch processing failed for {model_name} chunk {chunk_idx + 1}")
                    
            except Exception as e:
                print(f"Error processing batch chunk {chunk_idx + 1} for {model_name}: {str(e)}")
                
            finally:
                # Clean up input file
                if os.path.exists(input_file_path):
                    os.remove(input_file_path)
                    print(f"Cleaned up temporary file: {input_file_path}")
        
        # Calculate and display cost for all chunks combined
        if model_name in MODEL_PRICING and (total_input_tokens > 0 or total_output_tokens > 0):
            input_cost = (total_input_tokens / 1_000_000) * MODEL_PRICING[model_name]["input"]
            output_cost = (total_output_tokens / 1_000_000) * MODEL_PRICING[model_name]["output"]
            subset_total_cost = input_cost + output_cost
            
            # Estimate full dataset cost
            if subset_ratio < 1.0:
                estimated_full_cost = subset_total_cost / subset_ratio
                print(f"\nCost Analysis for {model_name}:")
                print(f"  Subset cost: ${subset_total_cost:.2f} (Input: ${input_cost:.2f}, Output: ${output_cost:.2f})")
                print(f"  Estimated full dataset cost: ${estimated_full_cost:.2f}")
            else:
                print(f"\nTotal cost for {model_name}: ${subset_total_cost:.2f} (Input: ${input_cost:.2f}, Output: ${output_cost:.2f})")
        
        return all_model_results, total_input_tokens, total_output_tokens
        
    else:
        # Process as a single batch (original logic)
        # Save batch input file
        input_file_path = save_batch_input_file(requests, model_name)
        
        try:
            # Upload the file
            input_file_id = upload_batch_file(input_file_path)
            
            # Create the batch job
            batch_id = create_batch(input_file_id, model_name)
            
            # Wait for completion
            completed_batch = wait_for_batch_completion(batch_id, model_name)
            
            if completed_batch:
                # Retrieve results
                output_file_id = completed_batch.output_file_id
                results_content = retrieve_batch_results(output_file_id, model_name)
                
                # Process results and get token counts
                results_by_prompt, input_tokens, output_tokens = process_batch_results(results_content, id_mapping, is_reasoning_model)
                
                # Extract formatted results
                model_results = extract_results_from_batch(results_by_prompt, model_name, is_reasoning_model)
                
                print(f"Successfully processed {len(model_results)} results for {model_name}")
                
                # Calculate and display cost for this subset
                if model_name in MODEL_PRICING:
                    input_cost = (input_tokens / 1_000_000) * MODEL_PRICING[model_name]["input"]
                    output_cost = (output_tokens / 1_000_000) * MODEL_PRICING[model_name]["output"]
                    subset_total_cost = input_cost + output_cost
                    
                    # Estimate full dataset cost
                    if subset_ratio < 1.0:
                        estimated_full_cost = subset_total_cost / subset_ratio
                        print(f"\nCost Analysis for {model_name}:")
                        print(f"  Subset cost: ${subset_total_cost:.2f} (Input: ${input_cost:.2f}, Output: ${output_cost:.2f})")
                        print(f"  Estimated full dataset cost: ${estimated_full_cost:.2f}")
                    else:
                        print(f"\nTotal cost for {model_name}: ${subset_total_cost:.2f} (Input: ${input_cost:.2f}, Output: ${output_cost:.2f})")
                
                return model_results, input_tokens, output_tokens
            else:
                print(f"Batch processing failed for {model_name}")
                return [], 0, 0
                
        except Exception as e:
            print(f"Error processing batch for {model_name}: {str(e)}")
            return [], 0, 0
        
        finally:
            # Clean up input file
            if os.path.exists(input_file_path):
                os.remove(input_file_path)
                print(f"Cleaned up temporary file: {input_file_path}")

# === Step 1: Use Claude's API to generate rephrasings for each prompt ===
prompts = [
    ("An insurance policy contains the following exclusion: 'We will not pay for loss or damage caused directly or indirectly by flood, surface water, waves, tides, tidal waves, overflow of any body of water, or their spray, all whether driven by wind or not.' If an insured files a claim for damage caused by a levee failure rather than natural causes, does the exclusion apply? Please analyze whether the insurance policy would cover the damage or not.", "Answer only 'Covered' if insurance covers the loss or 'Not Covered' if it does not, without any other explanation.", ("Covered", "Not"), "How confident are you that the insurance policy should cover the loss, on a scale from 0 (not confident) to 100 (most confident)? Answer only with a number, without any other explanation."),
    ("In a prenuptial agreement, if one party files a petition for divorce after seven years, withdraws it without serving it, and then files again after ten years, which filing date determines the number of years of marriage for calculating the payment amount? The agreement states that payment will be based on 'the number of full years they have been married at the time a Petition for Dissolution of Marriage is filed.' Please analyze whether 'a Petition' refers to the first or ultimate petition filed.", "Answer only 'First Petition' if the first filing date should be used or 'Ultimate Petition' if the ultimate filing date should be used, without any other explanation.", ("Ultimate", "First"), "How confident are you that the first filing date should be used, on a scale from 0 (not confident) to 100 (most confident)? Answer only with a number, without any other explanation."),
    ("Does the following contract term from 1961 naturally include only existing affiliates at the time of contract, or does it potentially encompass affiliates that might be created over time? The term binds [Company] and its 'other affiliate[s]' to a 50/50 royalty split after deducting fees charged by third parties that intermediate in foreign markets. Please analyze whether the term 'other affiliate[s]' includes only existing affiliates or includes future affiliates as well.", "Answer only 'Existing Affiliates' or 'Future Affiliates', without any other explanation.", ("Existing", "Future"), "How confident are you that the royalty split only includes existing affiliates, on a scale from 0 (not confident) to 100 (most confident)? Answer only with a number, without any other explanation."),
    ("A contractor and business corresponded about construction of a new foundry. The contractor offered to do the job either by offering an itemized list or charging cost + 10%. After a phone call where they allegedly agreed payment would be made 'in the usual manner', the foundry accepted in writing. If one party claims it is customary to pay 85% of payments due at the end of every month, but the other argues payments are only due upon substantial completion, how should the term 'usual manner' be interpreted? Does this term refer to the monthly installment payments or to payment upon completion?", "Answer only 'Monthly Installment Payments' or 'Payment Upon Completion', without any other explanation.", ("Monthly", "Payment"), "How confident are you that the payments should be made every month, on a scale from 0 (not confident) to 100 (most confident)? Answer only with a number, without any other explanation."),
    ("You are analyzing an insurance policy dispute. The policy states: '[Insurer will pay for] the felonious abstraction of insured property (1) from within the premises by a person making felonious entry therein by actual force and violence, of which force and violence there are visible marks made by tools, explosives, electricity or chemicals.' A business has experienced a theft where there is clear evidence that a third party committed the burglary. No inside job is suspected. Based on these terms, would this policy provide compensation for losses resulting from this substantiated third-party burglary? Please analyze whether coverage would be provided.", "Answer only 'Covered' if insurance covers the loss or 'Not Covered' if it does not, without any other explanation.", ("Covered", "Not"), "How confident are you that the insurance policy should cover the loss, on a scale from 0 (not confident) to 100 (most confident)? Answer only with a number, without any other explanation.")
]

rephrasings_results = []  # List to hold tuples of (original_prompt_parts, [rephrasing1, rephrasing2, ...])

# Check if we should load existing perturbations
if LOAD_EXISTING_PERTURBATIONS and os.path.exists(PERTURBATIONS_FILE):
    print(f"Loading existing perturbations from {PERTURBATIONS_FILE}...")
    try:
        with open(PERTURBATIONS_FILE, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        
        # Reconstruct the rephrasings_results structure
        for item in saved_data:
            prompt_parts = (
                item['original_main'],
                item['response_format'],
                tuple(item['target_tokens']),
                item['confidence_format']
            )
            rephrasings_results.append((prompt_parts, item['rephrasings']))
        
        print(f"Successfully loaded perturbations for {len(rephrasings_results)} prompts.")
        
        # Verify the loaded data matches our prompts
        if len(rephrasings_results) != len(prompts):
            print("Warning: Number of loaded perturbations doesn't match number of prompts!")
            print(f"Expected: {len(prompts)}, Got: {len(rephrasings_results)}")
            print("Falling back to generating new perturbations...")
            rephrasings_results = []
            LOAD_EXISTING_PERTURBATIONS = False
        else:
            # Verify each prompt matches
            for i, (loaded_parts, _) in enumerate(rephrasings_results):
                if loaded_parts != prompts[i]:
                    print(f"Warning: Prompt {i+1} doesn't match the loaded perturbation!")
                    print("Falling back to generating new perturbations...")
                    rephrasings_results = []
                    LOAD_EXISTING_PERTURBATIONS = False
                    break
    except Exception as e:
        print(f"Error loading perturbations: {str(e)}")
        print("Falling back to generating new perturbations...")
        rephrasings_results = []
        LOAD_EXISTING_PERTURBATIONS = False

# Generate new perturbations if needed
if not LOAD_EXISTING_PERTURBATIONS or not rephrasings_results:
    print("Generating rephrasings for each prompt using Claude...")
    for prompt_parts in prompts:
        main_prompt, response_format, target_tokens, confidence_format = prompt_parts
        all_rephrasings = []  # To collect all 400 rephrasings for this prompt
        
        # Run rephrasings
        number_of_sessions = 100
        for session in range(1, number_of_sessions + 1):
            print(f" - Session {session}/{number_of_sessions} for prompt: \"{main_prompt[:50]}...\"")
            
            # Construct the Claude prompt - only ask to rephrase the main part
            claude_prompt = (
                "Here is a question:\n###\"{main_prompt}\"###\n"
                "Please rephrase this question in 20 variations that differ from the original question but preserve the substance of the question. Each rephrasing should be a complete question, not just a fragment of a question. "
                "Number each rephrasing from 1 to 20."
            ).format(main_prompt=main_prompt)

            # Define function to call Claude's API
            def call_claude():
                return anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=5000,
                    temperature=0.9,
                    messages=[{"role": "user", "content": claude_prompt}]
                )
            
            # Call Claude's API using the retry mechanism
            try:
                response = retry_with_exponential_backoff(call_claude)
                rephrasings_text = response.content[0].text.strip()
                
                # Parse Claude's response to extract each rephrased prompt
                session_rephrasings = []
                for line in rephrasings_text.splitlines():
                    line = line.strip()
                    if not line or line.lower().startswith("here are"):
                        continue  # skip empty lines and lines starting with "Here are"
                    # Expected format: "1. <rephrased question>"
                    if line[0].isdigit():
                        # Split at the first dot and space after the number
                        parts = line.split('.', 1)
                        if len(parts) > 1:
                            rephrase = parts[1].strip()
                        else:
                            # If the format is "1 <rephrased question>" without a dot
                            rephrase = line.lstrip('0123456789').strip(" .-\t")
                        session_rephrasings.append(rephrase)
                    else:
                        # If the line doesn't start with a number, it might be a continuation of the previous line
                        if session_rephrasings:
                            # Append to the last rephrase (with a space in between for readability)
                            session_rephrasings[-1] += " " + line
                        else:
                            # In case a rephrase line isn't numbered (unexpected), treat the whole line as one rephrase
                            session_rephrasings.append(line)
                
                # Add this session's rephrasings to our collection
                all_rephrasings.extend(session_rephrasings)
                print(f"   - Added {len(session_rephrasings)} rephrasings (total: {len(all_rephrasings)}/2000)")
            except Exception as e:
                print(f"Error in session {session}: {str(e)}. Skipping this session.")
        
        # Store the original parts and its list of rephrasings
        rephrasings_results.append((prompt_parts, all_rephrasings))
        print(f" - Completed all sessions. Generated {len(all_rephrasings)} total rephrasings for prompt: \"{main_prompt[:50]}...\"")
    
    # Save the generated perturbations
    print(f"\nSaving generated perturbations to {PERTURBATIONS_FILE}...")
    try:
        # Prepare data for JSON serialization
        save_data = []
        for prompt_parts, rephrasings in rephrasings_results:
            save_data.append({
                'original_main': prompt_parts[0],
                'response_format': prompt_parts[1],
                'target_tokens': list(prompt_parts[2]),  # Convert tuple to list for JSON
                'confidence_format': prompt_parts[3],
                'rephrasings': rephrasings
            })
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(PERTURBATIONS_FILE), exist_ok=True)
        
        with open(PERTURBATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved perturbations to {PERTURBATIONS_FILE}")
    except Exception as e:
        print(f"Error saving perturbations: {str(e)}")
        print("Continuing with the generated perturbations...")

# === Step 2: Process each rephrasing with all models ===
results = []  # to store final outputs
print("\nSubmitting rephrased prompts to all models using Batch API...")

# Note: The Batch API allows us to process thousands of requests efficiently.
# Instead of making individual API calls for each prompt (which would take hours),
# we create batch files with all requests and submit them for processing.
# This is much more cost-effective and faster for large-scale experiments.
#
# The batch processing works as follows:
# 1. Create all requests for a model and save to a JSONL file
# 2. Upload the file to OpenAI
# 3. Create a batch job
# 4. Wait for the batch to complete (can take minutes to hours)
# 5. Download and process the results
#
# For reasoning models (o3, o4-mini, gpt-5), we can either:
# 1. When SKIP_REASONING_MODEL_LOGPROBS=True (default): Only use confidence method (no hidden-logprob proxy)
# 2. When SKIP_REASONING_MODEL_LOGPROBS=False: Create multiple requests to average sampled binary responses

# Extract prompt parts and rephrasings for batch processing
prompt_parts_list = [parts for parts, _ in rephrasings_results]
rephrasings_list = [rephrasings for _, rephrasings in rephrasings_results]

# Apply random subset selection if enabled
subset_ratio = 1.0
total_perturbations = sum(len(rephrasings) for rephrasings in rephrasings_list)

if PROCESS_RANDOM_SUBSET:
    print(f"\nApplying random subset selection (size: {SUBSET_SIZE}, seed: {RANDOM_SEED})...")
    prompt_parts_list, rephrasings_list, selected_indices, total_perturbations = create_random_subset(
        prompt_parts_list, rephrasings_list, SUBSET_SIZE, RANDOM_SEED
    )
    subset_ratio = SUBSET_SIZE / total_perturbations if total_perturbations > SUBSET_SIZE else 1.0

# Define output file path
output_excel = "gdrive/My Drive/Computational/llm_interpretation/results_30_multi_model.xlsx"

# Load existing results to filter out already processed perturbations
processed_perturbations = load_existing_results(output_excel)

# Track total costs across all models
model_costs = {}

# Process each model using batch API
if PROCESS_BATCHES_IN_PARALLEL and len(MODELS_TO_TEST) > 1:
    print(f"\nProcessing {len(MODELS_TO_TEST)} models in parallel (max {MAX_PARALLEL_BATCHES} at a time)...")
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_BATCHES, len(MODELS_TO_TEST))) as executor:
        # Submit all model batches
        future_to_model = {
            executor.submit(process_model_batch, model_name, prompt_parts_list, rephrasings_list, processed_perturbations, subset_ratio): model_name 
            for model_name in MODELS_TO_TEST
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                model_results, input_tokens, output_tokens = future.result()
                results.extend(model_results)
                
                # Store cost information
                if model_name in MODEL_PRICING and (input_tokens > 0 or output_tokens > 0):
                    model_costs[model_name] = {
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'subset_cost': (input_tokens / 1_000_000) * MODEL_PRICING[model_name]["input"] + 
                                      (output_tokens / 1_000_000) * MODEL_PRICING[model_name]["output"]
                    }
                
                print(f"\nCompleted processing for {model_name}: {len(model_results)} results")
            except Exception as e:
                print(f"\nError processing {model_name}: {str(e)}")
else:
    # Process models sequentially
    print(f"\nProcessing {len(MODELS_TO_TEST)} models sequentially...")
    
    for model_name in MODELS_TO_TEST:
        model_results, input_tokens, output_tokens = process_model_batch(model_name, prompt_parts_list, rephrasings_list, processed_perturbations, subset_ratio)
        results.extend(model_results)
        
        # Store cost information
        if model_name in MODEL_PRICING and (input_tokens > 0 or output_tokens > 0):
            model_costs[model_name] = {
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'subset_cost': (input_tokens / 1_000_000) * MODEL_PRICING[model_name]["input"] + 
                              (output_tokens / 1_000_000) * MODEL_PRICING[model_name]["output"]
            }

# === Step 3: Save all outputs to an Excel file ===
df = pd.DataFrame(results)
df.columns = ["Model", "Original Main Part", "Response Format", "Confidence Format", "Rephrased Main Part", 
              "Full Rephrased Prompt", "Full Confidence Prompt", "Model Response", "Model Confidence Response",
              "Log Probabilities", "Token_1_Prob", "Token_2_Prob", "Odds_Ratio", 
              "Confidence Value", "Weighted Confidence", "Confidence Top-K Mass", "Confidence Error Bound"]

# Convert log probabilities to strings to ensure they're stored properly
df["Log Probabilities"] = df["Log Probabilities"].astype(str)

# Periodically save results to avoid losing data in case of failure
def save_checkpoint(df, count):
    checkpoint_file = f"gdrive/My Drive/Computational/llm_interpretation/results_30_multi_model_checkpoint_{count}.xlsx"
    df.to_excel(checkpoint_file, index=False)
    print(f"Saved checkpoint with {len(df)} results to {checkpoint_file}")

# Save checkpoints every 100 results
if len(results) > 0:
    for i in range(0, len(results), 100):
        if i > 0:
            save_checkpoint(pd.DataFrame(results[:i]), i)

# Check if the output file already exists
if os.path.exists(output_excel):
    print(f"\nExisting results file found at '{output_excel}', appending new results...")
    try:
        # Read existing Excel file
        existing_df = pd.read_excel(output_excel)
        
        # Ensure column names match
        if list(existing_df.columns) == list(df.columns):
            # Combine existing and new results
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            
            # Save combined results
            combined_df.to_excel(output_excel, index=False)
            print(f"Successfully appended {len(df)} new results to the existing {len(existing_df)} results.")
        else:
            # If columns don't match, create a backup of the original and save the new results
            backup_file = output_excel.replace(".xlsx", "_backup.xlsx")
            existing_df.to_excel(backup_file, index=False)
            df.to_excel(output_excel, index=False)
            print(f"Column structure mismatch. Original file backed up at '{backup_file}' and new results saved separately.")
    except Exception as e:
        # If there's an error reading the existing file, save to a new file
        new_file = output_excel.replace(".xlsx", "_new.xlsx")
        df.to_excel(new_file, index=False)
        print(f"Error appending to existing file: {e}")
        print(f"New results have been saved to '{new_file}'.")
else:
    # If the file doesn't exist, create it
    df.to_excel(output_excel, index=False)
    print(f"\nNo existing results file found. New results have been saved to '{output_excel}'.")

print(f"\nProcess completed with {len(df)} results processed in this run.")

# Display cost summary
if model_costs and PROCESS_RANDOM_SUBSET and subset_ratio < 1.0:
    print("\n" + "="*60)
    print("COST ESTIMATION SUMMARY")
    print("="*60)
    print(f"Subset size: {SUBSET_SIZE} perturbations ({subset_ratio*100:.1f}% of total)")
    print(f"Total perturbations in full dataset: {total_perturbations}")
    print("\nPer-Model Cost Estimates:")
    print("-"*60)
    
    total_subset_cost = 0
    total_estimated_cost = 0
    
    for model_name in MODELS_TO_TEST:
        if model_name in model_costs:
            subset_cost = model_costs[model_name]['subset_cost']
            estimated_full_cost = subset_cost / subset_ratio
            total_subset_cost += subset_cost
            total_estimated_cost += estimated_full_cost
            
            print(f"\n{model_name}:")
            print(f"  Subset cost: ${subset_cost:.2f}")
            print(f"  Estimated full dataset cost: ${estimated_full_cost:.2f}")
        else:
            print(f"\n{model_name}: No cost data available")
    
    print("\n" + "-"*60)
    print(f"Total subset cost (all models): ${total_subset_cost:.2f}")
    print(f"Total estimated full dataset cost (all models): ${total_estimated_cost:.2f}")
    print("="*60)
elif model_costs and not PROCESS_RANDOM_SUBSET:
    print("\n" + "="*60)
    print("COST SUMMARY (Full Dataset)")
    print("="*60)
    
    total_cost = 0
    for model_name in MODELS_TO_TEST:
        if model_name in model_costs:
            cost = model_costs[model_name]['subset_cost']
            total_cost += cost
            print(f"{model_name}: ${cost:.2f}")
        else:
            print(f"{model_name}: No cost data available")
    
    print("-"*60)
    print(f"Total cost (all models): ${total_cost:.2f}")
    print("="*60)
