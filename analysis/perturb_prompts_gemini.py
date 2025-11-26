print("Gemini 2.0 Flash Perturbation Analysis - Version 1.2.0 (Confidence Only, No Delays)")

import os
import time
import json
import pandas as pd
import numpy as np
import re
from datetime import datetime
from pathlib import Path
import google.generativeai as genai
from google.generativeai import types
from config import GEMINI_API_KEY

# Configuration - use Path to resolve relative paths correctly
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
PERTURBATIONS_FILE = PROJECT_ROOT / "data" / "perturbations.json"
OUTPUT_EXCEL = PROJECT_ROOT / "results" / "gemini_perturbation_results.xlsx"

# Model configuration
# NOTE: gemini-2.5-pro is currently broken (returns MAX_TOKENS with no content)
# Using gemini-2.0-flash instead, though it doesn't support logprobs properly
MODEL_NAME = "gemini-2.0-flash"  # Working model (2.5 models are currently broken)

# Processing configuration
BATCH_SIZE = 50  # Save results every 50 perturbations
# Gemini 2.0 Flash rate limits: varies by tier
# Each perturbation now needs only 1 request (confidence only, binary format commented out)
# No artificial delays - let the API tell us when to slow down
DELAY_BETWEEN_REQUESTS = 0  # No artificial delay - rely on API rate limiting
MAX_RETRIES = 10  # Increased retries for better handling
INITIAL_RETRY_DELAY = 60  # Initial retry delay in seconds (1 minute)
MAX_RETRY_DELAY = 300  # Maximum retry delay in seconds (5 minutes)
START_FROM_PROMPT = 0  # 0-indexed, so 0 means prompt 1 (start from the beginning)
DEBUG_LOGPROBS = True  # Show detailed logprobs debugging

# Initialize Gemini client
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

# Simple rate limit tracking for retry logic
class RateLimitTracker:
    """Tracks rate limit errors for exponential backoff."""
    def __init__(self):
        self.last_429_time = None
        self.consecutive_429s = 0

    def record_429(self):
        """Record that we got a 429 error."""
        now = time.time()
        if self.last_429_time and (now - self.last_429_time) < 120:  # Within 2 minutes
            self.consecutive_429s += 1
        else:
            self.consecutive_429s = 1
        self.last_429_time = now

    def get_retry_delay(self, retry_count):
        """Calculate retry delay with exponential backoff."""
        # If we're getting repeated 429s, use longer delays
        if self.consecutive_429s > 3:
            base_delay = 120  # Start with 2 minutes for repeated failures
        else:
            base_delay = INITIAL_RETRY_DELAY

        # Exponential backoff with jitter
        delay = min(base_delay * (2 ** retry_count), MAX_RETRY_DELAY)
        # Add 10-20% jitter to avoid thundering herd
        import random
        delay = delay * (1 + random.uniform(0.1, 0.2))
        return delay

    def reset_429_counter(self):
        """Reset the consecutive 429 counter after a successful request."""
        self.consecutive_429s = 0

rate_limiter = RateLimitTracker()  # Simple tracker for handling 429 errors

def load_perturbations(file_path):
    """Load existing perturbations from JSON file."""
    print(f"Loading perturbations from {file_path}...")
    with open(str(file_path), 'r', encoding='utf-8') as f:
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
    output_file = Path(output_file)
    if output_file.exists():
        print(f"Loading existing results from {output_file}...")
        try:
            df = pd.read_excel(str(output_file))
            for _, row in df.iterrows():
                key = (row['Original Main Part'], row['Rephrased Main Part'])
                processed.add(key)
            print(f"Found {len(processed)} already processed perturbations")
        except Exception as e:
            print(f"Error loading existing results: {e}")
    return processed

def process_with_gemini(prompt, use_logprobs=True, retry_count=0):
    """Process a single prompt with Gemini API with retry logic."""
    # No proactive rate limiting - let the API tell us when to slow down

    try:
        # Configure generation with logprobs if requested
        if use_logprobs:
            generation_config = {
                "temperature": 1.0,  # Set to 1.0 for more varied responses
                "max_output_tokens": 500,
                "response_logprobs": True
            }
        else:
            generation_config = {
                "temperature": 1.0,  # Set to 1.0 for more varied responses
                "max_output_tokens": 500
            }
        
        # Show what we're sending
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        print(f"    -> Sending request (logprobs={use_logprobs}): {prompt_preview}")

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Reset 429 counter on success
        rate_limiter.reset_429_counter()

        # Extract text response - handle cases where no text is returned
        text_response = ""
        try:
            text_response = response.text.strip() if response.text else ""
        except ValueError as ve:
            # This happens when finish_reason is not STOP (e.g., MAX_TOKENS with no content)
            # Check if there's any content in the parts
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        # Try to extract text from parts
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                text_parts.append(part.text)
                        text_response = ''.join(text_parts).strip()

                # Log the finish reason for debugging
                if hasattr(candidate, 'finish_reason'):
                    finish_reason_name = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else str(candidate.finish_reason)
                    print(f"    [!] Response finished with reason: {finish_reason_name}")

        if text_response:
            print(f"    <- Got response: {text_response[:50]}..." if len(text_response) > 50 else f"    <- Got response: {text_response}")
        else:
            print(f"    <- No text in response (may still have logprobs)")

        # Extract logprobs if available
        logprobs_data = None
        if use_logprobs and hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'logprobs_result'):
                logprobs_data = candidate.logprobs_result

        return text_response, logprobs_data
        
    except Exception as e:
        error_str = str(e)
        # More specific rate limit detection to avoid false positives
        # Check for actual 429 errors or quota exceeded messages
        is_rate_limit = ('429' in error_str or
                         'quota' in error_str.lower() and 'exceeded' in error_str.lower() or
                         'rate limit' in error_str.lower() or
                         'too many requests' in error_str.lower())

        if is_rate_limit:
            # Rate limit error - retry with exponential backoff
            rate_limiter.record_429()

            if retry_count < MAX_RETRIES:
                # Try to extract retry delay from error message
                import re
                retry_match = re.search(r'retry[_\s]?(?:delay|after).*?(\d+)', error_str, re.IGNORECASE)
                if retry_match:
                    retry_delay = int(retry_match.group(1))
                else:
                    retry_delay = rate_limiter.get_retry_delay(retry_count)

                print(f"  Rate limit hit (consecutive: {rate_limiter.consecutive_429s}). Retrying in {retry_delay:.1f}s (attempt {retry_count + 1}/{MAX_RETRIES})...")
                time.sleep(retry_delay)
                return process_with_gemini(prompt, use_logprobs, retry_count + 1)
            else:
                print(f"  Max retries exceeded for rate limit error (consecutive 429s: {rate_limiter.consecutive_429s})")
                return None, None
        else:
            print(f"  Error processing prompt: {error_str[:200]}...")
            return None, None

def extract_token_probabilities(logprobs_data, target_tokens):
    """Extract probabilities for specific target tokens from logprobs."""
    token_1_prob = 0.0
    token_2_prob = 0.0

    if not logprobs_data:
        if DEBUG_LOGPROBS:
            print(f"      No logprobs data available")
        return token_1_prob, token_2_prob

    try:
        # Debug: Show what we're looking for
        if DEBUG_LOGPROBS:
            print(f"      Looking for tokens: {target_tokens}")

        # Gemini logprobs structure may vary, this is based on the API documentation
        if hasattr(logprobs_data, 'top_candidates') and logprobs_data.top_candidates:
            # Look at the first token position
            first_position = logprobs_data.top_candidates[0]

            if DEBUG_LOGPROBS:
                print(f"      Top 5 tokens at position 0:")
                for i, candidate in enumerate(first_position.candidates[:5]):
                    token_text = candidate.token
                    log_prob = candidate.log_probability
                    prob = np.exp(log_prob)
                    print(f"        {i+1}. '{token_text}': logprob={log_prob:.4f}, prob={prob:.4f}")

            for candidate in first_position.candidates:
                token_text = candidate.token
                log_prob = candidate.log_probability

                if token_text == target_tokens[0]:
                    token_1_prob = np.exp(log_prob)
                elif token_text == target_tokens[1]:
                    token_2_prob = np.exp(log_prob)
        else:
            if DEBUG_LOGPROBS:
                print(f"      No top_candidates in logprobs (response may have been truncated or blocked)")
    except Exception as e:
        print(f"      Error extracting token probabilities: {e}")

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
        if hasattr(logprobs_data, 'top_candidates') and len(logprobs_data.top_candidates) >= 1:
            first_pos = logprobs_data.top_candidates[0] if len(logprobs_data.top_candidates) > 0 else None
            second_pos = logprobs_data.top_candidates[1] if len(logprobs_data.top_candidates) > 1 else None
            third_pos = logprobs_data.top_candidates[2] if len(logprobs_data.top_candidates) > 2 else None
            
            if DEBUG_LOGPROBS:
                print(f"      Calculating weighted confidence from token combinations:")
                print(f"        Positions available: {len(logprobs_data.top_candidates)}")
            
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
                    three_digit_prob_sum = 0.0
                    
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
                                            three_digit_prob_sum += combined_prob
                                
                                # Add two-digit number probability (minus what becomes 100)
                                if 10 <= two_digit_value <= 99:
                                    combined_prob = first_prob * second_prob
                                    
                                    # Subtract probability that continues to form 100
                                    if two_digit_value == 10 and third_pos:
                                        # Find probability of "0" in third position
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
                    # Subtract probability that continues to form longer numbers
                    if second_pos and 1 <= first_digit <= 9:
                        # Find total probability of second position being a digit
                        second_digit_prob = 0.0
                        for second_cand in second_pos.candidates[:19]:
                            if second_cand.token.strip().isdigit() and len(second_cand.token.strip()) == 1:
                                second_digit_prob += np.exp(second_cand.log_probability)
                        
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
                
                if DEBUG_LOGPROBS:
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

def process_perturbations(prompt_parts_list, rephrasings_list, processed_set):
    """Process all perturbations with Gemini."""
    results = []
    total_processed = 0
    total_skipped = 0
    start_time = time.time()
    remaining = sum(len(rephrasings) for rephrasings in rephrasings_list) - len(processed_set)
    
    # Start from specified prompt
    for prompt_idx in range(START_FROM_PROMPT, len(prompt_parts_list)):
        prompt_parts = prompt_parts_list[prompt_idx]
        rephrasings = rephrasings_list[prompt_idx]
        orig_main, orig_format, target_tokens, confidence_format = prompt_parts
        
        print(f"\n{'='*80}")
        print(f"PROCESSING PROMPT {prompt_idx + 1}/{len(prompt_parts_list)}")
        print(f"{'='*80}")
        print(f"Original: {orig_main[:150]}..." if len(orig_main) > 150 else f"Original: {orig_main}")
        print(f"Target tokens: {target_tokens}")
        print(f"Total rephrasings: {len(rephrasings)}")
        
        for rephrase_idx, rephrased_main in enumerate(rephrasings):
            # Check if already processed
            if (orig_main, rephrased_main) in processed_set:
                total_skipped += 1
                continue
            
            # Show perturbation being processed
            print(f"\n  [Perturbation {rephrase_idx + 1}/{len(rephrasings)}]")
            print(f"    Rephrased: {rephrased_main[:150]}..." if len(rephrased_main) > 150 else f"    Rephrased: {rephrased_main}")

            # COMMENTED OUT: Binary format processing (not working with current models)
            # print(f"\n    Processing BINARY format...")
            # binary_prompt = f"{rephrased_main} {orig_format}"
            # binary_response, binary_logprobs = process_with_gemini(binary_prompt, use_logprobs=True)
            #
            # if binary_response is None:
            #     print(f"  Failed to process binary prompt {rephrase_idx + 1}")
            #     print(f"  Skipping this perturbation due to failure")
            #     continue

            # Set default values for binary format (since we're skipping it)
            binary_prompt = f"{rephrased_main} {orig_format}"
            binary_response = "N/A (skipped)"
            binary_logprobs = None
            token_1_prob = 0.0
            token_2_prob = 0.0
            odds_ratio = 0.0

            # # Extract token probabilities
            # token_1_prob, token_2_prob = extract_token_probabilities(binary_logprobs, target_tokens)
            # odds_ratio = token_1_prob / token_2_prob if token_2_prob > 0 else float('inf')
            # print(f"    Token probabilities: '{target_tokens[0]}'={token_1_prob:.4f}, '{target_tokens[1]}'={token_2_prob:.4f}")
            # if token_1_prob > 0 and token_2_prob > 0:
            #     print(f"    Odds ratio: {odds_ratio:.2f}")

            # Process confidence format ONLY
            print(f"\n    Processing CONFIDENCE format...")
            confidence_prompt = f"{rephrased_main} {confidence_format}"
            confidence_response, confidence_logprobs = process_with_gemini(confidence_prompt, use_logprobs=False)  # No logprobs needed

            if confidence_response is None:
                print(f"  Failed to process confidence prompt {rephrase_idx + 1}")
                print(f"  Skipping this perturbation due to failure")
                continue

            confidence_value = extract_confidence_value(confidence_response)
            # COMMENTED OUT: Weighted confidence calculation
            # weighted_confidence = calculate_weighted_confidence(confidence_logprobs)
            weighted_confidence = None  # Not calculated
            print(f"    Confidence: raw={confidence_value}")
            
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
                "Log Probabilities": str(binary_logprobs) if binary_logprobs else "N/A",
                "Token_1_Prob": token_1_prob,
                "Token_2_Prob": token_2_prob,
                "Odds_Ratio": odds_ratio,
                "Confidence Value": confidence_value,
                "Weighted Confidence": weighted_confidence
            }
            
            results.append(result)
            total_processed += 1
            
            # Progress summary
            print(f"\n    [OK] Completed perturbation {total_processed}/{remaining} (Prompt {prompt_idx + 1}/{len(prompt_parts_list)})")
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            if elapsed > 0:
                rate = total_processed / (elapsed / 60)  # perturbations per minute
                print(f"    Stats: {rate:.2f} perturbations/min, Est. remaining: {(remaining - total_processed) / rate:.1f} min")

            # No artificial delays - let the API rate limiting handle it
            
            # Save checkpoint periodically
            if total_processed % BATCH_SIZE == 0:
                save_results(results, OUTPUT_EXCEL, append=True)
                print(f"\n  [CHECKPOINT] Saved {len(results)} results to disk")
                print(f"  Total progress: {total_processed}/{remaining} perturbations processed")
                results = []  # Clear results after saving
    
    # Save any remaining results
    if results:
        save_results(results, OUTPUT_EXCEL, append=True)
    
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped (already processed): {total_skipped}")
    print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Average rate: {total_processed / ((time.time() - start_time) / 60):.2f} perturbations/minute")
    
    return total_processed

def save_results(results, output_file, append=False):
    """Save results to Excel file."""
    if not results:
        return

    df = pd.DataFrame(results)

    # Ensure output directory exists
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if append and output_file.exists():
        # Append to existing file
        try:
            existing_df = pd.read_excel(str(output_file))
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_excel(str(output_file), index=False)
            print(f"Appended {len(df)} results to {output_file}")
        except Exception as e:
            print(f"Error appending to existing file: {e}")
            # Save to a new file instead
            backup_file = output_file.parent / f"{output_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(str(backup_file), index=False)
            print(f"Saved {len(df)} results to backup file: {backup_file}")
    else:
        # Create new file
        df.to_excel(str(output_file), index=False)
        print(f"Saved {len(df)} results to {output_file}")

def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("GEMINI 2.0 FLASH PERTURBATION ANALYSIS (CONFIDENCE ONLY)")
    print("="*60)
    print("\nNOTE: Using gemini-2.0-flash as all 2.5 models are currently broken")
    print("      (they return MAX_TOKENS with no content)")
    print("      Binary format and weighted confidence are DISABLED")
    
    if START_FROM_PROMPT > 0:
        print(f"\nNOTE: Starting from prompt {START_FROM_PROMPT + 1} (0-indexed: {START_FROM_PROMPT})")
        print(f"      This is the affiliates question which should be more balanced")
    
    # Check for API key
    if not GEMINI_API_KEY:
        print("\nERROR: GEMINI_API_KEY not found!")
        print("Please set GEMINI_API_KEY in your config.py or .env file")
        return
    
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
    # Processing speed depends on actual API rate limits
    # Gemini 2.0 Flash typically allows much higher rates than older models
    # Rough estimate based on typical performance
    estimated_time = remaining * 0.2  # ~0.2 seconds per perturbation (optimistic estimate)
    print(f"\nEstimated processing time: {estimated_time/60:.1f} minutes ({estimated_time/3600:.1f} hours)")
    print(f"Note: Actual speed depends on API rate limits")
    print(f"Processing: Confidence values only (binary format disabled)")
    
    # Note: Gemini pricing varies by region and model
    # These are approximate values - update based on current pricing
    estimated_tokens = remaining * 200  # Rough estimate
    estimated_cost = (estimated_tokens / 1_000_000) * 0.075  # $0.075 per 1M tokens (approximate)
    print(f"Estimated cost: ${estimated_cost:.2f} (approximate)")
    
    # Confirm before proceeding
    print("\n" + "="*60)
    response = input("Proceed with processing? (yes/no): ")
    print("="*60)
    if response.lower() != 'yes':
        print("Processing cancelled.")
        return
    
    # Process perturbations
    print("\nStarting processing...")
    print("Press Ctrl+C at any time to stop (progress will be saved)\n")
    start_time = time.time()
    total_processed = process_perturbations(prompt_parts_list, rephrasings_list, processed_set)
    elapsed_time = time.time() - start_time
    
    print(f"\nTotal processing time: {elapsed_time/60:.1f} minutes")
    print(f"Average time per perturbation: {elapsed_time/total_processed:.2f} seconds")
    print(f"\nResults saved to: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()