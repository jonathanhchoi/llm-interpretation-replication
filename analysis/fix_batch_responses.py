#!/usr/bin/env python3
"""
Fix the incorrectly formatted batch responses file.
This extracts the actual response text from the string representation.
"""

import json
import re
import os

# Determine paths based on current directory
if os.path.basename(os.getcwd()) == 'analysis':
    BATCH_INPUT_FILE = "../data/gemini_batch_requests.jsonl"
    BATCH_OUTPUT_FILE = "../data/gemini_batch_responses.jsonl"
    FIXED_OUTPUT_FILE = "../data/gemini_batch_responses_fixed.jsonl"
else:
    BATCH_INPUT_FILE = "data/gemini_batch_requests.jsonl"
    BATCH_OUTPUT_FILE = "data/gemini_batch_responses.jsonl"
    FIXED_OUTPUT_FILE = "data/gemini_batch_responses_fixed.jsonl"

def extract_text_from_response_string(response_str):
    """Extract the actual text from the string representation of the response."""
    # Look for the text value in the response string
    # Pattern: text='...'
    match = re.search(r"text='([^']*)'", response_str)
    if match:
        return match.group(1)
    return ""

def main():
    print("Fixing batch responses file...")

    # Load custom IDs from the original requests
    print(f"Loading request IDs from {BATCH_INPUT_FILE}...")
    with open(BATCH_INPUT_FILE, 'r') as f:
        request_ids = [json.loads(line)['custom_id'] for line in f]
    print(f"Loaded {len(request_ids)} request IDs")

    # Read the incorrectly formatted responses
    print(f"\nReading responses from {BATCH_OUTPUT_FILE}...")
    with open(BATCH_OUTPUT_FILE, 'r') as f:
        responses = [json.loads(line) for line in f]
    print(f"Loaded {len(responses)} responses")

    # Fix each response
    print("\nFixing responses...")
    fixed_count = 0
    with open(FIXED_OUTPUT_FILE, 'w') as f:
        for idx, response_data in enumerate(responses):
            # Get the correct custom_id from the request
            custom_id = request_ids[idx] if idx < len(request_ids) else f"result_{idx}"

            # Extract the text from the response string
            response_str = response_data.get('response', {}).get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            actual_text = extract_text_from_response_string(response_str)

            # Create properly formatted response
            fixed_response = {
                "custom_id": custom_id,
                "response": {
                    "candidates": [{
                        "content": {
                            "parts": [{"text": actual_text}]
                        },
                        "logprobs_result": None  # These weren't in the original responses
                    }]
                }
            }

            f.write(json.dumps(fixed_response) + '\n')
            fixed_count += 1

            if (idx + 1) % 50 == 0:
                print(f"  Fixed {idx + 1} responses...")

    print(f"\n✓ Fixed {fixed_count} responses")
    print(f"  Saved to: {FIXED_OUTPUT_FILE}")

    # Show sample of fixed data
    print("\nSample of first 3 fixed responses:")
    with open(FIXED_OUTPUT_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line)
            print(f"  {data['custom_id']}: {data['response']['candidates'][0]['content']['parts'][0]['text']}")

if __name__ == "__main__":
    main()
