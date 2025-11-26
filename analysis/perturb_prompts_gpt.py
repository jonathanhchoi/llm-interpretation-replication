"""
GPT-4 Perturbation Analysis Script

This script evaluates GPT-4's responses to perturbed prompts, extracting
verbalized confidence scores for comparison with Claude and Gemini models.
"""

import json
import os
import re
import time
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_perturbations(file_path: str = "data/perturbations.json") -> List[Dict]:
    """Load perturbation data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_confidence_value(response_text: str) -> Optional[int]:
    """
    Extract confidence value from model response.
    Looks for the first number in the response.
    """
    if not response_text:
        return None

    # Look for first number in the response
    match = re.search(r'\b(\d+)\b', response_text)
    if match:
        value = int(match.group(1))
        # Validate it's in the expected range
        if 0 <= value <= 100:
            return value
    return None

def calculate_weighted_confidence_gpt(logprobs_data: Dict) -> Optional[float]:
    """
    Calculate weighted confidence from GPT logprobs.
    """
    if not logprobs_data or 'content' not in logprobs_data:
        return None

    weighted_sum = 0.0
    total_prob = 0.0

    for token_data in logprobs_data['content']:
        if 'top_logprobs' not in token_data:
            continue

        for logprob_item in token_data['top_logprobs']:
            token = logprob_item['token']
            prob = np.exp(logprob_item['logprob'])

            # Check if token is a number
            if token.isdigit():
                value = int(token)
                if 0 <= value <= 100:
                    weighted_sum += value * prob
                    total_prob += prob
            # Check for two-digit numbers
            elif len(token) == 2 and token.isdigit():
                value = int(token)
                if 0 <= value <= 100:
                    weighted_sum += value * prob
                    total_prob += prob
            # Check for "100"
            elif token == "100":
                weighted_sum += 100 * prob
                total_prob += prob

    if total_prob > 0:
        return weighted_sum / total_prob
    return None

def evaluate_gpt_perturbation(
    prompt: str,
    confidence_prompt: str,
    model: str = "gpt-4-0125-preview",
    temperature: float = 0.0,
    max_retries: int = 3
) -> Tuple[str, str, Optional[Dict], Optional[int], Optional[float]]:
    """
    Evaluate a single perturbation with GPT-4.
    Returns: (binary_response, confidence_response, logprobs, confidence_value, weighted_confidence)
    """
    for attempt in range(max_retries):
        try:
            # Get binary response
            binary_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=10
            )
            binary_answer = binary_response.choices[0].message.content.strip()

            # Get confidence response with logprobs
            confidence_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": confidence_prompt}],
                temperature=temperature,
                max_tokens=10,
                logprobs=True,
                top_logprobs=20
            )

            confidence_text = confidence_response.choices[0].message.content.strip()
            logprobs = confidence_response.choices[0].logprobs.to_dict() if confidence_response.choices[0].logprobs else None

            # Extract confidence value
            confidence_value = extract_confidence_value(confidence_text)

            # Calculate weighted confidence
            weighted_confidence = calculate_weighted_confidence_gpt(logprobs)

            return binary_answer, confidence_text, logprobs, confidence_value, weighted_confidence

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return "", "", None, None, None

def process_all_perturbations(
    perturbations: List[Dict],
    model: str = "gpt-4-0125-preview",
    output_file: str = "results/gpt4_perturbation_results.xlsx"
) -> pd.DataFrame:
    """
    Process all perturbations and save results to Excel file.
    """
    results = []

    for prompt_data in tqdm(perturbations, desc="Processing prompts"):
        original_main = prompt_data['original_main']
        response_format = prompt_data['response_format']
        confidence_format = prompt_data['confidence_format']
        target_tokens = prompt_data.get('target_tokens', [])

        # Process each rephrasing
        for rephrasing in tqdm(prompt_data['rephrasings'], desc="Processing rephrasings", leave=False):
            # Create full prompts
            full_prompt = f"{rephrasing}\n\n{response_format}"
            full_confidence_prompt = f"{rephrasing}\n\n{confidence_format}"

            # Evaluate with GPT-4
            binary_response, confidence_response, logprobs, confidence_value, weighted_confidence = \
                evaluate_gpt_perturbation(full_prompt, full_confidence_prompt, model)

            # Store results
            result = {
                'Model': model,
                'Original Main Part': original_main,
                'Response Format': response_format,
                'Confidence Format': confidence_format,
                'Rephrased Main Part': rephrasing,
                'Target Tokens': str(target_tokens),
                'Full Rephrased Prompt': full_prompt,
                'Full Confidence Prompt': full_confidence_prompt,
                'Model Response': binary_response,
                'Model Confidence Response': confidence_response,
                'Log Probabilities': json.dumps(logprobs) if logprobs else None,
                'Confidence Value': confidence_value,
                'Weighted Confidence': weighted_confidence
            }

            # Calculate token probabilities if we have target tokens
            if target_tokens and len(target_tokens) >= 2:
                # This would require more complex parsing of the binary response logprobs
                # For now, set to 0 as placeholders
                result['Token_1_Prob'] = 0
                result['Token_2_Prob'] = 0
                result['Odds_Ratio'] = 0

            results.append(result)

            # Rate limiting
            time.sleep(0.5)  # Adjust as needed for API rate limits

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save to Excel
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

    return df

def main():
    """Main function to run GPT-4 perturbation analysis."""
    # Load perturbations
    perturbations = load_perturbations()
    print(f"Loaded {len(perturbations)} prompts with rephrasings")

    # Process all perturbations
    df = process_all_perturbations(perturbations)

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total perturbations processed: {len(df)}")
    print(f"Valid confidence values: {df['Confidence Value'].notna().sum()}")
    print(f"Valid weighted confidence: {df['Weighted Confidence'].notna().sum()}")

    if df['Confidence Value'].notna().any():
        print(f"\nConfidence Value Statistics:")
        print(f"  Mean: {df['Confidence Value'].mean():.2f}")
        print(f"  Std: {df['Confidence Value'].std():.2f}")
        print(f"  Median: {df['Confidence Value'].median():.2f}")
        print(f"  Min: {df['Confidence Value'].min():.0f}")
        print(f"  Max: {df['Confidence Value'].max():.0f}")

    if df['Weighted Confidence'].notna().any():
        print(f"\nWeighted Confidence Statistics:")
        print(f"  Mean: {df['Weighted Confidence'].mean():.2f}")
        print(f"  Std: {df['Weighted Confidence'].std():.2f}")
        print(f"  Median: {df['Weighted Confidence'].median():.2f}")

    return df

if __name__ == "__main__":
    main()