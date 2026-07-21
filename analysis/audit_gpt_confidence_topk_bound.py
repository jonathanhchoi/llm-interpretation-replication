"""
Audit the top-k approximation error for GPT-4.1 confidence prompts.

The original perturbation spreadsheet saved binary-response logprobs but not
confidence-response logprobs. This script reruns the confidence prompts only,
records the probability mass assigned to valid 0-100 confidence tokens among
the returned top logprobs, and reports the worst-case absolute error bound:

    100 * (1 - top_k_valid_confidence_mass)
"""

import argparse
import csv
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


MODEL = "gpt-4.1-2025-04-14"
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"
INPUT_FILE = REPO_ROOT / "results" / "gpt41_perturbation_results.xlsx"
OUTPUT_FILE = REPO_ROOT / "results" / "gpt41_confidence_topk_audit.csv"


def confidence_topk_metrics(logprobs_content, max_confidence=100):
    if not logprobs_content:
        return None, None, None, None

    first_token = logprobs_content[0]
    valid_mass = 0.0
    weighted_sum = 0.0
    top_values = []

    for item in first_token.top_logprobs:
        match = re.fullmatch(r"\s*(\d+)\s*", item.token)
        if not match:
            continue

        value = int(match.group(1))
        if 0 <= value <= max_confidence:
            prob = math.exp(item.logprob)
            valid_mass += prob
            weighted_sum += value * prob
            top_values.append(value)

    if valid_mass <= 0:
        return None, None, None, None

    weighted_confidence = weighted_sum / valid_mass
    omitted_or_invalid_mass = max(0.0, 1.0 - valid_mass)
    error_bound = max_confidence * omitted_or_invalid_mass
    return valid_mass, error_bound, weighted_confidence, top_values


def load_completed(output_file):
    if not output_file.exists():
        return set()

    completed = set()
    with output_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(int(row["row_index"]))
    return completed


def append_result(output_file, result):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    exists = output_file.exists()
    with output_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_index",
                "response_text",
                "topk_valid_confidence_mass",
                "error_bound",
                "weighted_confidence",
                "top_values",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(result)


def print_summary(output_file):
    """Print the audit statistics reported in the paper from an existing CSV."""
    if not output_file.exists():
        raise FileNotFoundError(f"Audit results not found: {output_file}")

    audited = pd.read_csv(output_file)
    if "error_bound" not in audited.columns:
        raise ValueError(f"Missing error_bound column in {output_file}")

    error_bounds = pd.to_numeric(audited["error_bound"], errors="coerce").dropna()
    if error_bounds.empty:
        raise ValueError(f"No numeric error_bound values found in {output_file}")

    quantiles = error_bounds.quantile([0.5, 0.95, 0.99])
    print(f"Audit summary: {output_file}")
    print(f"N: {len(error_bounds)}")
    print(f"Median: {quantiles.loc[0.5]:.6g}")
    print(f"P95: {quantiles.loc[0.95]:.6g}")
    print(f"P99: {quantiles.loc[0.99]:.6g}")
    print(f"Max: {error_bounds.max():.6g}")


def audit_prompt(row_index, prompt, client, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=3,
                logprobs=True,
                top_logprobs=20,
            )
            choice = response.choices[0]
            valid_mass, error_bound, weighted_confidence, top_values = confidence_topk_metrics(
                choice.logprobs.content
            )
            return {
                "row_index": row_index,
                "response_text": choice.message.content.strip(),
                "topk_valid_confidence_mass": valid_mass,
                "error_bound": error_bound,
                "weighted_confidence": weighted_confidence,
                "top_values": " ".join(str(value) for value in top_values),
            }
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print statistics from the existing audit CSV without making API calls.",
    )
    args = parser.parse_args()

    if args.summary_only:
        print_summary(OUTPUT_FILE)
        return

    df = pd.read_excel(INPUT_FILE)
    completed = load_completed(OUTPUT_FILE)

    rows = [
        (idx, row["Full Confidence Prompt"])
        for idx, row in df.iterrows()
        if idx not in completed and isinstance(row["Full Confidence Prompt"], str)
    ]
    if args.sample:
        random.Random(args.seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"Auditing {len(rows)} confidence prompts; {len(completed)} already complete.")
    if rows:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(ENV_FILE)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            parser.error(f"OPENAI_API_KEY is required for API calls; set it in {ENV_FILE}")
        client = OpenAI(api_key=api_key)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_idx = {
                executor.submit(audit_prompt, idx, prompt, client): idx for idx, prompt in rows
            }
            for count, future in enumerate(as_completed(future_to_idx), start=1):
                result = future.result()
                append_result(OUTPUT_FILE, result)
                if count % 100 == 0:
                    print(f"Completed {count}/{len(rows)}")

    print_summary(OUTPUT_FILE)


if __name__ == "__main__":
    main()
