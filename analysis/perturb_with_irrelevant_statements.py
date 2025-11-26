"""
Perturbation Analysis with Irrelevant Statements
This script inserts random irrelevant statements from data/irrelevant_statements.txt
into legal scenarios to test model robustness.
"""

import os
import json
import random
import re
from typing import List, Dict, Tuple
from datetime import datetime

# File paths
IRRELEVANT_STATEMENTS_FILE = "data/irrelevant_statements.txt"
OUTPUT_JSON = "data/perturbations_irrelevant.json"
OUTPUT_READABLE = "data/perturbations_irrelevant_readable.txt"

# We will generate perturbations for every possible position in each scenario

# Legal scenarios
LEGAL_SCENARIOS = [
    {
        "name": "Insurance Policy Water Damage Exclusion",
        "main": "An insurance policy contains the following exclusion: 'We will not pay for loss or damage caused directly or indirectly by flood, surface water, waves, tides, tidal waves, overflow of any body of water, or their spray, all whether driven by wind or not.' If an insured files a claim for damage caused by a levee failure rather than natural causes, does the exclusion apply? Please analyze whether the insurance policy would cover the damage or not.",
        "response_format": "Answer only 'Covered' if insurance covers the loss or 'Not Covered' if it does not, without any other explanation.",
        "target_tokens": ["Covered", "Not"],
        "confidence_format": "How confident are you that the insurance policy should cover the loss, on a scale from 0 (not confident) to 100 (most confident)? Answer only with a number, without any other explanation."
    },
    {
        "name": "Prenuptial Agreement Filing Date",
        "main": "In a prenuptial agreement, if one party files a petition for divorce after seven years, withdraws it without serving it, and then files again after ten years, which filing date determines the number of years of marriage for calculating the payment amount? The agreement states that payment will be based on the number of full years they have been married at the time a 'Petition for Dissolution of Marriage is filed.' Please analyze whether a 'Petition' refers to the first or ultimate petition filed.",
        "response_format": "Answer only 'First' if the first petition determines the payment or 'Ultimate' if the ultimate petition does, without any other explanation.",
        "target_tokens": ["First", "Ultimate"],
        "confidence_format": "How confident are you that the first petition filing date should determine the payment calculation, on a scale from 0 (not confident) to 100 (most confident)? Answer only with a number, without any other explanation."
    },
    {
        "name": "Contract Term Affiliates Interpretation",
        "main": "Does the following contract term from 1961 naturally include only existing affiliates at the time of contract, or does it potentially encompass affiliates that might be created over time? The term binds '[Company] and its other affiliate[s]' to a 50/50 royalty split after deducting fees charged by third parties that intermediate in foreign markets. Please analyze whether the term 'other affiliate[s]' includes only existing affiliates or includes future affiliates as well.",
        "response_format": "Answer only 'Existing' if the term includes only existing affiliates or 'Future' if it includes future affiliates, without any other explanation.",
        "target_tokens": ["Existing", "Future"],
        "confidence_format": "How confident are you that the term includes only existing affiliates, on a scale from 0 (not confident) to 100 (most confident)? Answer only with a number, without any other explanation."
    },
    {
        "name": "Construction Payment Terms",
        "main": "A contractor and business corresponded about construction of a new foundry. The contractor offered to do the job either by offering an itemized list or charging cost + 10%. After a phone call where they allegedly agreed payment would be made in the 'usual manner', the foundry accepted in writing. If one party claims it is customary to pay 85% of payments due at the end of every month, but the other argues payments are only due upon substantial completion, how should the term 'usual manner' be interpreted? Does this term refer to the monthly installment payments or to payment upon completion?",
        "response_format": "Answer only 'Monthly' if payments are due monthly or 'Completion' if payments are due upon completion, without any other explanation.",
        "target_tokens": ["Monthly", "Completion"],
        "confidence_format": "How confident are you that payments should be made monthly, on a scale from 0 (not confident) to 100 (most confident)? Answer only with a number, without any other explanation."
    },
    {
        "name": "Insurance Policy Burglary Coverage",
        "main": "You are analyzing an insurance policy dispute. The policy states: '[Insurer will pay for] the felonious abstraction of insured property (1) from within the premises by a person making felonious entry therein by actual force and violence, of which force and violence there are visible marks made by tools, explosives, electricity or chemicals.' A business has experienced a theft where there is clear evidence that a third party committed the burglary. No inside job is suspected. Based on these terms, would this policy provide compensation for losses resulting from this substantiated third-party burglary? Please analyze whether coverage would be provided.",
        "response_format": "Answer only 'Covered' if the policy covers the burglary or 'Not Covered' if it does not, without any other explanation.",
        "target_tokens": ["Covered", "Not"],
        "confidence_format": "How confident are you that the policy should cover the burglary, on a scale from 0 (not confident) to 100 (most confident)? Answer only with a number, without any other explanation."
    }
]

def load_irrelevant_statements(file_path: str) -> List[str]:
    """Load irrelevant statements from text file."""
    statements = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Each line is a statement - just strip whitespace
            statement = line.strip()
            if statement:  # Skip empty lines
                statements.append(statement)

    print(f"Loaded {len(statements)} irrelevant statements")
    return statements

def insert_irrelevant_statement(text: str, statement: str, position_index: int) -> str:
    """
    Insert an irrelevant statement into the text at a specific position.

    Args:
        text: Original text
        statement: Irrelevant statement to insert
        position_index: Position index where to insert (0 = beginning, 1 = after first sentence, etc.)

    Returns:
        Modified text with irrelevant statement inserted
    """
    # Split text into sentences more carefully
    import re
    # Split on periods followed by space or end of string, but keep the period
    sentences = re.split(r'(?<=\.)\s+', text)

    # Clean up sentences that might not end with period
    cleaned_sentences = []
    for s in sentences:
        s = s.strip()
        if s and not s.endswith('.'):
            s += '.'
        if s:
            cleaned_sentences.append(s)

    # Ensure statement ends with period
    if not statement.endswith('.'):
        statement += '.'

    # Insert at specified position
    if position_index <= len(cleaned_sentences):
        cleaned_sentences.insert(position_index, statement)
    else:
        # If position is beyond the end, append at the end
        cleaned_sentences.append(statement)

    # Rejoin sentences
    return ' '.join(cleaned_sentences)

def get_all_insertion_positions(text: str) -> int:
    """
    Get the number of possible insertion positions for a given text.

    Args:
        text: Original text

    Returns:
        Number of possible positions (beginning + after each sentence)
    """
    import re
    sentences = re.split(r'(?<=\.)\s+', text)
    # Clean up and count actual sentences
    actual_sentences = [s.strip() for s in sentences if s.strip()]
    # Positions: beginning (0) + after each sentence (1, 2, ..., n)
    return len(actual_sentences) + 1

def generate_perturbations(scenarios: List[Dict], statements: List[str]) -> List[Dict]:
    """
    Generate perturbations by inserting irrelevant statements into scenarios at all possible positions.

    Args:
        scenarios: List of legal scenario dictionaries
        statements: List of irrelevant statements

    Returns:
        List of perturbation dictionaries
    """
    perturbations = []

    for scenario in scenarios:
        print(f"\nGenerating perturbations for: {scenario['name']}")

        # Create base entry for original scenario
        base_entry = {
            "scenario_name": scenario["name"],
            "original_main": scenario["main"],
            "response_format": scenario["response_format"],
            "target_tokens": scenario["target_tokens"],
            "confidence_format": scenario["confidence_format"],
            "perturbations_with_irrelevant": []
        }

        # Get number of possible positions for this scenario
        num_positions = get_all_insertion_positions(scenario["main"])
        print(f"  Number of insertion positions: {num_positions}")

        # Generate perturbations for each position and each statement combination
        perturbation_id = 1
        for position_index in range(num_positions):
            # For each position, generate a perturbation with EVERY irrelevant statement
            for statement in statements:
                # Create perturbed version
                perturbed_text = insert_irrelevant_statement(
                    scenario["main"],
                    statement,
                    position_index
                )

                # Determine position description
                if position_index == 0:
                    position_desc = "beginning"
                elif position_index == num_positions - 1:
                    position_desc = "end"
                else:
                    position_desc = f"after_sentence_{position_index}"

                perturbation = {
                    "perturbation_id": perturbation_id,
                    "irrelevant_statement": statement,
                    "position_index": position_index,
                    "position_description": position_desc,
                    "perturbed_text": perturbed_text
                }

                base_entry["perturbations_with_irrelevant"].append(perturbation)
                perturbation_id += 1

        perturbations.append(base_entry)
        print(f"  Generated {len(base_entry['perturbations_with_irrelevant'])} perturbations")

    return perturbations

def save_perturbations(perturbations: List[Dict], json_path: str, readable_path: str):
    """Save perturbations to both JSON and human-readable formats."""

    # Save as JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(perturbations, f, indent=2, ensure_ascii=False)
    print(f"\nSaved perturbations to {json_path}")

    # Create human-readable version
    with open(readable_path, 'w', encoding='utf-8') as f:
        f.write("PERTURBATIONS WITH IRRELEVANT STATEMENTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total scenarios: {len(perturbations)}\n")
        total_perturbations = sum(len(p['perturbations_with_irrelevant']) for p in perturbations)
        f.write(f"Total perturbations: {total_perturbations}\n")
        for p in perturbations:
            f.write(f"  {p['scenario_name']}: {len(p['perturbations_with_irrelevant'])} perturbations\n")
        f.write("=" * 80 + "\n\n")

        for scenario_data in perturbations:
            f.write(f"\nSCENARIO: {scenario_data['scenario_name']}\n")
            f.write("-" * 60 + "\n")
            f.write(f"ORIGINAL:\n{scenario_data['original_main']}\n")
            f.write(f"\nRESPONSE FORMAT: {scenario_data['response_format']}\n")
            f.write(f"TARGET TOKENS: {scenario_data['target_tokens']}\n")
            f.write("-" * 60 + "\n")

            for pert in scenario_data['perturbations_with_irrelevant']:
                f.write(f"\nPerturbation #{pert['perturbation_id']}:\n")
                f.write(f"Irrelevant Statement: {pert['irrelevant_statement']}\n")
                f.write(f"Position: {pert['position_description']} (index: {pert['position_index']})\n")
                f.write(f"Perturbed Text:\n{pert['perturbed_text']}\n")
                f.write("-" * 40 + "\n")

            f.write("\n" + "=" * 80 + "\n")

    print(f"Saved human-readable version to {readable_path}")

def main():
    """Main function to generate perturbations with irrelevant statements."""
    print("Starting Perturbation Generation with Irrelevant Statements")
    print("=" * 60)

    # Load irrelevant statements
    statements = load_irrelevant_statements(IRRELEVANT_STATEMENTS_FILE)

    # Generate perturbations
    perturbations = generate_perturbations(
        LEGAL_SCENARIOS,
        statements
    )

    # Save results
    save_perturbations(perturbations, OUTPUT_JSON, OUTPUT_READABLE)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("-" * 60)
    total_perturbations = sum(len(p['perturbations_with_irrelevant']) for p in perturbations)
    print(f"Total scenarios processed: {len(perturbations)}")
    print(f"Total perturbations generated: {total_perturbations}")
    print(f"Unique irrelevant statements available: {len(statements)}")
    print("=" * 60)

    # Print sample for verification
    if perturbations and perturbations[0]['perturbations_with_irrelevant']:
        print("\nSAMPLE PERTURBATION:")
        print("-" * 60)
        sample = perturbations[0]['perturbations_with_irrelevant'][0]
        print(f"Scenario: {perturbations[0]['scenario_name']}")
        print(f"Irrelevant Statement: {sample['irrelevant_statement']}")
        print(f"Position: {sample['position_description']} (index: {sample['position_index']})")
        print(f"\nFirst 500 characters of perturbed text:")
        print(sample['perturbed_text'][:500] + "...")

if __name__ == "__main__":
    main()