"""
Extract questions from Survey 2 for LLM evaluation.
This script extracts the 50 unique questions from word_meaning_survey_results_part_2.csv
"""
import pandas as pd
import os
import sys

def extract_survey2_questions(survey_file):
    """Extract unique questions from Survey 2 CSV file."""
    # Load the survey file
    df = pd.read_csv(survey_file)

    # Get the row with question text (first row)
    headers = df.iloc[0]

    questions = []
    question_to_col = {}

    # Extract questions from column headers
    for col in df.columns:
        if col.startswith('Q') and '_' in col:
            # Skip attention check questions (Q*_8)
            if col.endswith('_8'):
                continue

            text = headers[col]
            if pd.notna(text) and isinstance(text, str):
                # Extract the actual question part after " - "
                if ' - ' in text:
                    question_text = text.split(' - ')[-1].strip()
                    # Avoid duplicates
                    if question_text not in questions:
                        questions.append(question_text)
                        question_to_col[question_text] = col

    return questions, question_to_col

def main():
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    survey_file = os.path.join(data_dir, 'word_meaning_survey_results_part_2.csv')

    print("="*80)
    print("EXTRACTING SURVEY 2 QUESTIONS FOR LLM EVALUATION")
    print("="*80)

    # Extract questions
    questions, question_to_col = extract_survey2_questions(survey_file)

    print(f"\nExtracted {len(questions)} unique questions from Survey 2\n")

    # Display all questions
    print("Survey 2 Questions:")
    print("-" * 80)
    for i, q in enumerate(questions, 1):
        print(f"{i:2d}. {q}")

    # Generate Python list format for easy copying
    print("\n" + "="*80)
    print("PYTHON LIST FORMAT (for compare_instruct_models.py):")
    print("="*80)
    print("\nprompts_survey2 = [")
    for q in questions:
        # Escape quotes
        q_escaped = q.replace('"', '\\"')
        print(f'    "{q_escaped}",')
    print("]")

    # Save to file for easy reference
    output_file = os.path.join(data_dir, 'question_list_part_2.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(q + '\n')

    print(f"\n[OK] Questions saved to: {output_file}")
    print(f"[OK] Total questions: {len(questions)}")

    return questions

if __name__ == "__main__":
    main()
