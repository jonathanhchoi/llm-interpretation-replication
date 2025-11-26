import pandas as pd
import numpy as np
import os

def load_demographic_data(filepaths):
    """Load and combine demographic data from multiple files.

    Args:
        filepaths: A string (single file path) or list of file paths to load and combine

    Returns:
        Combined DataFrame with demographic data
    """
    # Handle single filepath or list of filepaths
    if isinstance(filepaths, str):
        filepaths = [filepaths]

    dfs = []
    for filepath in filepaths:
        df_temp = pd.read_csv(filepath)
        dfs.append(df_temp)

    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)

    return df

def clean_demographic_data(df):
    """Clean demographic data by handling missing values and standardizing categories."""
    # Drop respondents with CONSENT_REVOKED in any column
    df = df[~df.apply(lambda row: row.astype(str).str.contains('CONSENT_REVOKED', case=False).any(), axis=1)].copy()

    # Convert Age to numeric
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

    # Replace DATA_EXPIRED with NaN for cleaner analysis
    df = df.replace('DATA_EXPIRED', np.nan)

    return df

def calculate_demographics_stats(df):
    """Calculate demographic statistics for the table."""
    stats = {}

    # Total N
    stats['n_total'] = len(df)

    # Age statistics
    age_data = df['Age'].dropna()
    stats['age_mean'] = age_data.mean()
    stats['age_std'] = age_data.std()
    stats['age_min'] = age_data.min()
    stats['age_max'] = age_data.max()
    stats['age_n'] = len(age_data)

    # Gender distribution
    gender_counts = df['Sex'].value_counts()
    stats['gender'] = {}
    for gender, count in gender_counts.items():
        if pd.notna(gender):
            stats['gender'][gender] = {
                'count': count,
                'percentage': (count / len(df)) * 100
            }

    # Ethnicity distribution
    ethnicity_counts = df['Ethnicity simplified'].value_counts()
    stats['ethnicity'] = {}
    for ethnicity, count in ethnicity_counts.items():
        if pd.notna(ethnicity):
            stats['ethnicity'][ethnicity] = {
                'count': count,
                'percentage': (count / len(df)) * 100
            }

    # Employment status distribution
    employment_counts = df['Employment status'].value_counts()
    stats['employment'] = {}
    for status, count in employment_counts.items():
        if pd.notna(status):
            stats['employment'][status] = {
                'count': count,
                'percentage': (count / len(df)) * 100
            }

    # Student status distribution
    student_counts = df['Student status'].value_counts()
    stats['student'] = {}
    for status, count in student_counts.items():
        if pd.notna(status):
            stats['student'][status] = {
                'count': count,
                'percentage': (count / len(df)) * 100
            }

    return stats

def generate_latex_table(stats):
    """Generate LaTeX table from demographic statistics."""

    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Demographic Characteristics of Survey Respondents}")
    latex.append("\\label{tab:demographics}")
    latex.append("\\begin{tabular}{lrr}")
    latex.append("\\hline")
    latex.append("\\textbf{Characteristic} & \\textbf{N} & \\textbf{\\%} \\\\")
    latex.append("\\hline")

    # Total N
    latex.append(f"\\textbf{{Total Respondents}} & {stats['n_total']} & 100.0 \\\\")
    latex.append("\\hline")

    # Age
    latex.append("\\textbf{Age} & & \\\\")
    latex.append(f"\\quad Mean (SD) & {stats['age_mean']:.1f} ({stats['age_std']:.1f}) & \\\\")
    latex.append(f"\\quad Range & {stats['age_min']:.0f}--{stats['age_max']:.0f} & \\\\")
    latex.append("\\hline")

    # Gender
    latex.append("\\textbf{Gender} & & \\\\")
    for gender in sorted(stats['gender'].keys()):
        count = stats['gender'][gender]['count']
        pct = stats['gender'][gender]['percentage']
        latex.append(f"\\quad {gender} & {count} & {pct:.1f} \\\\")
    latex.append("\\hline")

    # Ethnicity
    latex.append("\\textbf{Ethnicity} & & \\\\")
    # Sort by count (descending)
    ethnicity_sorted = sorted(stats['ethnicity'].items(),
                             key=lambda x: x[1]['count'],
                             reverse=True)
    for ethnicity, data in ethnicity_sorted:
        count = data['count']
        pct = data['percentage']
        latex.append(f"\\quad {ethnicity} & {count} & {pct:.1f} \\\\")
    latex.append("\\hline")

    # Student Status
    latex.append("\\textbf{Student Status} & & \\\\")
    # Sort by count (descending)
    student_sorted = sorted(stats['student'].items(),
                           key=lambda x: x[1]['count'],
                           reverse=True)
    for status, data in student_sorted:
        count = data['count']
        pct = data['percentage']
        latex.append(f"\\quad {status} & {count} & {pct:.1f} \\\\")
    latex.append("\\hline")

    # Employment Status
    latex.append("\\textbf{Employment Status} & & \\\\")
    # Sort by count (descending)
    employment_sorted = sorted(stats['employment'].items(),
                              key=lambda x: x[1]['count'],
                              reverse=True)
    for status, data in employment_sorted:
        count = data['count']
        pct = data['percentage']
        # Escape special LaTeX characters in employment status
        status_escaped = status.replace("'", "'")
        latex.append(f"\\quad {status_escaped} & {count} & {pct:.1f} \\\\")
    latex.append("\\hline")

    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)

def save_latex_table(latex_content, output_file='demographics_table.tex'):
    """Save LaTeX table to file."""
    with open(output_file, 'w') as f:
        f.write(latex_content)
    print(f"\nLaTeX table saved to {output_file}")

def print_summary(stats):
    """Print a summary of demographic statistics."""
    print("\n" + "="*80)
    print("DEMOGRAPHIC SUMMARY")
    print("="*80)

    print(f"\nTotal Respondents: {stats['n_total']}")

    print(f"\nAge:")
    print(f"  Mean (SD): {stats['age_mean']:.1f} ({stats['age_std']:.1f})")
    print(f"  Range: {stats['age_min']:.0f}-{stats['age_max']:.0f}")
    print(f"  N: {stats['age_n']}")

    print(f"\nGender:")
    for gender in sorted(stats['gender'].keys()):
        count = stats['gender'][gender]['count']
        pct = stats['gender'][gender]['percentage']
        print(f"  {gender}: {count} ({pct:.1f}%)")

    print(f"\nEthnicity:")
    ethnicity_sorted = sorted(stats['ethnicity'].items(),
                             key=lambda x: x[1]['count'],
                             reverse=True)
    for ethnicity, data in ethnicity_sorted:
        count = data['count']
        pct = data['percentage']
        print(f"  {ethnicity}: {count} ({pct:.1f}%)")

    print(f"\nStudent Status:")
    student_sorted = sorted(stats['student'].items(),
                           key=lambda x: x[1]['count'],
                           reverse=True)
    for status, data in student_sorted:
        count = data['count']
        pct = data['percentage']
        print(f"  {status}: {count} ({pct:.1f}%)")

    print(f"\nEmployment Status:")
    employment_sorted = sorted(stats['employment'].items(),
                              key=lambda x: x[1]['count'],
                              reverse=True)
    for status, data in employment_sorted:
        count = data['count']
        pct = data['percentage']
        print(f"  {status}: {count} ({pct:.1f}%)")

    print("\n" + "="*80)

def main():
    # Get the base directory (parent of survey_analysis)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')

    # Load demographic data from both files
    print("Loading demographic data...")
    demographic_files = [
        os.path.join(data_dir, 'demographic_data.csv'),
        os.path.join(data_dir, 'demographic_data_part_2.csv')
    ]

    df = load_demographic_data(demographic_files)
    print(f"Loaded {len(df)} total respondents from {len(demographic_files)} files")

    # Clean data
    print("Cleaning data...")
    df = clean_demographic_data(df)

    # Calculate statistics
    print("Calculating demographic statistics...")
    stats = calculate_demographics_stats(df)

    # Print summary
    print_summary(stats)

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    latex_table = generate_latex_table(stats)

    # Save to file
    output_file = os.path.join(base_dir, 'demographics_table.tex')
    save_latex_table(latex_table, output_file)

    print(f"\nDemographic analysis complete!")
    print(f"LaTeX table saved to: {output_file}")

if __name__ == "__main__":
    main()
