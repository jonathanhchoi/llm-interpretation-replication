"""
Script to randomly extract 50 statements from irrelevant_statements.txt
and format them as an enumerated LaTeX list.
"""

import random
from pathlib import Path

def main():
    # Read the statements file
    script_dir = Path(__file__).parent
    input_file = script_dir / "irrelevant_statements.txt"
    output_file = script_dir / "irrelevant_statements_sample.tex"

    with open(input_file, 'r', encoding='utf-8') as f:
        statements = [line.strip() for line in f if line.strip()]

    # Randomly select 50 statements
    random.seed(42)  # For reproducibility; remove or change for different samples
    selected = random.sample(statements, 50)

    # Generate LaTeX enumerated list
    latex_lines = [
        "\\begin{enumerate}",
    ]

    for statement in selected:
        # Escape special LaTeX characters
        escaped = statement.replace('&', '\\&').replace('%', '\\%').replace('$', '\\$')
        escaped = escaped.replace('#', '\\#').replace('_', '\\_')
        # Handle degree symbols and other special chars
        escaped = escaped.replace('°', '$^\\circ$')
        escaped = escaped.replace('−', '$-$')  # minus sign
        escaped = escaped.replace('×', '$\\times$')
        escaped = escaped.replace('π', '$\\pi$')
        # Handle Unicode superscripts and subscripts
        escaped = escaped.replace('⁻¹⁹', '$^{-19}$')
        escaped = escaped.replace('⁻³⁴', '$^{-34}$')
        escaped = escaped.replace('²³', '$^{23}$')
        escaped = escaped.replace('₂', '$_2$')
        escaped = escaped.replace('²', '$^2$')
        escaped = escaped.replace('³', '$^3$')
        # Handle en-dash
        escaped = escaped.replace('–', '--')

        latex_lines.append(f"    \\item {escaped}")

    latex_lines.append("\\end{enumerate}")

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines))

    print(f"Generated {output_file} with 50 randomly selected statements.")

if __name__ == "__main__":
    main()
