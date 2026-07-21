"""Replicate the 49-question base-versus-tuned checkpoint comparison.

This analysis is deliberately offline and self-contained. It uses the tracked
model token probabilities and the canonical cleaned human survey sample, then
bootstraps questions to quantify uncertainty in each checkpoint's MAE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from survey_analysis_consolidated import load_cleaned_question_responses


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_RESULTS_PATH = REPO_ROOT / "data" / "model_comparison_results.csv"
SURVEY_PATHS = [
    REPO_ROOT / "data" / "word_meaning_survey_results.csv",
    REPO_ROOT / "data" / "word_meaning_survey_results_part_2.csv",
]
OUTPUT_JSON_PATH = REPO_ROOT / "results" / "base_vs_tuned_49q.json"
OUTPUT_TABLE_PATH = REPO_ROOT / "results" / "base_vs_tuned_49q_table.tex"

EXPECTED_HUMAN_N = 884
EXPECTED_QUESTIONS = 49
N_BOOTSTRAP = 10_000
SEED = 42

# Exact checkpoints for the three matched families analyzed in main.tex.
MODEL_FAMILIES = {
    "Falcon": {
        "base": "tiiuae/falcon-7b",
        "tuned": "tiiuae/falcon-7b-instruct",
    },
    "StableLM": {
        "base": "stabilityai/stablelm-base-alpha-7b",
        "tuned": "stabilityai/stablelm-tuned-alpha-7b",
    },
    "RedPajama": {
        "base": "togethercomputer/RedPajama-INCITE-7B-Base",
        "tuned": "togethercomputer/RedPajama-INCITE-7B-Instruct",
    },
}

QUOTE_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a0": " ",
    }
)


def canonicalize_question(value: object) -> str:
    """Normalize curly quotes, case, and whitespace for question matching."""
    return " ".join(str(value).translate(QUOTE_TRANSLATION).split()).casefold()


def load_human_means() -> tuple[dict[str, float], dict[str, object]]:
    """Load the paper's cleaned N=884 sample and return 0--1 question means."""
    question_responses, exclusion_stats = load_cleaned_question_responses(
        [str(path) for path in SURVEY_PATHS],
        require_consent=False,
    )
    final_count = int(exclusion_stats.get("final_count", -1))
    if final_count != EXPECTED_HUMAN_N:
        raise ValueError(
            f"Expected cleaned human N={EXPECTED_HUMAN_N}, found N={final_count}"
        )

    human_means: dict[str, float] = {}
    for question, raw_responses in question_responses.items():
        key = canonicalize_question(question)
        if key in human_means:
            raise ValueError(f"Duplicate canonical human question: {question!r}")

        responses = np.asarray(raw_responses, dtype=float)
        if responses.size == 0 or not np.isfinite(responses).all():
            raise ValueError(f"Invalid human responses for question: {question!r}")
        if np.any((responses < 0.0) | (responses > 100.0)):
            raise ValueError(f"Human responses outside 0--100: {question!r}")
        human_means[key] = float(responses.mean() / 100.0)

    return human_means, exclusion_stats


def load_checkpoint_errors(
    human_means: dict[str, float],
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Return sorted per-question absolute errors for each exact checkpoint."""
    frame = pd.read_csv(MODEL_RESULTS_PATH)
    required_columns = {
        "prompt",
        "model",
        "base_or_instruct",
        "yes_prob",
        "no_prob",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Model results missing columns: {sorted(missing_columns)}"
        )

    checkpoint_order = [
        family_models[role]
        for family_models in MODEL_FAMILIES.values()
        for role in ("base", "tuned")
    ]
    frame = frame.loc[frame["model"].isin(checkpoint_order)].copy()
    found_models = set(frame["model"])
    expected_models = set(checkpoint_order)
    if found_models != expected_models:
        raise ValueError(
            "Exact checkpoint filter mismatch: "
            f"missing={sorted(expected_models - found_models)}, "
            f"unexpected={sorted(found_models - expected_models)}"
        )

    frame["yes_prob"] = pd.to_numeric(frame["yes_prob"], errors="coerce")
    frame["no_prob"] = pd.to_numeric(frame["no_prob"], errors="coerce")
    denominator = frame["yes_prob"] + frame["no_prob"]
    invalid_probability = (
        ~np.isfinite(frame["yes_prob"])
        | ~np.isfinite(frame["no_prob"])
        | ~np.isfinite(denominator)
        | (frame["yes_prob"] < 0.0)
        | (frame["no_prob"] < 0.0)
        | (denominator <= 0.0)
    )
    if invalid_probability.any():
        bad_rows = frame.loc[invalid_probability, ["model", "prompt"]]
        raise ValueError(
            "Invalid yes/no probability rows: "
            f"{bad_rows.to_dict(orient='records')}"
        )

    frame["relative_yes_probability"] = frame["yes_prob"] / denominator
    frame["question_key"] = frame["prompt"].map(canonicalize_question)

    errors_by_model: dict[str, np.ndarray] = {}
    reference_question_keys: tuple[str, ...] | None = None
    for family_models in MODEL_FAMILIES.values():
        for role, expected_role in (("base", "base"), ("tuned", "instruct")):
            model_id = family_models[role]
            checkpoint = frame.loc[frame["model"] == model_id].copy()
            roles = set(checkpoint["base_or_instruct"].dropna().astype(str))
            if roles != {expected_role}:
                raise ValueError(
                    f"{model_id} has role labels {sorted(roles)}, "
                    f"expected only {expected_role!r}"
                )
            if checkpoint["question_key"].duplicated().any():
                duplicates = checkpoint.loc[
                    checkpoint["question_key"].duplicated(keep=False), "prompt"
                ].tolist()
                raise ValueError(f"{model_id} has duplicate questions: {duplicates}")

            checkpoint["human_mean"] = checkpoint["question_key"].map(human_means)
            matched = checkpoint.dropna(subset=["human_mean"]).copy()
            if len(matched) != EXPECTED_QUESTIONS:
                unmatched = checkpoint.loc[
                    checkpoint["human_mean"].isna(), "prompt"
                ].tolist()
                raise ValueError(
                    f"{model_id} matched {len(matched)} questions; "
                    f"expected {EXPECTED_QUESTIONS}. Unmatched: {unmatched}"
                )

            matched = matched.sort_values("question_key").reset_index(drop=True)
            question_keys = tuple(matched["question_key"])
            if reference_question_keys is None:
                reference_question_keys = question_keys
            elif question_keys != reference_question_keys:
                raise ValueError(
                    f"{model_id} does not contain the same 49-question battery"
                )

            errors = np.abs(
                matched["relative_yes_probability"].to_numpy(dtype=float)
                - matched["human_mean"].to_numpy(dtype=float)
            )
            if errors.shape != (EXPECTED_QUESTIONS,) or not np.isfinite(errors).all():
                raise ValueError(f"Invalid error vector for {model_id}")
            errors_by_model[model_id] = errors

    return errors_by_model, checkpoint_order


def bootstrap_checkpoint_maes(
    errors_by_model: dict[str, np.ndarray],
    checkpoint_order: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], dict[str, list[int]]]:
    """Bootstrap each checkpoint with a deterministic independent RNG stream."""
    child_sequences = np.random.SeedSequence(SEED).spawn(len(checkpoint_order))
    summaries: dict[str, dict[str, float]] = {}
    distributions: dict[str, np.ndarray] = {}
    spawn_keys: dict[str, list[int]] = {}

    for model_id, child_sequence in zip(checkpoint_order, child_sequences):
        errors = errors_by_model[model_id]
        rng = np.random.default_rng(child_sequence)
        indices = rng.integers(
            0,
            len(errors),
            size=(N_BOOTSTRAP, len(errors)),
        )
        bootstrap_maes = errors[indices].mean(axis=1)
        ci_lower, ci_upper = np.percentile(bootstrap_maes, [2.5, 97.5])
        summaries[model_id] = {
            "mae": float(errors.mean()),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "n_questions": int(len(errors)),
        }
        distributions[model_id] = bootstrap_maes
        spawn_keys[model_id] = list(child_sequence.spawn_key)

    return summaries, distributions, spawn_keys


def two_sided_bootstrap_p(differences: np.ndarray) -> float:
    """Return the two-sided empirical tail probability around zero."""
    lower_tail = float(np.mean(differences <= 0.0))
    upper_tail = float(np.mean(differences >= 0.0))
    return min(1.0, 2.0 * min(lower_tail, upper_tail))


def build_results() -> dict[str, object]:
    """Run the complete analysis and return the serializable result payload."""
    human_means, exclusion_stats = load_human_means()
    errors_by_model, checkpoint_order = load_checkpoint_errors(human_means)
    checkpoint_summaries, distributions, spawn_keys = bootstrap_checkpoint_maes(
        errors_by_model,
        checkpoint_order,
    )

    family_results: dict[str, object] = {}
    for family, model_ids in MODEL_FAMILIES.items():
        base_id = model_ids["base"]
        tuned_id = model_ids["tuned"]
        differences = distributions[tuned_id] - distributions[base_id]
        ci_lower, ci_upper = np.percentile(differences, [2.5, 97.5])

        family_results[family] = {
            "base": {
                "model_id": base_id,
                **checkpoint_summaries[base_id],
            },
            "tuned": {
                "model_id": tuned_id,
                **checkpoint_summaries[tuned_id],
            },
            "tuned_minus_base": {
                "mean": float(differences.mean()),
                "observed_difference": float(
                    checkpoint_summaries[tuned_id]["mae"]
                    - checkpoint_summaries[base_id]["mae"]
                ),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
                "p_value_two_sided": two_sided_bootstrap_p(differences),
            },
        }

    return {
        "metadata": {
            "model_results_path": "data/model_comparison_results.csv",
            "survey_paths": [
                "data/word_meaning_survey_results.csv",
                "data/word_meaning_survey_results_part_2.csv",
            ],
            "human_cleaned_n": int(exclusion_stats["final_count"]),
            "human_question_count": int(len(human_means)),
            "matched_questions_per_checkpoint": EXPECTED_QUESTIONS,
            "probability_formula": "yes_prob / (yes_prob + no_prob)",
            "bootstrap_unit": "question",
            "bootstrap_resamples": N_BOOTSTRAP,
            "bootstrap_ci": "percentile [2.5, 97.5]",
            "seed": SEED,
            "rng_scheme": (
                "numpy SeedSequence(seed).spawn(6), one independent stream "
                "per checkpoint in reported family/base-tuned order"
            ),
            "rng_spawn_keys": spawn_keys,
            "difference_method": (
                "tuned checkpoint bootstrap MAE minus independently resampled "
                "base checkpoint bootstrap MAE"
            ),
            "p_value_method": (
                "two times the smaller empirical probability that the "
                "difference is <= 0 or >= 0, capped at 1"
            ),
        },
        "families": family_results,
    }


def format_interval(point: float, lower: float, upper: float) -> str:
    return f"{point:.3f} [{lower:.3f}, {upper:.3f}]"


def format_signed_interval(point: float, lower: float, upper: float) -> str:
    return f"{point:+.3f} [{lower:+.3f}, {upper:+.3f}]"


def format_p_value(value: float, latex: bool = False) -> str:
    if value < 0.001:
        return r"$<0.001$" if latex else "<0.001"
    return f"{value:.3f}"


def write_outputs(results: dict[str, object]) -> None:
    """Write machine-readable results and a manuscript-ready LaTeX table."""
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\begin{tabular}{lcccc}",
        r"\hline",
        (
            r"\textbf{Model} & \textbf{Base MAE [95\% CI]} & "
            r"\textbf{Tuned MAE [95\% CI]} & "
            r"\textbf{Tuned $-$ Base [95\% CI]} & \textbf{Two-sided $p$} \\"
        ),
        r"\hline",
    ]
    for family, family_result in results["families"].items():
        base = family_result["base"]
        tuned = family_result["tuned"]
        difference = family_result["tuned_minus_base"]
        lines.append(
            f"{family} & "
            f"{format_interval(base['mae'], base['ci_lower'], base['ci_upper'])} & "
            f"{format_interval(tuned['mae'], tuned['ci_lower'], tuned['ci_upper'])} & "
            f"{format_signed_interval(difference['mean'], difference['ci_lower'], difference['ci_upper'])} & "
            f"{format_p_value(difference['p_value_two_sided'], latex=True)} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            (
                r"\caption{Mean absolute error (MAE) for matched base and tuned "
                r"checkpoints on 49 ordinary-meaning questions. Confidence "
                r"intervals are percentile intervals from 10,000 question-level "
                r"bootstrap resamples. Tuned-minus-base differences subtract "
                r"independent checkpoint bootstrap distributions.}"
            ),
            r"\label{tab:base_vs_tuned_49q_replication}",
            r"\end{table}",
        ]
    )
    OUTPUT_TABLE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(results: dict[str, object]) -> None:
    """Print a concise plain-text summary."""
    print(
        f"{'Family':<12} {'Base MAE [95% CI]':<25} "
        f"{'Tuned MAE [95% CI]':<25} {'Tuned - Base [95% CI]':<29} {'p':>7}"
    )
    print("-" * 104)
    for family, family_result in results["families"].items():
        base = family_result["base"]
        tuned = family_result["tuned"]
        difference = family_result["tuned_minus_base"]
        print(
            f"{family:<12} "
            f"{format_interval(base['mae'], base['ci_lower'], base['ci_upper']):<25} "
            f"{format_interval(tuned['mae'], tuned['ci_lower'], tuned['ci_upper']):<25} "
            f"{format_signed_interval(difference['mean'], difference['ci_lower'], difference['ci_upper']):<29} "
            f"{format_p_value(difference['p_value_two_sided']):>7}"
        )
    print(
        f"\nCleaned human N={results['metadata']['human_cleaned_n']}; "
        f"49 matched questions/checkpoint; {N_BOOTSTRAP:,} bootstrap resamples; "
        f"seed={SEED}."
    )


def main() -> None:
    results = build_results()
    write_outputs(results)
    print_table(results)
    print(
        "\nWrote results/base_vs_tuned_49q.json and "
        "results/base_vs_tuned_49q_table.tex"
    )


if __name__ == "__main__":
    main()
