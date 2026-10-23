from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "reports" / "rag_evaluation"
RESULTS_PATH = RESULTS_DIR / "baseline_vs_rag.csv"
SCORES_PATH = RESULTS_DIR / "manual_scores.csv"
RETRIEVED_PATH = RESULTS_DIR / "retrieved_examples.csv"
REPORT_PATH = RESULTS_DIR / "evaluation_report.md"
SCORE_COLUMNS = {
    "relevance_1_to_5": "Relevance",
    "clarity_1_to_5": "Clarity",
    "actionability_1_to_5": "Actionability",
    "grounding_1_to_5": "Grounding",
}

def main() -> None:
    for path in (RESULTS_PATH, SCORES_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required evaluation file is missing: {path}"
            )

    results = pd.read_csv(RESULTS_PATH).fillna(
        {"error": "", "draft": ""}
    )
    scores = pd.read_csv(SCORES_PATH)
    retrieved = (
        pd.read_csv(RETRIEVED_PATH)
        if RETRIEVED_PATH.exists()
        else pd.DataFrame()
    )

    lines = [
        "# SupportOps AI: Baseline vs. RAG Evaluation",
        "",
        "## Purpose",
        "",
        "Compare direct LLM generation with generation augmented by "
        "semantic retrieval of historical support examples.",
        "",
        "## Experiment setup",
        "",
        "- Model and temperature are recorded per generation in the results CSV.",
        "- Both modes use the same customer message and verified facts "
        "for each scenario.",
        "- Baseline uses no retrieved examples.",
        "- RAG uses the configured top-k semantic retrieval results.",
        "- The retriever is built from `data/processed/train.csv` only.",
        "- This is a small, exploratory evaluation, not a statistically "
        "conclusive benchmark.",
        "",
        "## Run status",
        "",
        f"- Generation records: {len(results)}",
    ]

    if "mode" in results.columns:
        for mode in ("baseline", "rag"):
            subset = results[results["mode"] == mode]
            successes = (
                subset["error"].astype(str).str.strip().eq("").sum()
            )
            lines.append(
                f"- {mode.title()}: {successes}/{len(subset)} "
                "generation calls succeeded"
            )

    if "error" in results.columns:
        failures = results[
            results["error"].astype(str).str.strip().ne("")
        ]
        if not failures.empty:
            lines.extend(["", "### Errors"])
            for _, row in failures.iterrows():
                lines.append(
                    f"- `{row.get('scenario_id', 'unknown')}` "
                    f"({row.get('mode', 'unknown')}): "
                    f"{row['error']}"
                )

    lines.extend(["", "## Latency", ""])

    successful = results[
        results["error"].astype(str).str.strip().eq("")
    ].copy()

    if not successful.empty and {
        "mode", "latency_seconds"
    }.issubset(successful.columns):
        latency = successful.groupby("mode")[
            "latency_seconds"
        ].agg(["count", "mean", "median"])

        lines.extend([
            "| Mode | Successful calls | Mean seconds | Median seconds |",
            "|---|---:|---:|---:|",
        ])

        for mode, row in latency.iterrows():
            lines.append(
                f"| {mode} | {int(row['count'])} | "
                f"{row['mean']:.3f} | {row['median']:.3f} |"
            )
    else:
        lines.append("No successful generations were available to summarize.")

    lines.extend([
        "",
        "Latency includes per-request work performed inside the evaluation "
        "function. The one-time retriever build is excluded. Results from "
        "a few CPU-based calls should be treated as descriptive observations.",
        "",
        "## Manual response-quality review",
        "",
    ])

    available_scores = []
    for column, label in SCORE_COLUMNS.items():
        if column in scores.columns:
            scores[column] = pd.to_numeric(
                scores[column], errors="coerce"
            )
            if scores[column].notna().any():
                available_scores.append((column, label))

    if available_scores:
        lines.append("| Mode | " + " | ".join(
            label for _, label in available_scores
        ) + " |")
        lines.append("|---|" + "|".join(
            "---:" for _ in available_scores
        ) + "|")

        for mode in ("baseline", "rag"):
            subset = scores[scores["mode"] == mode]
            values = []
            for column, _ in available_scores:
                mean = subset[column].mean()
                values.append(
                    f"{mean:.2f}" if pd.notna(mean) else "Not scored"
                )
            lines.append(f"| {mode} | " + " | ".join(values) + " |")
    else:
        lines.append(
            "Manual scoring has not been completed. Fill in "
            "`manual_scores.csv`, then rerun this script."
        )

    if "unsupported_claims" in scores.columns:
        claims = scores.copy()
        claims["unsupported_claims"] = (
            claims["unsupported_claims"]
            .astype(str).str.strip().str.lower()
        )
        valid = claims[
            claims["unsupported_claims"].isin(["yes", "no"])
        ]

        lines.extend(["", "### Unsupported claims", ""])
        if valid.empty:
            lines.append(
                "No valid Yes/No assessments recorded yet."
            )
        else:
            for mode in ("baseline", "rag"):
                subset = valid[valid["mode"] == mode]
                yes_count = (
                    subset["unsupported_claims"] == "yes"
                ).sum()
                lines.append(
                    f"- {mode.title()}: {yes_count}/"
                    f"{len(subset)} scored drafts contained "
                    "unsupported claims."
                )

    lines.extend(["", "## Retrieval audit", ""])
    if not retrieved.empty and "scenario_id" in retrieved.columns:
        counts = retrieved.groupby("scenario_id").size()
        for scenario_id, count in counts.items():
            lines.append(
                f"- `{scenario_id}`: {count} retrieved examples recorded"
            )
    else:
        lines.append(
            "No retrieved examples were recorded, or the retrieval "
            "output file is empty. Check the evaluation output."
        )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- Only three synthetic customer scenarios were evaluated.",
        "- Manual quality ratings are subjective and require consistent criteria.",
        "- Similarity scores do not establish factual correctness.",
        "- Historical examples can be irrelevant or contain claims that "
        "are not verified for the current customer.",
        "- A successful model call does not guarantee a safe or correct draft.",
        "- Results do not establish production readiness.",
        "",
        "## Conclusion",
        "",
        "Complete this section after reviewing the recorded outputs and "
        "manual scores. State whether RAG helped, hurt, or made little "
        "difference on these scenarios, citing specific examples. Do not "
        "generalize beyond this small evaluation.",
        "",
    ])

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {REPORT_PATH}")
    print(f"Generations recorded: {len(results)}")
    print(f"Manual score rows: {len(scores)}")


if __name__ == "__main__":
    main()
