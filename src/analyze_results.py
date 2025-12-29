import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.api import GLM, add_constant
from statsmodels.genmod.families import Binomial
from statsmodels.stats.proportion import proportions_ztest


OBJECTS = [
    "apple", "book", "coin", "drum", "emerald", "feather", "glove", "hammer",
    "ink", "jar", "key", "lantern", "map", "needle", "orb", "pen", "quill",
    "ring", "stone", "token", "umbrella", "vase", "whistle", "yarn", "compass",
    "crown", "dagger", "envelope", "flag", "goblet", "harp", "idol", "jewel",
    "kite", "lens", "mirror", "notebook", "oar", "pouch", "rope", "spear",
    "tablet", "urn", "violin", "wheel", "anchor", "badge", "candle", "flute",
    "gem", "helmet", "instrument", "journal", "kettle", "locket", "medal",
]


def load_results(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000) -> Tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def segmented_logistic(df: pd.DataFrame) -> Dict[str, float]:
    best = {"break": None, "aic": np.inf}
    n_values = sorted(df["n_characters"].unique())
    y = df["correct"].astype(int).values
    for b in n_values[1:-1]:
        x1 = np.minimum(df["n_characters"].values, b)
        x2 = np.maximum(0, df["n_characters"].values - b)
        long_flag = (df["length"] == "long").astype(int).values
        X = np.column_stack([x1, x2, long_flag])
        X = add_constant(X, has_constant="add")
        model = GLM(y, X, family=Binomial()).fit()
        if model.aic < best["aic"]:
            best = {"break": b, "aic": float(model.aic)}
    return best


def error_types(df: pd.DataFrame) -> Dict[str, int]:
    counts = Counter()
    for _, row in df.iterrows():
        answer = (row.get("response") or "").lower()
        if row["correct"]:
            counts["correct"] += 1
            continue
        if not answer.strip():
            counts["empty"] += 1
            continue
        matched_obj = None
        for obj in OBJECTS:
            if obj in answer:
                matched_obj = obj
                break
        if matched_obj:
            counts["confusion"] += 1
        else:
            counts["other"] += 1
    return dict(counts)


def main() -> None:
    results_path = os.path.join("results", "model_outputs.jsonl")
    df = load_results(results_path)
    if df.empty:
        raise RuntimeError("No results found. Run src/run_experiment.py first.")
    latest_run = df["run_id"].max()
    df = df[df["run_id"] == latest_run].copy()

    metrics = {
        "overall_accuracy": float(df["correct"].mean()),
        "overall_recency": float(df["recency_correct"].mean()),
        "overall_random": float(df["random_correct"].mean()),
    }

    grouped = df.groupby(["n_characters", "length"], as_index=False)
    rows = []
    for (n, length), group in grouped:
        acc = group["correct"].mean()
        ci_low, ci_high = bootstrap_ci(group["correct"].astype(int).values)
        rows.append({
            "n_characters": int(n),
            "length": length,
            "accuracy": float(acc),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "count": int(len(group)),
        })
    metrics["accuracy_by_condition"] = rows

    # Short vs long tests per N
    tests = []
    for n in sorted(df["n_characters"].unique()):
        short = df[(df["n_characters"] == n) & (df["length"] == "short")]
        long = df[(df["n_characters"] == n) & (df["length"] == "long")]
        if short.empty or long.empty:
            continue
        count = np.array([short["correct"].sum(), long["correct"].sum()])
        nobs = np.array([len(short), len(long)])
        stat, pval = proportions_ztest(count, nobs)
        tests.append({
            "n_characters": int(n),
            "z": float(stat),
            "p_value": float(pval),
        })
    metrics["length_tests"] = tests

    # Logistic regression
    y = df["correct"].astype(int)
    X = add_constant(
        pd.DataFrame({
            "n_characters": df["n_characters"],
            "long_flag": (df["length"] == "long").astype(int),
        }),
        has_constant="add",
    )
    model = GLM(y, X, family=Binomial()).fit()
    metrics["logistic_regression"] = {
        "params": {k: float(v) for k, v in model.params.to_dict().items()},
        "pvalues": {k: float(v) for k, v in model.pvalues.to_dict().items()},
    }

    segmented = segmented_logistic(df)
    if segmented["break"] is not None:
        segmented["break"] = int(segmented["break"])
    metrics["segmented_fit"] = segmented
    metrics["error_types"] = error_types(df)

    os.makedirs(os.path.join("results", "plots"), exist_ok=True)

    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=plot_df, x="n_characters", y="accuracy", hue="length", marker="o")
    for _, row in plot_df.iterrows():
        plt.fill_between(
            [row["n_characters"] - 0.2, row["n_characters"] + 0.2],
            [row["ci_low"], row["ci_low"]],
            [row["ci_high"], row["ci_high"]],
            alpha=0.1,
        )
    plt.title("Accuracy vs Character Count")
    plt.xlabel("Number of characters")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "plots", "accuracy_by_n.png"))
    plt.close()

    error_counts = metrics["error_types"]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(error_counts.keys()), y=list(error_counts.values()))
    plt.title("Error Type Distribution")
    plt.xlabel("Error type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join("results", "plots", "error_types.png"))
    plt.close()

    with open(os.path.join("results", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    summary_path = os.path.join("results", "summary.csv")
    plot_df.to_csv(summary_path, index=False)
    print("Saved metrics and plots.")


if __name__ == "__main__":
    main()
