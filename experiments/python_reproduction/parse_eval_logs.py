import re
import os
import sys
import pandas as pd
from tabulate import tabulate
import argparse


def parse_eval_logs(log_file):
    """Parse evaluation logs and extract relevant metrics."""
    with open(log_file, "r") as f:
        content = f.read()

    # Define regex patterns for extracting information
    dataset_pattern = r"Evaluating ([A-Za-z0-9\-_]+) models for all seeds"
    seed_pattern = r"Processing seed (\d+)"
    experiment_pattern = r"Evaluating ([a-z\.]+) experiment"
    method_pattern = r"Using method: ([a-z]+)"

    # Metric patterns
    accuracy_pattern = r"Accuracy: ([0-9\.]+)"
    coverage_pattern = r"Coverage: ([0-9\.]+)"
    size_pattern = r"Size: ([0-9\.]+)"
    class_size_pattern = r"Class size (\d+): ([0-9\.]+)"
    group_size_pattern = r"Group ([a-z]+) size (\d+): ([0-9\.]+)"
    group_miscoverage_pattern = r"Group ([a-z]+) miscoverage (\d+): ([0-9\.]+)"
    coverage_confusion_pattern = r"Coverage confusion (\d+)-(\d+): ([0-9\.]+)"

    # Initialize data list
    data = []

    # Find all evaluation blocks
    current_dataset = None
    eval_blocks = re.finditer(
        r"Evaluating ([a-z\.]+) experiment[\s\S]*?Coverage confusion \d+-\d+: [0-9\.]+",
        content,
    )

    for block in eval_blocks:
        eval_text = block.group(0)

        # Find current dataset
        dataset_matches = re.findall(dataset_pattern, content[: block.start()])
        if dataset_matches:
            current_dataset = dataset_matches[-1]

        # Extract information
        seed_matches = re.findall(seed_pattern, content[: block.start()])
        seed = seed_matches[-1] if seed_matches else None

        experiment_matches = re.search(experiment_pattern, eval_text)
        experiment = experiment_matches.group(1) if experiment_matches else None

        method_matches = re.search(method_pattern, eval_text)
        method = method_matches.group(1) if method_matches else None

        accuracy_matches = re.search(accuracy_pattern, eval_text)
        accuracy = float(accuracy_matches.group(1)) if accuracy_matches else None

        coverage_matches = re.search(coverage_pattern, eval_text)
        coverage = float(coverage_matches.group(1)) if coverage_matches else None

        size_matches = re.search(size_pattern, eval_text)
        size = float(size_matches.group(1)) if size_matches else None

        # Extract class sizes
        class_sizes = {}
        for match in re.finditer(class_size_pattern, eval_text):
            class_idx = int(match.group(1))
            class_size = float(match.group(2))
            class_sizes[class_idx] = class_size

        # Extract group sizes and miscoverage
        group_metrics = {}
        for match in re.finditer(group_size_pattern, eval_text):
            group_name = match.group(1)
            group_idx = int(match.group(2))
            group_size = float(match.group(3))
            group_key = f"{group_name}_size_{group_idx}"
            group_metrics[group_key] = group_size

        for match in re.finditer(group_miscoverage_pattern, eval_text):
            group_name = match.group(1)
            group_idx = int(match.group(2))
            miscoverage = float(match.group(3))
            group_key = f"{group_name}_miscoverage_{group_idx}"
            group_metrics[group_key] = miscoverage

        # Extract coverage confusion
        confusion_metrics = {}
        for match in re.finditer(coverage_confusion_pattern, eval_text):
            a, b = int(match.group(1)), int(match.group(2))
            value = float(match.group(3))
            confusion_key = f"coverage_confusion_{a}_{b}"
            confusion_metrics[confusion_key] = value

        # Create a data entry
        entry = {
            "dataset": current_dataset,
            "seed": seed,
            "experiment": experiment,
            "method": method,
            "accuracy": accuracy,
            "coverage": coverage,
            "size": size,
            **{f"class_size_{i}": class_sizes.get(i, None) for i in range(10)},
            **group_metrics,
            **confusion_metrics,
        }
        data.append(entry)

    return data


def save_as_csv(data, output_file):
    """Save data as CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"CSV saved to {output_file}")


def display_table(data):
    """Display data as a formatted table."""
    # Create simplified DataFrame for display
    display_df = pd.DataFrame(data)[
        ["dataset", "seed", "experiment", "method", "accuracy", "coverage", "size"]
    ]
    print(tabulate(display_df, headers="keys", tablefmt="fancy_grid"))

    # Display additional tables for specific metrics
    print("\nClass Sizes:")
    class_size_cols = [
        col for col in pd.DataFrame(data).columns if col.startswith("class_size_")
    ]
    if class_size_cols:
        class_df = pd.DataFrame(data)[
            ["dataset", "seed", "experiment", "method"] + class_size_cols
        ]
        print(
            tabulate(class_df, headers="keys", tablefmt="fancy_grid", showindex=False)
        )

    print("\nGroup Metrics:")
    group_cols = [col for col in pd.DataFrame(data).columns if "group" in col.lower()]
    if group_cols:
        group_df = pd.DataFrame(data)[
            ["dataset", "seed", "experiment", "method"] + group_cols
        ]
        print(
            tabulate(group_df, headers="keys", tablefmt="fancy_grid", showindex=False)
        )

    print("\nCoverage Confusion:")
    confusion_cols = [
        col for col in pd.DataFrame(data).columns if "confusion" in col.lower()
    ]
    if confusion_cols:
        confusion_df = pd.DataFrame(data)[
            ["dataset", "seed", "experiment", "method"] + confusion_cols
        ]
        print(
            tabulate(
                confusion_df, headers="keys", tablefmt="fancy_grid", showindex=False
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Parse evaluation logs and generate CSV/tabular output"
    )
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument(
        "--csv", help="Output CSV file path", default="eval_results.csv"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Show summary statistics"
    )
    args = parser.parse_args()

    data = parse_eval_logs(args.log_file)
    save_as_csv(data, args.csv)
    display_table(data)

    if args.summary:
        # Display summary statistics
        df = pd.DataFrame(data)
        print("\nSummary Statistics:")
        print(
            df.groupby(["dataset", "experiment", "method"]).agg(
                {
                    "accuracy": ["mean", "std"],
                    "coverage": ["mean", "std"],
                    "size": ["mean", "std"],
                }
            )
        )


if __name__ == "__main__":
    main()
