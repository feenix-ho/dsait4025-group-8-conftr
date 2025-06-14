#!/usr/bin/env python

import re
import os
import sys
import pandas as pd
from tabulate import tabulate
import argparse


def parse_eval_logs(log_file):
    """Parse evaluation logs and extract relevant metrics."""
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Define regex patterns for extracting information
    dataset_pattern = r"===== ([A-Za-z0-9\-_]+) EVALUATION ====="
    seed_pattern = r"Processing seed (\d+)"
    experiment_pattern = r"Evaluating ([a-z\. ]+) experiment"
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

    # Find all dataset sections
    dataset_sections = re.finditer(
        r"===== ([A-Za-z0-9\-_]+) EVALUATION =====[\s\S]+?(?=\n=====|\Z)",
        content,
    )

    for ds_match in dataset_sections:
        ds_text = ds_match.group(0)
        dataset_name = re.search(dataset_pattern, ds_text).group(1)

        # Find seed processing blocks
        seed_blocks = re.finditer(
            r"Processing seed (\d+)[\s\S]+?(?=Processing seed|\Z)", ds_text
        )

        for seed_block in seed_blocks:
            seed_text = seed_block.group(0)
            seed_value = re.search(r"Processing seed (\d+)", seed_text).group(1)

            # Find experiment blocks
            experiment_blocks = re.finditer(
                r"Evaluating ([a-z\. ]+) experiment[\s\S]+?(?=Evaluating|Completed seed|\Z)",
                seed_text,
            )

            for exp_block in experiment_blocks:
                exp_text = exp_block.group(0)
                exp_match = re.search(experiment_pattern, exp_text)
                experiment_name = exp_match.group(1).strip() if exp_match else None

                # Handle experiment name variations
                if "conformal.training" in exp_text:
                    experiment_name = "conformal.training"
                elif "conformal training" in exp_text:
                    experiment_name = "conformal.training"
                elif "baseline" in exp_text:
                    experiment_name = "baseline"

                # Find method blocks
                method_blocks = re.finditer(
                    r"Using method: ([a-z]+)[\s\S]+?(?=Using method:|Evaluating|Completed seed|\Z)",
                    exp_text,
                )

                for method_block in method_blocks:
                    method_text = method_block.group(0)
                    method_name = re.search(method_pattern, method_text).group(1)

                    # Extract metrics
                    accuracy_match = re.search(accuracy_pattern, method_text)
                    accuracy = (
                        float(accuracy_match.group(1)) if accuracy_match else None
                    )

                    coverage_match = re.search(coverage_pattern, method_text)
                    coverage = (
                        float(coverage_match.group(1)) if coverage_match else None
                    )

                    size_match = re.search(size_pattern, method_text)
                    size = float(size_match.group(1)) if size_match else None

                    # Extract class sizes
                    class_sizes = {}
                    for match in re.finditer(class_size_pattern, method_text):
                        class_idx = int(match.group(1))
                        class_size = float(match.group(2))
                        class_sizes[class_idx] = class_size

                    # Extract group metrics
                    group_metrics = {}
                    for match in re.finditer(group_size_pattern, method_text):
                        group_name = match.group(1)
                        group_idx = int(match.group(2))
                        group_size = float(match.group(3))
                        group_key = f"{group_name}_size_{group_idx}"
                        group_metrics[group_key] = group_size

                    for match in re.finditer(group_miscoverage_pattern, method_text):
                        group_name = match.group(1)
                        group_idx = int(match.group(2))
                        try:
                            miscoverage = float(match.group(3))
                        except ValueError:
                            # Handle potential parsing issues
                            continue
                        group_key = f"{group_name}_miscoverage_{group_idx}"
                        group_metrics[group_key] = miscoverage

                    # Extract confusion metrics
                    confusion_metrics = {}
                    for match in re.finditer(coverage_confusion_pattern, method_text):
                        a, b = int(match.group(1)), int(match.group(2))
                        try:
                            value = float(match.group(3))
                        except ValueError:
                            continue
                        confusion_key = f"coverage_confusion_{a}_{b}"
                        confusion_metrics[confusion_key] = value

                    # Create data entry
                    entry = {
                        "dataset": dataset_name,
                        "seed": seed_value,
                        "experiment": experiment_name,
                        "method": method_name,
                        "accuracy": accuracy,
                        "coverage": coverage,
                        "size": size,
                        **{
                            f"class_size_{i}": class_sizes.get(i, None)
                            for i in range(10)
                        },
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
    df = pd.DataFrame(data)
    if len(df) == 0:
        print("No data found in the log file!")
        return

    # Show core metrics table
    display_cols = [
        "dataset",
        "seed",
        "experiment",
        "method",
        "accuracy",
        "coverage",
        "size",
        "training_epochs",
    ]
    display_cols = [col for col in display_cols if col in df.columns]
    print(tabulate(df[display_cols], headers="keys", tablefmt="fancy_grid"))

    # Display summary grouping by experiment type and method
    print("\nSummary by experiment type and method:")
    summary = (
        df.groupby(["dataset", "experiment", "method"])
        .agg(
            {
                "accuracy": ["mean", "std"],
                "coverage": ["mean", "std"],
                "size": ["mean", "std"],
            }
        )
        .reset_index()
    )
    print(tabulate(summary, headers="keys", tablefmt="fancy_grid"))


def main():
    parser = argparse.ArgumentParser(
        description="Parse evaluation logs and generate CSV/tabular output"
    )
    parser.add_argument(
        "--log_file", help="Path to the log file", default="eval_logs.txt"
    )
    parser.add_argument(
        "--csv", help="Output CSV file path", default="eval_results.csv"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Show summary statistics"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed metrics tables"
    )
    args = parser.parse_args()

    data = parse_eval_logs(args.log_file)
    if not data:
        print("No data extracted from log file!")
        return

    save_as_csv(data, args.csv)
    display_table(data)

    if args.detailed:
        df = pd.DataFrame(data)

        # Display additional tables for specific metrics
        print("\nClass Sizes:")
        class_size_cols = [col for col in df.columns if col.startswith("class_size_")]
        if class_size_cols:
            class_df = df[["dataset", "seed", "experiment", "method"] + class_size_cols]
            print(
                tabulate(
                    class_df, headers="keys", tablefmt="fancy_grid", showindex=False
                )
            )

        print("\nGroup Metrics:")
        group_cols = [col for col in df.columns if "group" in col.lower()]
        if group_cols:
            group_df = df[["dataset", "seed", "experiment", "method"] + group_cols]
            print(
                tabulate(
                    group_df, headers="keys", tablefmt="fancy_grid", showindex=False
                )
            )

        print("\nCoverage Confusion:")
        confusion_cols = [col for col in df.columns if "confusion" in col.lower()]
        if confusion_cols:
            confusion_df = df[
                ["dataset", "seed", "experiment", "method"] + confusion_cols
            ]
            print(
                tabulate(
                    confusion_df, headers="keys", tablefmt="fancy_grid", showindex=False
                )
            )


if __name__ == "__main__":
    main()
