import pandas as pd
from pathlib import Path

from pipeline.labor_transfer import analyze_labor_transfer, expand_labor_transfer_labels
from validation.metrics import agreement_rate


def run(
    validated_task_df: pd.DataFrame,
    output_path: Path,
    random_seed: int = 42,
    sample_size: int = 50,
) -> tuple[pd.DataFrame, dict]:
    """
    Run labor transfer analysis on a sample and compute label/task_match distributions.

    :param validated_task_df: DataFrame from task mapping validation
                              (must have columns: conversation, level_0_task, title).
    :param output_path: Path to save/load labor transfer results.
    :param random_seed: Seed for reproducibility.
    :param sample_size: Number of rows to sample.
    :return: Tuple of (expanded DataFrame with all LT fields,
             dict with label and task_match distributions).
    """
    sample_df = validated_task_df.sample(
        min(sample_size, len(validated_task_df)),
        random_state=random_seed,
    ).copy()

    if not output_path.exists():
        labor_transfer_labels = analyze_labor_transfer(df=sample_df)
        sample_df["labor_transfer"] = labor_transfer_labels
        sample_df.to_csv(output_path, index=False)
    else:
        sample_df = pd.read_csv(output_path)

    final_df = expand_labor_transfer_labels(df=sample_df, label_column="labor_transfer")

    distributions = {
        "label": final_df["label"].value_counts(normalize=True) * 100,
        "task_match": final_df["task_match"].value_counts(normalize=True) * 100,
    }

    print("Label distribution:")
    print(distributions["label"])
    print("\nTask match distribution:")
    print(distributions["task_match"])

    return final_df, distributions
