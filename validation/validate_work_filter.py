import ast
import pandas as pd
from pathlib import Path

from pipeline.work_filter import filter_work_conversations
from validation.metrics import binary_classification_metrics, print_metrics


def load_labeled_sample(path: Path) -> pd.DataFrame:
    """
    Load the manually labeled sample and parse the conversation column.

    :param path: Path to the labeled CSV file.
    :return: DataFrame with the conversation column parsed from string to list.
    """
    df = pd.read_csv(path)
    df["conversation"] = df["conversation"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    return df


def run(
    labeled_path: Path,
    output_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Run work-related classification on the labeled sample and compute metrics.

    Saves or loads results from output_path to avoid redundant API calls.

    :param labeled_path: Path to the labeled CSV (must have 'is_work_related_human' column).
    :param output_path: Path to save/load model predictions.
    :return: Tuple of (annotated DataFrame, metrics dict).
    """
    if not output_path.exists():
        df = load_labeled_sample(labeled_path)
        answers = filter_work_conversations(
            conversations=df,
            path=output_path,
        )
        df["is_work_related_model"] = answers
        df.to_csv(output_path, index=False)
    else:
        df = pd.read_csv(output_path)

    metrics = binary_classification_metrics(
        y_true=df["is_work_related_human"],
        y_pred=df["is_work_related_model"],
    )
    print_metrics(metrics)
    print()
    print(
        pd.crosstab(
            df["is_work_related_human"],
            df["is_work_related_model"],
            rownames=["Actual"],
            colnames=["Predicted"],
        )
    )

    return df, metrics
