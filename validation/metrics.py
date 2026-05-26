import pandas as pd
from sklearn.metrics import cohen_kappa_score


def work_related_metrics(
    y_true: pd.Series, y_pred: pd.Series, pos_label: str = "Yes"
) -> dict:
    """
    Compute accuracy, FPR, TPR, and Cohen's kappa for work-related classification.
    Each label can be either "Yes", "Maybe" and "No"; for FPR and TPR, "Yes" is considered the positive class.
    For Cohen's kappa, all three classes are considered.

    :param y_true: Series of ground truth labels.
    :param y_pred: Series of model predictions.
    :param pos_label: The string that represents the positive class.
    :return: Dict with keys accuracy, fpr, tpr, cohen_kappa.
    """
    tp = sum((y_pred == pos_label) & (y_true == pos_label))
    tn = sum((y_pred != pos_label) & (y_true != pos_label))
    fp = sum((y_pred == pos_label) & (y_true != pos_label))
    fn = sum((y_pred != pos_label) & (y_true == pos_label))
    total = len(y_true)

    return {
        "accuracy": (tp + tn) / total if total > 0 else 0,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def agreement_rate(df: pd.DataFrame, column: str, value: str = "Yes") -> float:
    """
    Compute the fraction of rows where column equals value.

    :param df: Input DataFrame.
    :param column: Column to evaluate.
    :param value: Target value to count as agreement.
    :return: Agreement rate as a float between 0 and 1.
    """
    return len(df[df[column] == value]) / len(df) if len(df) > 0 else 0.0


def print_metrics(metrics: dict) -> None:
    """
    Pretty-print a metrics dict.

    :param metrics: Dict of metric name → value.
    """
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
