import pandas as pd
from pathlib import Path

from pipeline.task_mapper import map_conversation_to_task_direct, filter_task_mappings
from validation.metrics import agreement_rate


def run(
    labeled_df: pd.DataFrame,
    hierarchy: dict,
    output_path: Path,
    consensus_output_path: Path,
    random_seed: int = 42,
    sample_size: int = 150,
) -> tuple[pd.DataFrame, float]:
    """
    Run task mapping on a sample of the labeled dataset and compute agreement.

    Saves intermediate and consensus-filtered results to disk.

    :param labeled_df: Labeled DataFrame (output of work filter validation,
                       filtered to is_work_related_model == "Yes").
    :param hierarchy: Hierarchy dict (same format used in the main pipeline).
    :param output_path: Path to save/load the raw task mapping output.
    :param consensus_output_path: Path to save the consensus-filtered output.
    :param random_seed: Seed for reproducibility.
    :param sample_size: Number of rows to sample for validation.
    :return: Tuple of (validated DataFrame with job_title/answer_text columns,
             task plausibility agreement rate as float 0–1).
    """
    sample_df = labeled_df.sample(sample_size, random_state=random_seed)

    if not output_path.exists():
        task_mapping = map_conversation_to_task_direct(
            conversations=sample_df[["conversation"]],
            tasks=hierarchy,
            path=output_path,
        )
        validated_df = pd.merge(
            left=sample_df,
            right=task_mapping,
            on="conversation",
            how="inner",
        )
        validated_df.to_csv(output_path, index=False)
    else:
        validated_df = pd.read_csv(output_path)

    # Apply consensus filter and split job_title from task text
    validated_df = filter_task_mappings(df=validated_df, column_name="level_0_task")
    validated_df["job_title"] = validated_df["level_0_task"].apply(
        lambda x: x.split(":")[0] if pd.notnull(x) else None
    )
    validated_df["answer_text"] = validated_df["level_0_task"].apply(
        lambda x: x.split(":")[1] if pd.notnull(x) else None
    )

    validated_df.to_csv(consensus_output_path, index=False)
    print(f"Consensus-filtered rows: {validated_df.shape[0]}")

    # Compute agreement on the 'task_plausible' column (human annotation)
    if "task_plausible" in validated_df.columns:
        rate = agreement_rate(df=validated_df, column="task_plausible", value="Yes")
        print(f"Task plausibility agreement: {rate:.2%}")
        return validated_df, rate

    print("Column 'task_plausible' not found — skipping agreement calculation.")
    return validated_df, None
