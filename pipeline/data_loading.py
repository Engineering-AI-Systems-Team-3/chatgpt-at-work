from typing import cast

import pandas as pd
from pathlib import Path
from datasets import Dataset, load_dataset, load_from_disk

from utilities.config import (
    WILDCHAT_DATASET,
    LANGUAGE_FIELD,
    TARGET_LANGUAGE,
    SAMPLE_PERCENTAGE,
    RANDOM_SEED,
    WILDCHAT_FULL,
    WILDCHAT_ENGLISH,
)
from utilities.llm_utils import strip_messages


def load_wildchat(
    wildchat_full_path: Path = WILDCHAT_FULL,
    wildchat_english_path: Path = WILDCHAT_ENGLISH,
) -> Dataset:
    """
    Load the WildChat dataset, filtering for English conversations.
    Caches both the full and English-only versions to disk.

    :param wildchat_full_path: Path to cache the full WildChat dataset.
    :param wildchat_english_path: Path to cache the English-only subset.
    :return: HuggingFace Dataset containing English conversations.
    """
    if not wildchat_full_path.exists():
        wildchat_ds = cast(Dataset, load_dataset(path=WILDCHAT_DATASET, split="train"))
        wildchat_ds.save_to_disk(wildchat_full_path)

    if not wildchat_english_path.exists():
        wildchat_ds = cast(Dataset, load_from_disk(wildchat_full_path))
        english_rows = wildchat_ds.filter(
            lambda x: x[LANGUAGE_FIELD] == TARGET_LANGUAGE
        )
        english_rows.save_to_disk(wildchat_english_path)
    else:
        english_rows = cast(Dataset, load_from_disk(wildchat_english_path))

    return english_rows


def sample_conversations(
    dataset: Dataset,
    sample_percentage: float = SAMPLE_PERCENTAGE,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Sample a percentage of conversations from the dataset.

    :param dataset: HuggingFace Dataset to sample from.
    :param sample_percentage: Fraction of the dataset to sample.
    :param random_seed: Random seed for reproducibility.
    :return: Sampled DataFrame.
    """
    total_rows = len(dataset)
    sample_size = int(sample_percentage * total_rows)
    return cast(
        pd.DataFrame,
        dataset.shuffle(seed=random_seed).select(range(sample_size)).to_pandas(),
    )


def preprocess_conversations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess conversations: keep relevant columns, normalize message format,
    strip metadata from messages, and remove duplicates.

    :param df: Raw DataFrame from WildChat.
    :return: Cleaned DataFrame with columns [conversation, timestamp, country, state, hashed_ip].
    """
    columns_to_keep = ["conversation", "timestamp", "country", "state", "hashed_ip"]
    df = df[columns_to_keep].copy()

    df["conversation"] = df["conversation"].apply(
        lambda x: [dict(msg) for msg in x] if not isinstance(x, list) else x
    )
    df["conversation"] = df["conversation"].apply(
        lambda conv: strip_messages(conv) if isinstance(conv, list) else None
    )
    df.drop_duplicates(subset=["conversation"], inplace=True)
    return df
