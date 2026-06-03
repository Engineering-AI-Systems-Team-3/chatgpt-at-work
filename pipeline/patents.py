import json
import ast
from typing import Optional
import pandas as pd
import functools
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from utilities import (
    MAX_WORKERS,
    PATENTS_BATCH_INPUT_FILE,
    PATENTS_BATCH_OUTPUT_FILE,
    RateLimiter,
    submit_and_retrieve,
    parallelize_llm_call,
    ExecutionMode,
    RATE_LIMIT,
)
from prompts import patents_auto_aug


def parse_patents_batch_results(output_path: Path) -> dict:
    """
    Parse batch output for patents classification results.
    Extracts E0/E1/E2 labels from model responses.
    """
    results = {}
    with open(output_path, "r") as f:
        for line in f:
            record = json.loads(line)
            custom_id = record["custom_id"]

            if record.get("error") is not None:
                results[custom_id] = None
                continue

            content = record["response"]["body"]["choices"][0]["message"]["content"]
            results[custom_id] = _parse_patents_content(content)

    return results


def _extract_label_from_response(response: str) -> Optional[str]:
    """
    Helper function to extract E0/E1/E2 label from a direct LLM response string.
    """
    try:
        parsed_response = ast.literal_eval(response)
        if isinstance(parsed_response, dict) and "label" in parsed_response:
            return parsed_response["label"]
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing response: {e}")
    return None


def _parse_patents_content(content: str) -> Optional[str]:
    """
    Helper function to parse the actual LLM string response.
    Can be used by both direct and batch execution parsing.
    """
    try:
        parsed_content = json.loads(content)
        result = _extract_label_from_response(parsed_content)
        if result in {"E0", "E1", "E2"}:
            return result
        else:
            print(f"Unexpected label in parsed content: {result}")

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"Error parsing: {e}")

    # Fallback in case the model returns text instead of JSON
    if "E0" in content:
        return "E0"
    elif "E1" in content:
        return "E1"
    elif "E2" in content:
        return "E2"

    return None


def _direct_execution(client: OpenAI, df: pd.DataFrame) -> list:
    """Run patents questions via parallel direct LLM calls."""
    formatted_messages = []

    for _, row in df.iterrows():
        messages = patents_auto_aug(
            abstract=row["abstract"],
            task=row["task"],
        )
        formatted_messages.append(messages)

    rate_limiter = RateLimiter(rate=RATE_LIMIT)  # requests per minute
    llm_call_func = functools.partial(
        parallelize_llm_call, client=client, rate_limiter=rate_limiter
    )

    print(f"Running {len(formatted_messages)} direct Patents calls via OpenRouter...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        responses_raw = list(
            tqdm(
                executor.map(llm_call_func, formatted_messages),
                total=len(formatted_messages),
                desc="Patents",
            )
        )

    return [_parse_patents_content(res) for res in responses_raw]


def _batch_execution(client: OpenAI, df: pd.DataFrame) -> list:
    """Run patents questions via OpenAI batch job."""
    formatted_messages = []
    word_id = "patents"

    for _, row in df.iterrows():
        messages = patents_auto_aug(
            abstract=row["abstract"],
            task=row["task"],
        )
        formatted_messages.append(messages)

    results = submit_and_retrieve(
        client=client,
        messages=formatted_messages,
        batch_input_file=PATENTS_BATCH_INPUT_FILE,
        batch_output_file=PATENTS_BATCH_OUTPUT_FILE,
        word_id=word_id,
        parse_func=parse_patents_batch_results,
    )

    return [
        (
            results.get(f"{word_id}_{i}", "Unknown")
            if results.get(f"{word_id}_{i}") is not None
            else "Unknown"
        )
        for i in range(len(formatted_messages))
    ]


def analyze_patents(
    client: OpenAI, df: pd.DataFrame, execution_mode: ExecutionMode
) -> list:
    """
    Analyze patents for Automation/Augmentation.

    :param client: OpenAI client instance.
    :param df: DataFrame with columns [conversation, task, title].
    :param execution_mode: ExecutionMode enum to pick between BATCH or DIRECT.
    :return: List of parsed dictionaries/strings, one per row in df.
    """
    if execution_mode == ExecutionMode.DIRECT:
        return _direct_execution(client=client, df=df)
    elif execution_mode == ExecutionMode.BATCH:
        return _batch_execution(client=client, df=df)
    else:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")
