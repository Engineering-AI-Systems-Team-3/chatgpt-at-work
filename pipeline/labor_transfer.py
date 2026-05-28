import json
import ast
import pandas as pd
import functools
import concurrent.futures
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from utilities import (
    MAX_WORKERS,
    LABOR_TRANSFER_BATCH_INPUT_FILE,
    LABOR_TRANSFER_BATCH_OUTPUT_FILE,
    RateLimiter,
    submit_and_retrieve,
    parallelize_llm_call,
    ExecutionMode,
    format_conversation,
    RATE_LIMIT,
)
from prompts import labor_transfer


def parse_labor_transfer_batch_results(output_path: Path) -> dict:
    """
    Parse batch output for labor transfer classification results.
    Extracts LT0/LT1/LT2 labels from model responses.
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
            results[custom_id] = _parse_labor_transfer_content(content)

    return results


def _parse_labor_transfer_content(content: str):
    """
    Helper function to parse the actual LLM string response.
    Can be used by both direct and batch execution parsing.
    """
    try:
        parsed_content = json.loads(content)
        return parsed_content
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"Error parsing: {e}")

    # Fallback in case the model returns text instead of JSON
    if "LT0" in content:
        return "LT0"
    elif "LT1" in content:
        return "LT1"
    elif "LT2" in content:
        return "LT2"

    return None


def _clean_messages_payload(
    messages_list: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Clean the messages payload to ensure all content is UTF-8 encoded strings.

    :param messages_list: List of message dictionaries with 'role' and 'content'.
    :return: Cleaned list of message dictionaries with UTF-8 encoded content.
    """
    cleaned = []
    for msg in messages_list:
        safe_content = str(msg["content"]).encode("utf-8", "replace").decode("utf-8")
        cleaned.append({"role": msg["role"], "content": safe_content})
    return cleaned


def _direct_execution(client: OpenAI, df: pd.DataFrame) -> list:
    """Run labor transfer questions via parallel direct LLM calls."""
    formatted_messages = []

    for _, row in df.iterrows():
        clean_conversation = format_conversation(row["conversation"])

        messages = labor_transfer(
            title=row["job_title"],
            task=row["selected_task"],
            conversation=clean_conversation,
        )
        clean_messages = _clean_messages_payload(messages_list=messages)
        formatted_messages.append(clean_messages)

    rate_limiter = RateLimiter(rate=RATE_LIMIT)  # requests per minute
    llm_call_func = functools.partial(
        parallelize_llm_call, client=client, rate_limiter=rate_limiter
    )

    print(
        f"Running {len(formatted_messages)} direct Labor Transfer calls via OpenRouter..."
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        responses_raw = list(
            tqdm(
                executor.map(llm_call_func, formatted_messages),
                total=len(formatted_messages),
                desc="Labor Transfer",
            )
        )

    # Use the same exact parsing strategy as batch processing
    return [_parse_labor_transfer_content(res) for res in responses_raw]


def expand_labor_transfer_labels(
    df: pd.DataFrame, label_column: str = "labor_transfer"
) -> pd.DataFrame:
    """
    Expand the labor transfer JSON column into individual columns
    (interaction_type, task_match, label, lt1_reason, transferred_from,
    transferred_from_other, rationale, confidence).

    :param df: DataFrame containing a column with labor transfer JSON strings.
    :param label_column: Name of the column holding the JSON strings.
    :return: DataFrame with the JSON fields expanded into separate columns,
             with the original label_column dropped.
    """
    fields = [
        "interaction_type",
        "task_match",
        "label",
        "lt1_reason",
        "transferred_from",
        "transferred_from_other",
        "rationale",
        "confidence",
    ]
    rows = []
    for _, row in df.iterrows():
        raw = row[label_column]
        try:
            data = ast.literal_eval(raw) if isinstance(raw, str) else raw
            rows.append({field: data.get(field) for field in fields})
        except Exception:
            rows.append({field: None for field in fields})

    expanded = pd.DataFrame(rows)
    result = pd.concat([df.reset_index(drop=True), expanded], axis=1)
    return result.drop(columns=[label_column])


def _batch_execution(client: OpenAI, df: pd.DataFrame) -> list:
    """Run labor transfer questions via OpenAI batch job."""
    formatted_messages = []
    word_id = "labor_transfer"

    for _, row in df.iterrows():
        clean_conversation = format_conversation(row["conversation"])
        messages = labor_transfer(
            title=row["job_title"],
            task=row["selected_task"],
            conversation=clean_conversation,
        )
        clean_messages = _clean_messages_payload(messages_list=messages)
        formatted_messages.append(clean_messages)

    results = submit_and_retrieve(
        client=client,
        messages=formatted_messages,
        batch_input_file=LABOR_TRANSFER_BATCH_INPUT_FILE,
        batch_output_file=LABOR_TRANSFER_BATCH_OUTPUT_FILE,
        word_id=word_id,
        parse_func=parse_labor_transfer_batch_results,
    )

    return [
        (
            results.get(f"{word_id}_{i}", "Unknown")
            if results.get(f"{word_id}_{i}") is not None
            else "Unknown"
        )
        for i in range(len(formatted_messages))
    ]


def analyze_labor_transfer(
    client: OpenAI, df: pd.DataFrame, execution_mode: ExecutionMode
) -> list:
    """
    Analyze labor transfer for each conversation.

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
