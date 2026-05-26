import json
import functools
import concurrent.futures
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path
from utilities import (
    MAX_WORKERS,
    INPUT_BATCHES_DIR,
    OUTPUT_BATCHES_DIR,
    RateLimiter,
    submit_and_retrieve,
    strip_messages,
    format_conversation,
    parallelize_llm_call,
    ExecutionMode,
    RateLimiter,
)
from prompts import work_related


def parse_work_related_batch_results(output_path: Path) -> dict:
    """
    Parse the batch output file for work-related classification results.

    :param output_path: Path to the JSONL batch output file.
    :return: Dictionary mapping custom_id to "Yes", "Maybe", "No", or "ERROR".
    """
    results = {}
    with open(output_path, "r") as f:
        for line in f:
            record = json.loads(line)

            if record.get("error") is not None:
                results[record["custom_id"]] = "ERROR"
                continue

            custom_id = record["custom_id"]
            content = record["response"]["body"]["choices"][0]["message"]["content"]
            answer = "Unknown"

            try:
                parsed_content = json.loads(content)
                if "answer" in parsed_content:
                    val = parsed_content["answer"]
                    answer = _parse_work_related_response(val)
            except json.JSONDecodeError:
                answer = "Unknown"

            if answer == "Unknown":
                print(
                    f"Unexpected response for {custom_id}. Complete content: {content}"
                )

            results[custom_id] = answer

    return results


def _parse_work_related_direct_results(content: str) -> str:
    """
    Parse the direct response content for work-related classification results.

    :param content: The raw content string from the model's response.
    :return: "Yes", "Maybe", "No", or "Unknown" based on the parsed content.
    """
    try:
        parsed = json.loads(content)
        return _parse_work_related_response(parsed.get("answer"))
    except (json.JSONDecodeError, AttributeError):
        return "Unknown"


def _parse_work_related_response(val) -> str:
    """
    Shared mapping logic for both BATCH and DIRECT modes.

    :param val: The value to parse, expected to be 2/"2", 1/"1", or 0/"0".
    :return: "Yes", "Maybe", "No", or "Unknown" based on the value.
    """
    if val in (2, "2"):
        return "Yes"
    if val in (1, "1"):
        return "Maybe"
    if val in (0, "0"):
        return "No"
    return "Unknown"


def filter_work_conversations(
    client: OpenAI,
    conversations: pd.DataFrame,
    path: Path,
    execution_mode: ExecutionMode,
) -> list:
    """
    Filter conversations to determine which ones are work-related using the GPT model.

    :param conversations: DataFrame containing a 'conversation' column.
    :param path: Path used to derive batch file names and save output.
    :return: List of "Yes"/"Maybe"/"No" answers, one per row in conversations.
    """
    formatted_messages = []
    word_id = path.stem.split("_")[0]
    batch_input_file = INPUT_BATCHES_DIR / f"{word_id}_.jsonl"
    batch_output_file = OUTPUT_BATCHES_DIR / f"{word_id}_.jsonl"

    for _, row in conversations.iterrows():
        raw_conversation = strip_messages(row["conversation"])
        conversation = format_conversation(conversation=raw_conversation)
        messages = work_related(conversation=conversation)
        formatted_messages.append(messages)

    if execution_mode == ExecutionMode.BATCH:
        results = submit_and_retrieve(
            client=client,
            messages=formatted_messages,
            batch_input_file=batch_input_file,
            batch_output_file=batch_output_file,
            word_id=word_id,
            parse_func=parse_work_related_batch_results,
        )
        answers = [
            results.get(f"{word_id}_{i}") for i in range(len(formatted_messages))
        ]
    else:
        rate_limiter = RateLimiter(
            rate=20 / 60
        )  # free tier supports 20 requests per minute
        llm_call_func = functools.partial(
            parallelize_llm_call, client=client, rate_limiter=rate_limiter
        )

        print(f"Running {len(formatted_messages)} direct Work Related calls...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            raw_responses = list(
                tqdm(
                    executor.map(llm_call_func, formatted_messages),
                    total=len(formatted_messages),
                    desc="Work Related",
                )
            )

        answers = [_parse_work_related_direct_results(r) for r in raw_responses]

    return answers
