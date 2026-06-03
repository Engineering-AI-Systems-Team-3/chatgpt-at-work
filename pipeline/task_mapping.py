import json
import ast
import re
import concurrent.futures
import functools
import pandas as pd
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from prompts import occupation_mapping, task_mapping
from utilities import (
    BATCH_SIZE,
    MAX_WORKERS,
    PROFESSION_MAPPING_BATCH_INPUT_FILE,
    PROFESSION_MAPPING_BATCH_OUTPUT_FILE,
    TASK_MAPPING_BATCH_INPUT_FILE,
    TASK_MAPPING_BATCH_OUTPUT_FILE,
    ExecutionMode,
    RateLimiter,
    submit_and_retrieve,
    format_options,
    parallelize_llm_call,
    RATE_LIMIT,
)


def parse_task_mapping_batch_results(
    output_path: Path, n_batches: int
) -> dict[str, str | list[str]]:
    """
    Parse batch output files for task mapping results across multiple batch chunks.

    :param output_path: Base path for batch output files (suffixed with _{i}).
    :param n_batches: Number of batch chunks to read.
    :return: Dictionary mapping custom_id to the parsed answer string.
    """
    results = {}
    for i in range(n_batches):
        current_path = output_path.with_stem(f"{output_path.stem}_{i}")
        if not current_path.exists():
            continue
        with open(current_path, "r") as f:
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
                        answer = parsed_content["answer"]
                    elif (
                        "command" in parsed_content
                        and "args" in parsed_content["command"]
                    ):
                        args = parsed_content["command"]["args"]
                        answer = (
                            args.get("message", "Unknown")
                            if isinstance(args, dict)
                            else "Unknown"
                        )
                except (KeyError, ValueError, json.JSONDecodeError):
                    print(
                        f"Wrong JSON format for {custom_id}, content: {content[:100]}"
                    )

                results[custom_id] = answer

    return results


def check_consensus(items: list[str] | str) -> str | None:
    """
    Determine the majority profession from a list of "profession:task" strings.
    Returns the profession:task pair if any profession appears in at least half
    of the items, otherwise None.

    :param items: List (or stringified list) of "profession:task" strings.
    :return: Winning "profession:task" string, or None if no consensus.
    """
    profession_stats: dict[str, dict] = {}
    try:
        if isinstance(items, str):
            items_list = ast.literal_eval(items)
        else:
            items_list = items

        for item in items_list:
            profession, task = item.split(":")
            stats = profession_stats.setdefault(profession, {"count": 0, "task": task})
            stats["count"] += 1

        max_count = max(s["count"] for s in profession_stats.values())
        if max_count >= len(items_list) / 2:
            for profession, stats in profession_stats.items():
                if stats["count"] == max_count:
                    return f"{profession}:{stats['task']}"
    except Exception as e:
        # print(f"Error processing items: {items}: {e}")
        return None
    return None


def filter_task_mappings(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Keep only rows where a majority profession consensus exists.

    :param df: DataFrame containing task mapping results.
    :param column_name: Column holding the list of profession:task candidates.
    :return: Filtered DataFrame with the consensus result in column_name.
    """
    filtered_df = df.copy()
    filtered_df[column_name] = filtered_df[column_name].apply(check_consensus)
    return filtered_df.dropna(subset=[column_name])


def _parse_direct_responses(raw_responses: list[str]) -> list[list[str]]:
    """
    Parses a string response into a clean list of strings. It can handle both profession-task pairs and standalone
    tasks, depending on the format of the input string.

    :param raw_responses: List of raw string responses from the model.
    :return: List of lists of strings, where each inner list contains the parsed tasks or profession-task pairs for a
    single conversation.
    """
    parsed_results = []
    for raw_content in raw_responses:
        ans = ["Unknown"]
        if raw_content:
            try:
                parsed_content = json.loads(raw_content)
                if "answer" in parsed_content:
                    ans_content = parsed_content["answer"]
                    if isinstance(ans_content, str):
                        try:
                            ans = (
                                ast.literal_eval(ans_content)
                                if ans_content.startswith("[")
                                else [ans_content.strip()]
                            )
                        except (ValueError, SyntaxError):
                            ans = [ans_content.strip()]
                    elif isinstance(ans_content, list):
                        ans = [str(a) for a in ans_content]
            except (KeyError, ValueError, json.JSONDecodeError, TypeError):
                pass
        parsed_results.append(ans)
    return parsed_results


def _direct_execution(
    client: OpenAI,
    conversations: pd.DataFrame,
    tasks: pd.DataFrame,
    path: Path,
    n_options: int,
) -> pd.DataFrame:
    """
    Execute the profession and task mapping in DIRECT mode, making individual API calls for each conversation.

    :param client: OpenAI client instance.
    :param conversations: DataFrame with a 'conversation' column.
    :param tasks: DataFrame containing task information.
    :param path: Path to save the resulting CSV.
    :param n_options: Number of options to include in the prompt.
    :return: DataFrame with columns [conversation, professions, tasks].
    """

    rate_limiter = RateLimiter(rate=RATE_LIMIT)
    conversations_list = conversations["conversation"].tolist()

    print(f"Conversations to process: {(conversations_list)}")

    llm_call_func = functools.partial(
        parallelize_llm_call, client=client, rate_limiter=rate_limiter
    )

    professions = tasks["Title"].unique().tolist()
    options_str = format_options(tasks=professions)
    formatted_messages = [
        occupation_mapping(conversation=c, options_str=options_str, n_options=n_options)
        for c in conversations_list
    ]

    print(formatted_messages[0:5])

    print(
        f"Running {len(formatted_messages)} direct Profession mapping calls via OpenRouter..."
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        profession_responses_raw = list(
            tqdm(
                executor.map(llm_call_func, formatted_messages),
                total=len(formatted_messages),
                desc="Profession Mapping",
            )
        )

    profession_responses = _parse_direct_responses(
        raw_responses=profession_responses_raw
    )

    # Task assignment
    task_messages = []
    for conv, profs in zip(conversations_list, profession_responses):
        all_tasks = []
        for profession in profs:
            filtered_tasks_df = tasks[tasks["Title"] == profession]
            if not filtered_tasks_df.empty:
                for task in filtered_tasks_df["Task"].tolist():
                    entry = f"{profession}:{task}"
                    all_tasks.append(entry)

        task_options_str = format_options(tasks=all_tasks)
        task_messages.append(
            task_mapping(
                conversation=conv, options_str=task_options_str, n_options=n_options
            )
        )

    print(f"Running {len(task_messages)} direct Task mapping calls via OpenRouter...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        task_responses_raw = list(
            tqdm(
                executor.map(llm_call_func, task_messages),
                total=len(task_messages),
                desc="Task Mapping",
            )
        )

    task_responses = _parse_direct_responses(raw_responses=task_responses_raw)

    result = pd.DataFrame(
        {
            "conversation": conversations_list,
            "professions": profession_responses,
            "tasks": task_responses,
        }
    )
    result.to_csv(path, index=False)
    return result


def _batch_execution(
    client: OpenAI,
    conversations: pd.DataFrame,
    tasks: pd.DataFrame,
    path: Path,
    n_options: int,
) -> pd.DataFrame:
    """
    Map conversations to tasks via a two-step process: first assign professions, then assign tasks based on the
    identified professions.

    :param client: OpenAI client instance.
    :param conversations: DataFrame with a 'conversation' column.
    :param tasks: DataFrame containing task information.
    :param path: Path to save the resulting CSV.
    :return: DataFrame with columns [conversation, professions, tasks].
    """
    conversations_list = conversations["conversation"].tolist()
    n_batches = len(conversations_list) // BATCH_SIZE + 1
    professions_word_id = "profession"
    tasks_word_id = "task"

    # Profession assignment
    formatted_messages = []
    professions = tasks["Title"].unique().tolist()
    options_str = format_options(tasks=professions)
    for conversation in conversations_list:
        prompt = occupation_mapping(
            conversation=conversation, options_str=options_str, n_options=n_options
        )
        formatted_messages.append(prompt)

    results = submit_and_retrieve(
        client=client,
        messages=formatted_messages,
        batch_input_file=PROFESSION_MAPPING_BATCH_INPUT_FILE,
        batch_output_file=PROFESSION_MAPPING_BATCH_OUTPUT_FILE,
        word_id=professions_word_id,
        parse_func=lambda _: parse_task_mapping_batch_results(
            output_path=PROFESSION_MAPPING_BATCH_OUTPUT_FILE, n_batches=n_batches
        ),
    )
    profession_responses = []
    for i in range(len(conversations_list)):
        ans = results.get(f"{professions_word_id}_{i}", ["Unknown"])
        if isinstance(ans, str):
            try:
                ans = ast.literal_eval(ans) if ans.startswith("[") else [ans.strip()]
            except:
                ans = [ans.strip()]
        profession_responses.append(ans)

    # Task assignment
    formatted_messages = []
    for conv, professions in zip(conversations_list, profession_responses):

        all_tasks = []
        for profession in professions:
            filtered_tasks = tasks[tasks["Title"] == profession]["Task"].tolist()
            all_tasks.extend(filtered_tasks)

        options_str = format_options(tasks=all_tasks)
        prompt = task_mapping(
            conversation=conv, options_str=options_str, n_options=n_options
        )
        formatted_messages.append(prompt)

    results = submit_and_retrieve(
        client=client,
        messages=formatted_messages,
        batch_input_file=TASK_MAPPING_BATCH_INPUT_FILE,
        batch_output_file=TASK_MAPPING_BATCH_OUTPUT_FILE,
        word_id=tasks_word_id,
        parse_func=lambda _: parse_task_mapping_batch_results(
            output_path=TASK_MAPPING_BATCH_OUTPUT_FILE, n_batches=n_batches
        ),
    )
    task_responses = []
    for i in range(len(conversations_list)):
        ans = results.get(f"{tasks_word_id}_{i}", ["Unknown"])
        if isinstance(ans, str):
            try:
                ans = ast.literal_eval(ans) if ans.startswith("[") else [ans.strip()]
            except:
                ans = [ans.strip()]
        task_responses.append(ans)

    result = pd.DataFrame(
        {
            "conversation": conversations_list,
            "professions": profession_responses,
            "tasks": task_responses,
        }
    )
    result.to_csv(path, index=False)
    return result


def map_conversation_to_task(
    client: OpenAI,
    conversations: pd.DataFrame,
    tasks: pd.DataFrame,
    path: Path,
    execution_mode: ExecutionMode,
) -> pd.DataFrame:
    """
    Map conversations to tasks via a two-step process: first assign professions, then assign tasks based on the
    identified professions.

    :param client: OpenAI client instance.
    :param conversations: DataFrame with a 'conversation' column.
    :param tasks: DataFrame containing task information.
    :param path: Path to save the resulting CSV.
    :return: DataFrame with columns [conversation, professions, tasks].
    """
    n_options = 5
    if execution_mode == ExecutionMode.DIRECT:
        return _direct_execution(
            client=client,
            conversations=conversations,
            tasks=tasks,
            path=path,
            n_options=n_options,
        )
    elif execution_mode == ExecutionMode.BATCH:
        return _batch_execution(
            client=client,
            conversations=conversations,
            tasks=tasks,
            path=path,
            n_options=n_options,
        )
    else:
        raise ValueError(f"Unsupported execution mode: {execution_mode}")
