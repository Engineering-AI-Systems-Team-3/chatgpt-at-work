import ast
import copy
import datetime
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from openai import RateLimitError

from utilities.config import OPENROUTER_MODEL_ID, WAIT_TIME


class RobustEncoder(json.JSONEncoder):
    """JSON encoder that handles common non-serialisable types gracefully."""

    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return str(obj)


# Prompt helpers


def get_messages(path: Path) -> list[dict[str, str]]:
    """
    Load a prompt template from a JSON file.

    :param path: Path to the JSON file containing a list of message dicts.
    :return: List of message dicts (``role`` / ``content`` pairs).
    """
    with open(path, "r") as f:
        return json.load(f)


def build_messages(template: list[dict[str, str]], **kwargs) -> list[dict[str, str]]:
    """
    Fill ``{placeholder}`` slots in a prompt template with concrete values.

    :param template: List of message dicts with optional ``{key}`` placeholders
        in their ``content`` field.
    :param kwargs: Key-value pairs mapping placeholder names to replacement strings.
    :return: Deep copy of the template with all placeholders replaced.
    """
    msgs = copy.deepcopy(template)
    for msg in msgs:
        for key, value in kwargs.items():
            if "{" + key + "}" in msg["content"]:
                msg["content"] = msg["content"].replace("{" + key + "}", value)
    return msgs


#  Conversation helpers


def strip_messages(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Remove all fields except ``role`` and ``content`` from WildChat messages.

    :param msgs: List of messages in the original WildChat format.
    :return: Stripped list suitable for feeding to an LLM.
    """
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


def format_conversation(conversation: list[dict[str, str]] | str) -> str:
    """
    Render a conversation as a plain-text string for use in prompts.

    :param conversation: Either a list of ``{role, content}`` dicts or a
        string representation of such a list (will be parsed with
        ``ast.literal_eval``).
    :return: Multi-line string with ``role: content`` on each line.
    """
    if isinstance(conversation, str):
        conversation = ast.literal_eval(conversation)

    formatted = ""
    for message in conversation:
        role = message["role"]
        content = message["content"]
        formatted += f"{role}: {content}\n"
    return formatted


#  LLM call


def get_gpt_response(
    client,
    messages: list[dict[str, str]],
    model_id: str = OPENROUTER_MODEL_ID,
    max_retries: int = 5,
    wait_time: int = WAIT_TIME,
) -> str:
    """
    Call the chat completions endpoint with basic retry logic.

    Attempts to extract a clean answer from ``<answer>`` tags, a fenced JSON
    block, or a bare JSON object/array in that order.  Falls back to returning
    the raw stripped content.

    :param client: Initialised OpenAI client.
    :param messages: Conversation history to send.
    :param model_id: Model to use.
    :param max_retries: Maximum number of attempts before giving up.
    :param wait_time: Base wait time (seconds) used in the rate-limit backoff.
    :return: Extracted answer string, or ``""`` if all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
            )

            if not hasattr(response, "choices") or not response.choices:
                print(f"Warning: Empty response from API (attempt {attempt + 1})")
                continue

            content = response.choices[0].message.content
            if content is None:
                print(
                    f"Warning: received null content from API (attempt {attempt + 1})"
                )
                continue

            dirty_result = content.strip()

            tag_match = re.search(r"<answer>(.*?)</answer>", dirty_result, re.DOTALL)
            if tag_match:
                return tag_match.group(1)

            json_match = re.search(
                r"```(?:json)?\s*(.*?)\s*```", dirty_result, re.DOTALL
            )
            if json_match:
                return json_match.group(1).strip()

            bracket_match = re.search(r"(\[.*\]|\{.*\})", dirty_result, re.DOTALL)
            if bracket_match:
                return bracket_match.group(1).strip()

            return dirty_result

        except RateLimitError:
            wait = wait_time**attempt
            print(
                f"Rate limited (429). Waiting {wait}s before retry "
                f"{attempt + 1}/{max_retries}..."
            )
            time.sleep(wait)

    print(f"Failed after {max_retries} retries.")
    return ""


def parallelize_llm_call(client, messages: list[dict[str, str]]) -> str:
    """
    Thin wrapper around :func:`get_gpt_response` for use with
    ``concurrent.futures`` executors.

    :param client: Initialised OpenAI client.
    :param messages: Messages to send.
    :return: Model response string.
    """
    return get_gpt_response(client=client, messages=messages)


#  DataFrame helpers


def format_tasks(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Normalise task names: strip whitespace, lower-case, remove punctuation.

    :param df: DataFrame containing task names.
    :param column_name: Name of the column to normalise (modified in-place).
    :return: The same DataFrame with the column normalised.
    """
    df[column_name] = (
        df[column_name].str.strip().str.lower().str.replace(r"[^\w\s]", "", regex=True)
    )
    return df


def format_options(tasks: list[str]) -> str:
    """
    Render a list of tasks as a newline-separated string for prompts.

    :param tasks: List of task names.
    :return: One task name per line.
    """
    return "".join(f"{task}\n" for task in tasks)


def format_last_level_options(professions: list[str], tasks: list[str]) -> str:
    """
    Render a list of profession-task pairs as a newline-separated string for prompts.

    :param professions: List of profession names.
    :param tasks: List of task names corresponding to the professions.
    :return: One profession-task pair per line, formatted as ``profession: task``.
    """
    options_str = ""
    for profession, task in zip(professions, tasks):
        options_str += f"<option><profession>{profession}</profession>\n<task>{task}</task></option>\n"
    return options_str
