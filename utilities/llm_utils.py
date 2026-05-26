import ast
import datetime
import json
import re
import time
import numpy as np
import pandas as pd
import random
from typing import cast
from openai import OpenAI, RateLimitError
from utilities.config import OPENROUTER_MODEL_ID
from utilities.RateLimiter import RateLimiter


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
        msg = cast(dict, message)
        role = msg["role"]
        content = msg["content"]
        formatted += f"{role}: {content}\n"
    return formatted


#  LLM call


def get_gpt_response(
    client,
    messages: list[dict[str, str]],
    model_id: str = OPENROUTER_MODEL_ID,
    max_retries: int = 5,
    base_wait: int = 1,
    max_wait: int = 30,
) -> str:
    """
    Call the chat completions endpoint with basic retry logic.

    Attempts to extract a clean answer a fenced JSON block, or a bare JSON object/array
    in that order.  Falls back to returning the raw stripped content.

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

            json_match = re.search(
                r"```(?:json)?\s*(.*?)\s*```", dirty_result, re.DOTALL
            )
            if json_match:
                return json_match.group(1).strip()

            bracket_match = re.search(r"(\[.*\]|\{.*\})", dirty_result, re.DOTALL)
            if bracket_match:
                return bracket_match.group(1).strip()

            return dirty_result

        except RateLimitError as e:
            retry_after = None
            if hasattr(e, "response") and e.response is not None:
                retry_after = e.response.headers.get("Retry-After")

            if retry_after:
                wait = float(retry_after)
            else:
                wait = min(base_wait * (2**attempt), max_wait)

            print(
                f"Rate limit hit (attempt {attempt + 1}). Retrying in {wait:.2f} seconds..."
            )

            jitter = random.uniform(0, wait * 0.2)
            time.sleep(wait + jitter)

    print(f"Failed after {max_retries} retries.")
    return ""


def parallelize_llm_call(
    messages: list[dict[str, str]], client: OpenAI, rate_limiter: RateLimiter
) -> str:
    """
    Thin wrapper around :func:`get_gpt_response` for use with
    ``concurrent.futures`` executors.

    :param client: Initialised OpenAI client.
    :param messages: Messages to send.
    :return: Model response string.
    """
    rate_limiter.acquire()
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


# TODO: format_last_level_options non serve più, mentre format_options deve gestire professioni e task


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
