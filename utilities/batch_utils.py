import json
import os
import time
from pathlib import Path
from openai import InternalServerError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from utilities.config import BATCH_SIZE, MODEL_ID, WAIT_TIME


def _create_batch_file(
    filename: Path,
    messages: list,
    word_id: str,
    id_offset: int = 0,
    model_id: str = MODEL_ID,
) -> Path:
    """
    Write a JSONL batch file ready for the OpenAI Batch API.

    :param filename: Destination path for the batch file.
    :param messages: List of message lists, one per request.
    :param word_id: Prefix used to build unique custom_id values.
    :param id_offset: Integer offset added to the row index in custom_id.
    :param model_id: Model to target inside each request body.
    :return: The path that was written.
    """
    os.makedirs(filename.parent, exist_ok=True)

    with open(filename, "w+") as f:
        for idx in range(len(messages)):
            request = {
                "custom_id": f"{word_id}_{idx + id_offset}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_id,
                    "messages": messages[idx],
                    "response_format": {"type": "json_object"},
                },
            }
            f.write(json.dumps(request) + "\n")

    return filename


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(InternalServerError),
)
def _submit_batch_file(client, file_name: Path) -> str:
    """
    Upload a batch file to OpenAI and create a batch job.

    :param client: Initialised OpenAI client.
    :param file_name: Path to the JSONL batch file to upload.
    :return: The batch job ID.
    """
    batch_file = client.files.create(
        file=open(file_name, "rb"),
        purpose="batch",
    )

    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    return batch_job.id


def _retrieve_batch_job_results(
    client,
    batch_job_id: str,
    output_path: Path,
    wait_time: int = WAIT_TIME,
) -> None:
    """
    Poll until a batch job finishes, then download the output to *output_path*.

    :param client: Initialised OpenAI client.
    :param batch_job_id: ID returned by :func:`submit_batch_file`.
    :param output_path: Where to write the raw JSONL output.
    :param wait_time: Seconds to sleep between status checks.
    :raises Exception: If the job ends in a non-completed terminal state.
    :raises ValueError: If the job completes but has no output file.
    """
    terminal_states = ["completed", "failed", "cancelled", "expired", "cancelling"]

    while True:
        batch_job = client.batches.retrieve(batch_job_id)

        if batch_job.status in terminal_states:
            if batch_job.status != "completed":
                raise Exception(
                    f"Batch job {batch_job_id} terminated with status: {batch_job.status}."
                )

            if batch_job.error_file_id:
                errors = client.files.content(batch_job.error_file_id).content
                print(
                    f"Warning: Batch {batch_job_id} had partial errors:\n"
                    f"{errors.decode('utf-8')[:1000]}"
                )

            if not batch_job.output_file_id:
                raise ValueError(
                    f"Batch {batch_job_id} completed but has no output file."
                )

            result = client.files.content(batch_job.output_file_id).content
            os.makedirs(output_path.parent, exist_ok=True)
            with open(output_path, "wb+") as file:
                file.write(result)
            break

        else:
            print(
                f"Batch job {batch_job_id} is still in status {batch_job.status}. "
                f"Waiting for {wait_time} seconds before retrying..."
            )
            time.sleep(wait_time)


def submit_and_retrieve(
    client,
    messages: list,
    batch_input_file: Path,
    batch_output_file: Path,
    word_id: str,
    parse_func,
    batch_size: int = BATCH_SIZE,
) -> dict[str, str]:
    """
    Submit *messages* in chunks, skip chunks whose output already exists,
    and return a combined results dict.

    :param client: Initialised OpenAI client.
    :param messages: Full list of message lists to process.
    :param batch_input_file: Template path for chunk batch files (stem is suffixed with ``_<i>``).
    :param batch_output_file: Template path for chunk output files (stem is suffixed with ``_<i>``).
    :param word_id: Prefix for custom_id values inside each batch file.
    :param parse_func: Callable that receives a chunk output path and returns a dict.
    :param batch_size: Maximum number of messages per batch chunk.
    :return: Merged dict of all parsed results across chunks.
    """
    chunks = [messages[i : i + batch_size] for i in range(0, len(messages), batch_size)]
    already_existing_chunks: list[int] = []

    print(
        f"Submitting messages for {batch_output_file.stem}: "
        f"total {len(messages)} messages in {len(chunks)} batches "
        f"of up to {batch_size} messages each"
    )

    all_results: dict[str, str] = {}

    for idx in range(len(chunks)):
        chunk_output_file = batch_output_file.with_stem(
            f"{batch_output_file.stem}_{idx}"
        )
        if chunk_output_file.exists():
            print(
                f"Skipping batch {idx} ({len(chunks[idx])} requests): "
                f"{chunk_output_file} already exists"
            )
            all_results.update(parse_func(chunk_output_file))
            already_existing_chunks.append(idx)

    print(
        f"{already_existing_chunks} batches already exist. "
        f"Skipping those batches and retrieving results from existing files."
    )

    if len(already_existing_chunks) == len(chunks):
        return all_results

    batch_jobs: list[tuple[str, Path]] = []
    for i, chunk in enumerate(chunks):
        if i in already_existing_chunks:
            continue
        chunk_batch_file = batch_input_file.with_stem(f"{batch_input_file.stem}_{i}")
        chunk_output_file = batch_output_file.with_stem(f"{batch_output_file.stem}_{i}")
        _create_batch_file(
            filename=chunk_batch_file,
            messages=chunk,
            word_id=word_id,
            id_offset=i * batch_size,
        )
        job_id = _submit_batch_file(client=client, file_name=chunk_batch_file)
        batch_jobs.append((job_id, chunk_output_file))
        print(f"Submitted batch {i} ({len(chunk)} requests): {job_id}")

    for job_id, chunk_output_file in batch_jobs:
        _retrieve_batch_job_results(
            client=client, batch_job_id=job_id, output_path=chunk_output_file
        )
        all_results.update(parse_func(chunk_output_file))

    return all_results
