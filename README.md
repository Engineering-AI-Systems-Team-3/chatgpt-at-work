# chatgpt-at-work

To study how artificial intelligence may reshape work, Anthropic mapped Claude queries to occupational tasks to estimate which tasks are most exposed to automation by LLMs. It is not clear whether the same patterns hold for ChatGPT.

We test this by mapping millions of publicly available ChatGPT interaction logs [1] to tasks and skills in the Occupational Information Network (O\*NET), including detailed work activities (DWAs). We:

- map ChatGPT interactions to O\*NET tasks and DWAs;
- map the same interactions to the roles a chatbot can take (for example, friend, colleague, or trainer), using the role set in [2];
- evaluate both mappings with a human-in-the-loop protocol, with a focus on separating plausible automation targets from spurious matches; and
- release a dataset and analysis code so others can reproduce our results.

## References

[1] WildChat Datasets: 4.8M and 50M

[2] Frictionless Love: Associations Between AI Companion Roles and Behavioral Addiction

## Project structure

This repository maps ChatGPT interactions to O*NET tasks and includes code, data, and notebooks to reproduce the experiments and evaluations. Below is a short overview of the main files and directories so that new contributors can find the parts they need.

- **pyproject.toml**: Project metadata and dependency list (requires Python >= 3.12.3).
- **pipeline/**: Reusable pipeline functions for loading data, sampling, preprocessing, task mapping, labor-transfer analysis, and timezone normalization.
- **data/**: Datasets, caches and derived outputs.
	- `data/input/`: Raw input CSVs and WildChat exports.
	- `data/batches/`: JSONL batch inputs and outputs.
	- `data/chroma/`: Chroma/SQLite vector stores and indexes used for nearest-neighbour lookups.
	- `data/output/`: Generated CSV outputs from runs.
- **prompts/**: Python prompt definitions.
- **utilities/**: Helper modules for configurable paths and environment variables.
- **notebooks/**: Jupyter notebooks for exploration, pipeline examples and validation (`main.ipynb`, `pipeline.ipynb`, `validation.ipynb`).
- **manually_labeled/**: Human-labeled datasets used for evaluation and validation.
- **output/** and **plots/**: Example outputs and plotting notebooks.
- **validation/**: Validation scripts and helpers.

