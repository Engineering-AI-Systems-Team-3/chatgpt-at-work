import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

MAX_WORKERS = 128
WAIT_TIME = 60
RANDOM_SEED = 42
BUFFER_SIZE = 10_000
SAMPLE_PERCENTAGE = 0.035
VALIDATION_SAMPLE = 100
CHECKPOINT_INTERVAL = 10
BATCH_SIZE = 5_000

#  Model
# MODEL_ID = "nvidia/nemotron-3-super-120b-a12b:free"
# MODEL_ID = "gpt-5.4-nano-2026-03-17"
MODEL_ID = "gpt-5-mini-2025-08-07"

#  API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#  WildChat dataset
WILDCHAT_DATASET = "allenai/WildChat-4.8M"
CONVERSATION_FIELD = "conversation"
LANGUAGE_FIELD = "language"
TARGET_LANGUAGE = "English"

PARENT_COLUMN = "parent_id"
COUNT_VARIABLE = "onet_task_count"

#  Directory layout
DATA_PATH = Path("./data")
INPUT_DATA_DIR = DATA_PATH / "input"
OUTPUT_DIR = DATA_PATH / "output"
LABELED_DATA_DIR = DATA_PATH / "manually_labeled"
PROMPTS_DIR = Path("./prompts")
BATCHES_DIR = DATA_PATH / "batches"
INPUT_BATCHES_DIR = BATCHES_DIR / "input"
OUTPUT_BATCHES_DIR = BATCHES_DIR / "output"

#  Input files
RAW_1P_DATA_PATH = INPUT_DATA_DIR / "aei_raw_1p_api_2026-02-05_to_2026-02-12.csv"
RAW_CLAUDE_DATA_PATH = INPUT_DATA_DIR / "aei_raw_claude_ai_2026-02-05_to_2026-02-12.csv"
TASK_STATEMENTS_PATH = INPUT_DATA_DIR / "TaskStatements.csv"
TASK_RATING_PATH = INPUT_DATA_DIR / "TaskRatings.xlsx"

WILDCHAT_FULL = INPUT_DATA_DIR / "WildChat-4.8M"
WILDCHAT_ENGLISH = INPUT_DATA_DIR / "WildChat-4.8M-english"
WILDCHAT_SAMPLES_FILE = INPUT_DATA_DIR / f"WildChat-4.8M-sample-{SAMPLE_PERCENTAGE}.csv"

# Labeled files
WORK_RELATED_LABELED_OUTPUT_PATH = LABELED_DATA_DIR / "work_related_750.csv"
TASK_MAPPING_LABELED_OUTPUT_PATH = LABELED_DATA_DIR / "task_mapping_750.csv"

#  Prompt files
WORK_RELATED_PROMPT_PATH = PROMPTS_DIR / "work_related.json"
TASK_MAPPING_PROMPT_PATH = PROMPTS_DIR / "task_mapping.json"
LAST_LEVEL_TASK_MAPPING_PROMPT_PATH = PROMPTS_DIR / "last_level_task_mapping.json"
LABOR_TRANSFER_PROMPT_PATH = PROMPTS_DIR / "labor_transfer.json"

#  Output files
WORK_RELATED_OUTPUT_PATH = OUTPUT_DIR / f"work_related_{SAMPLE_PERCENTAGE}.csv"
TASK_MAPPING_OUTPUT_PATH = OUTPUT_DIR / f"task_mapping_{SAMPLE_PERCENTAGE}.csv"
TIMEZONES_OUTPUT_PATH = OUTPUT_DIR / f"timezones_{SAMPLE_PERCENTAGE}.csv"
LABOR_TRANSFER_OUTPUT_FILE = (
    OUTPUT_DIR / f"labor_transfer_{SAMPLE_PERCENTAGE}_output.csv"
)

#  Batch Input files
WORK_RELATED_BATCH_FILE = INPUT_BATCHES_DIR / f"work_related_{SAMPLE_PERCENTAGE}.jsonl"
LABOR_TRANSFER_BATCH_FILE = (
    INPUT_BATCHES_DIR / f"labor_transfer_{SAMPLE_PERCENTAGE}.jsonl"
)
LEVEL_2_BATCH_FILE = INPUT_BATCHES_DIR / f"level2_{SAMPLE_PERCENTAGE}.jsonl"
LEVEL_1_BATCH_FILE = INPUT_BATCHES_DIR / f"level1_{SAMPLE_PERCENTAGE}.jsonl"
LEVEL_0_BATCH_FILE = INPUT_BATCHES_DIR / f"level0_{SAMPLE_PERCENTAGE}.jsonl"

# Batch Output files
LABOR_TRANSFER_BATCH_OUTPUT_FILE = (
    OUTPUT_BATCHES_DIR / f"labor_transfer_{SAMPLE_PERCENTAGE}_output.jsonl"
)
WORK_RELATED_BATCH_OUTPUT_FILE = (
    OUTPUT_BATCHES_DIR / f"work_related_{SAMPLE_PERCENTAGE}_output.jsonl"
)
LEVEL_2_BATCH_OUTPUT_FILE = (
    OUTPUT_BATCHES_DIR / f"level2_{SAMPLE_PERCENTAGE}_output.jsonl"
)
LEVEL_1_BATCH_OUTPUT_FILE = (
    OUTPUT_BATCHES_DIR / f"level1_{SAMPLE_PERCENTAGE}_output.jsonl"
)
LEVEL_0_BATCH_OUTPUT_FILE = (
    OUTPUT_BATCHES_DIR / f"level0_{SAMPLE_PERCENTAGE}_output.jsonl"
)

#  O*NET
MAJOR_CATEGORIES = {
    "11": "Management Occupations",
    "13": "Business and Financial Operations Occupations",
    "15": "Computer and Mathematical Occupations",
    "17": "Architecture and Engineering Occupations",
    "19": "Life, Physical, and Social Science Occupations",
    "21": "Community and Social Service Occupations",
    "23": "Legal Occupations",
    "25": "Educational Instruction and Library Occupations",
    "27": "Arts, Design, Entertainment, Sports, and Media Occupations",
    "29": "Healthcare Practitioners and Technical Occupations",
    "31": "Healthcare Support Occupations",
    "33": "Protective Service Occupations",
    "35": "Food Preparation and Serving Related Occupations",
    "37": "Building and Grounds Cleaning and Maintenance Occupations",
    "39": "Personal Care and Service Occupations",
    "41": "Sales and Related Occupations",
    "43": "Office and Administrative Support Occupations",
    "45": "Farming, Fishing, and Forestry Occupations",
    "47": "Construction and Extraction Occupations",
    "49": "Installation, Maintenance, and Repair Occupations",
    "51": "Production Occupations",
    "53": "Transportation and Material Moving Occupations",
}

HIERARCHY_PATH = DATA_PATH / "hierarchy" / "onet_hierarchy.json"
