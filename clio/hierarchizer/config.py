NEIGHBORHOOD_SIZE = 40  # target items per neighborhood
TOP_MIN = 10  # stop when top level has >= TOP_MIN items ...
TOP_MAX = 15  # ... and <= TOP_MAX items
CONTRASTIVE_M = 10  # nearest items outside neighborhood shown to the LLM
MODEL = "gpt-5-mini-2025-08-07"  # LLM used for proposals, dedup, assignment, rename
TEMPERATURE = 1.0  # temperature for LLM calls in hierarchization process. Same value used by Anthropic in their paper
RANDOM_SEED = 42  # random seed for KMeans
