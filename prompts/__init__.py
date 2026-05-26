from .labor_transfer_prompt import build as labor_transfer
from .task_mapping_prompt import build as task_mapping
from .work_related_prompt import build as work_related
from .occupation_mapping_prompt import build as occupation_mapping

__all__ = [
    "labor_transfer",
    "task_mapping",
    "work_related",
    "occupation_mapping",
]
