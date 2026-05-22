import json
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class HierarchyLevel:
    items: list[str]
    parent_indices: Optional[list[int]] = None


@dataclass
class Hierarchy:
    levels: list[HierarchyLevel] = field(default_factory=list)
    # levels[0]  = base (leaf nodes, i.e., O*NET tasks)
    # levels[-1] = top  (root nodes)

    def to_dict(self) -> dict:
        return {
            "levels": [
                {
                    "items": lv.items,
                    "parent_indices": lv.parent_indices,
                }
                for lv in self.levels
            ]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hierarchy":
        h = cls()
        for lv in d["levels"]:
            h.levels.append(
                HierarchyLevel(
                    items=lv["items"],
                    parent_indices=lv.get("parent_indices"),
                )
            )
        return h

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Hierarchy":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
