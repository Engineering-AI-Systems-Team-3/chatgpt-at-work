from __future__ import annotations

import json
import copy
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Any
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from clio.hierarchizer.config import (
    NEIGHBORHOOD_SIZE,
    TOP_MIN,
    TOP_MAX,
    CONTRASTIVE_M,
    MODEL,
)
from clio.hierarchizer.prompts import (
    PROPOSE_PROMPT,
    DEDUP_PROMPT,
    ASSIGN_PROMPT,
    RENAME_PROMPT,
)
from clio.hierarchizer.models import Hierarchy, HierarchyLevel
from utilities import RANDOM_SEED, submit_and_retrieve


def _parse_batch_json_results(output_path: Path) -> dict[str, dict[str, Any]]:
    """
    Reads the JSONL output file of the OpenAI Batch and parses the generated JSON content.
    Returns a dictionary {custom_id: parsed_json}.

    :param output_path: Path to the JSONL output file from the Batch API.
    :return: A dictionary mapping each custom_id to the parsed JSON content of the LLM response. If a line fails to parse, the value will be an empty dictionary and a warning will be printed.
    """
    results = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            custom_id = data.get("custom_id")

            if data["response"]["status_code"] != 200:
                print(
                    f"[Warning] Batch call failed for {custom_id}: {data['response']}"
                )
                continue

            content = data["response"]["body"]["choices"][0]["message"]["content"]
            try:
                results[custom_id] = json.loads(content)
            except json.JSONDecodeError:
                print(
                    f"[Warning] Impossibile decodificare JSON per {custom_id}:\n{content}"
                )
                results[custom_id] = {}
    return results


class Hierarchizer:
    """
    Builds (or loads from cache) a multi-level hierarchy over a flat list of text items.
    """

    def __init__(self, client: OpenAI, model: str = MODEL):
        self.client = client
        self.model = model
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2", device="mps")
        self.batch_dir = Path(__file__).parent.parent.parent / "data" / "batches"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        self.current_level = 0

    def build(
        self,
        items: list[str],
        save_path: Optional[Path] = None,
        resume: bool = True,
    ) -> Hierarchy:
        """
        Build the full hierarchy from a flat list of items. Saves to save_path if provided.

        :param items: List of text items to build the hierarchy over (e.g. O*NET task descriptions).
        :param save_path: Optional path to save the resulting hierarchy JSON.
        :return: The resulting Hierarchy object.
        """
        if resume and save_path is not None and save_path.exists():
            print(f"[Hierarchizer] Checkpoint found. Resuming from {save_path}...")
            hierarchy = Hierarchy.load(save_path)
            current_items = hierarchy.levels[-1].items
            self.current_level = len(hierarchy.levels) - 1
            print(
                f"[Hierarchizer] Resuming at Level {self.current_level} with {len(current_items)} items."
            )
        else:
            hierarchy = Hierarchy()
            hierarchy.levels.append(HierarchyLevel(items=items))
            current_items = items
            self.current_level = 0

        while True:
            n_current = len(current_items)
            top_target = (TOP_MAX + TOP_MIN) // 2
            n_target = max(TOP_MAX, round((top_target * n_current) ** 0.5))

            print(f"[Hierarchizer] Building level above {n_current} items...")
            parent_items, parent_indices = self._build_one_level(
                items=current_items, n_target=n_target
            )
            hierarchy.levels[-1].parent_indices = parent_indices
            hierarchy.levels.append(HierarchyLevel(items=parent_items))
            current_items = parent_items

            if save_path is not None:
                checkpoint_path = save_path.with_name(
                    f"{save_path.stem}_level_{self.current_level}{save_path.suffix}"
                )
                hierarchy.save(checkpoint_path)
                print(
                    f"[Hierarchizer] Intermediate step (Level {self.current_level}) saved to {save_path}"
                )

            if TOP_MIN <= len(current_items) <= TOP_MAX:
                print(
                    f"[Hierarchizer] Reached top level with {len(current_items)} items. Done."
                )
                break
            if len(current_items) <= TOP_MIN:
                print(f"[Hierarchizer] Only {len(current_items)} items left, stopping.")
                break
            self.current_level += 1

        if save_path is not None:
            hierarchy.save(save_path)
            print(f"[Hierarchizer] Hierarchy saved to {save_path}")

        return hierarchy

    @staticmethod
    def load(path: Path) -> Hierarchy:
        h = Hierarchy.load(path)
        print(
            f"[Hierarchizer] Loaded hierarchy from {path} "
            f"({len(h.levels)} levels, "
            f"{len(h.levels[0].items)} base items, "
            f"{len(h.levels[-1].items)} top items)"
        )
        return h

    def _format_messages(self, prompt_template: list[dict], **kwargs) -> list[dict]:

        return [
            {"role": msg["role"], "content": msg["content"].format(**kwargs)}
            for msg in prompt_template
        ]

    def _build_one_level(
        self, items: list[str], n_target: int
    ) -> tuple[list[str], list[int]]:
        """
        Build one level of the hierarchy above the given items, returning the new parent items and their indices for each child.
        The way it works is:
        1. Generate embeddings for all items and cluster them into neighborhoods using KMeans.
        2. For each neighborhood, call the LLM to propose candidate parent items, using also CONTRASTIVE_M nearest items outside the neighborhood as contrastive examples.
        3. Deduplicate the proposed parent items across neighborhoods by calling the LLM again.
        4. Assign each child item to the best-fitting parent item by calling the LLM for each child.
        5. Rename each parent item based on its assigned children by calling the LLM again.

        :param items: List of text items to build the next level above.
        :param n_target: The target number of parent items for the new level.
        :return: A tuple of (parent_items, parent_indices) where:
            - parent_items: List of proposed parent item names for the new level.
            - parent_indices: List of indices into parent_items for each item in the input list, indicating which parent each item was assigned to.
        """
        criteria = (
            "The cluster name should be a sentence in the imperative that captures the task "
            "being performed. For example, 'Develop and debug software applications' or "
            "'Draft and refine professional business emails'."
        )

        embeddings = self._embed(items)
        kmeans, neighborhood_labels = self._run_kmeans_clustering(items, embeddings)
        raw_proposals = self._stage_propose(
            items=items,
            embeddings=embeddings,
            kmeans=kmeans,
            neighborhood_labels=neighborhood_labels,
            criteria=criteria,
        )
        parent_items = self._stage_deduplicate(
            proposals=raw_proposals, n_target=n_target, criteria=criteria
        )
        parent_indices = self._stage_assign(
            items=items, parent_items=parent_items, criteria=criteria
        )

        """
        Some parents may not have any children assigned to them. To avoid wasting LLM calls on renaming empty parents, we remove them from the hierarchy and re-index the parent_indices. Re-indexing is done by mapping old parent indices to new ones based on the non-empty parents. For example, if we have 5 proposed parents and only parents 0, 2, and 4 have children assigned, we keep those and remap their indices to 0, 1, and 2 respectively. The parent_indices are then updated to reflect this new indexing.
        """
        assigned = set(parent_indices)
        non_empty = [i for i in range(len(parent_items)) if i in assigned]
        if len(non_empty) < len(parent_items):
            n_removed = len(parent_items) - len(non_empty)
            print(f"Removing {n_removed} parent(s) with no assigned children.")
            remap = {old_idx: new_idx for new_idx, old_idx in enumerate(non_empty)}
            parent_items = [parent_items[i] for i in non_empty]
            parent_indices = [remap[i] for i in parent_indices]

        parent_items = self._stage_rename(
            items=items,
            parent_items=parent_items,
            parent_indices=parent_indices,
            criteria=criteria,
        )

        return parent_items, parent_indices

    def _run_kmeans_clustering(
        self, items: list[str], embeddings: np.ndarray
    ) -> tuple[KMeans, np.ndarray]:
        """
        Run K-Means clustering on the given items and their embeddings.

        :param items: List of text items to cluster.
        :param embeddings: 2D numpy array of item embeddings.
        :return: A tuple of (kmeans, neighborhood_labels) where:
            - kmeans: The fitted KMeans model.
            - neighborhood_labels: The cluster labels for each item.
        """
        # K is chosen so that the average neighborhood size is around NEIGHBORHOOD_SIZE
        k = max(1, round(len(items) / NEIGHBORHOOD_SIZE))
        print(f"Clustering {len(items)} items into {k} neighborhoods...")

        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init="auto")
        neighborhood_labels = kmeans.fit_predict(embeddings)
        return kmeans, neighborhood_labels

    def _embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts using the embedding model.

        :param texts: List of text strings to embed.
        :return: A 2D numpy array of shape (len(texts), embedding_dimension). Each row corresponds to the embedding of the respective text.
        """
        return self.embedding_model.encode(texts, show_progress_bar=True)

    def _nearest_to_centroid(
        self,
        centroid: np.ndarray,
        embeddings: np.ndarray,
        candidate_indices: list[int],
        m: int,
    ) -> list[int]:
        """
        Given a centroid and a list of candidate item indices, return the indices of the m candidates whose embeddings are closest to the centroid.
        The way it works is:
        1. Extract the embeddings of the candidate items.
        2. Compute the Euclidean distances between each candidate embedding and the centroid.
        3. Sort the candidates by their distances and return the indices of the m closest ones.

        :param centroid: The centroid embedding to compare against (1D numpy array).
        :param embeddings: The full array of item embeddings (2D numpy array).
        :param candidate_indices: List of indices into embeddings that are the candidates to consider.
        :param m: The number of nearest neighbors to return.
        :return: A list of indices (from candidate_indices) corresponding to the m nearest neighbors to the centroid.
        """

        if not candidate_indices:
            return []
        candidate_embs = embeddings[candidate_indices]
        dists = np.linalg.norm(candidate_embs - centroid, axis=1)
        sorted_idx = np.argsort(dists)[:m]
        return [candidate_indices[i] for i in sorted_idx]

    def _stage_propose(
        self,
        items: list[str],
        embeddings: np.ndarray,
        kmeans: KMeans,
        neighborhood_labels: np.ndarray,
        criteria: str,
    ) -> list[str]:
        """
        For each neighborhood, call the LLM to propose candidate parent items, using also CONTRASTIVE_M nearest items outside the neighborhood as contrastive examples.

        :param items: List of text items to build the next level above.
        :param embeddings: 2D numpy array of item embeddings.
        :param kmeans: The fitted KMeans model.
        :param neighborhood_labels: The cluster labels for each item.
        :param criteria: The criteria for proposing parent items.
        :return: A list of raw proposed parent item names from the LLM, before deduplication.
        """
        print(f"[Stage 1/4] Generating parent proposals via Batch API...")
        propose_messages = []
        k = kmeans.n_clusters

        for nb_idx in range(k):
            member_indices = np.where(neighborhood_labels == nb_idx)[0].tolist()
            centroid = kmeans.cluster_centers_[nb_idx]

            non_member_indices = np.where(neighborhood_labels != nb_idx)[0].tolist()

            neighbor_indices = self._nearest_to_centroid(
                centroid=centroid,
                embeddings=embeddings,
                candidate_indices=non_member_indices,
                m=CONTRASTIVE_M,
            )

            members = [items[i] for i in member_indices]
            neighbors = [items[i] for i in neighbor_indices]

            n_target = max(2, round(len(members) / 10))

            msgs = self._format_messages(
                prompt_template=PROPOSE_PROMPT,
                cluster_list="\n".join(f"- {m}" for m in members),
                neighbor_list="\n".join(f"- {n}" for n in neighbors),
                desired_names=n_target,
                min_names=max(1, int(0.5 * n_target)),
                max_names=int(1.5 * n_target),
                criteria=criteria,
            )
            propose_messages.append(msgs)

        propose_batch_in = (
            self.batch_dir / "input" / f"propose_lv{self.current_level}_in.jsonl"
        )
        propose_batch_out = (
            self.batch_dir / "output" / f"propose_lv{self.current_level}_out.jsonl"
        )

        propose_results = submit_and_retrieve(
            client=self.client,
            messages=propose_messages,
            batch_file=propose_batch_in,
            output_file=propose_batch_out,
            word_id="propose",
            parse_func=_parse_batch_json_results,
        )

        all_proposals = []
        for i in range(k):
            res_dict = propose_results.get(f"propose_{i}", {})
            clusters = res_dict.get("answer", [])
            all_proposals.extend(clusters)

        print(f"Collected {len(all_proposals)} raw proposals.")
        return all_proposals

    def _stage_deduplicate(
        self, proposals: list[str], n_target: int, criteria: str
    ) -> list[str]:
        """
        Merges/splits proposals across neighborhoods by calling the LLM to deduplicate them.

        :param proposals: List of raw proposed parent item names from the LLM, before deduplication.
        :param n_target: The target number of parent items for the new level.
        :param criteria: The criteria for deduplication.
        :return: A list of deduplicated parent item names, where similar proposals have been merged and overly broad proposals have been split, resulting in a more distinct and manageable set of parent items
        """
        print(f"[Stage 2/4] Deduplicating proposals...")
        if len(proposals) <= TOP_MAX:
            print("    Proposals within limits, skipping LLM deduplication.")
            return proposals

        msgs = self._format_messages(
            prompt_template=DEDUP_PROMPT,
            cluster_names="\n".join(f"- {p}" for p in proposals),
            desired_names=n_target,
            min_names=max(1, int(n_target * 0.5)),
            max_names=int(n_target * 1.5),
            criteria=criteria,
        )

        dedup_batch_in = (
            self.batch_dir / "input" / f"dedup_lv{self.current_level}_in.jsonl"
        )
        dedup_batch_out = (
            self.batch_dir / "output" / f"dedup_lv{self.current_level}_out.jsonl"
        )

        dedup_results = submit_and_retrieve(
            client=self.client,
            messages=[msgs],
            batch_file=dedup_batch_in,
            output_file=dedup_batch_out,
            word_id="dedup",
            parse_func=_parse_batch_json_results,
        )

        res_dict = dedup_results.get("dedup_0", {})
        parent_items = res_dict.get("answer", proposals)
        print(f"Reduced to {len(parent_items)} distinct parent items.")
        return parent_items

    def _stage_assign(
        self, items: list[str], parent_items: list[str], criteria: str
    ) -> list[int]:
        """
        Each child item is assigned to the best-fitting parent item by calling the LLM for each child, providing the list of parent items as options.

        :param items: List of text items to build the next level above.
        :param parent_items: List of proposed parent item names for the new level.
        :param criteria: The criteria for assignment.
        :return: List of indices into parent_items for each item in the input list, indicating which parent each item was assigned to.
        """
        print(f"[Stage 3/4] Assigning {len(items)} items to parents...")
        assign_messages = []
        for item in items:
            shuffled_parents = copy.copy(parent_items)
            random.shuffle(shuffled_parents)

            msgs = self._format_messages(
                prompt_template=ASSIGN_PROMPT,
                cluster_name=item,
                parents="\n".join(f"{p}\n" for p in shuffled_parents),
                criteria=criteria,
            )
            assign_messages.append(msgs)

        assign_batch_in = (
            self.batch_dir / "input" / f"assign_lv{self.current_level}_in.jsonl"
        )
        assign_batch_out = (
            self.batch_dir / "output" / f"assign_lv{self.current_level}_out.jsonl"
        )

        assign_results = submit_and_retrieve(
            client=self.client,
            messages=assign_messages,
            batch_file=assign_batch_in,
            output_file=assign_batch_out,
            word_id="assign",
            parse_func=_parse_batch_json_results,
        )

        parent_indices = []
        for i, item in enumerate(items):
            res_dict = assign_results.get(f"assign_{i}", {})
            assigned_name = res_dict.get("answer", "").strip().lower()

            matched_idx = 0
            for j, p in enumerate(parent_items):
                if p.lower() == assigned_name:
                    matched_idx = j
                    break
            parent_indices.append(matched_idx)

        return parent_indices

    def _stage_rename(
        self,
        items: list[str],
        parent_items: list[str],
        parent_indices: list[int],
        criteria: str,
    ) -> list[str]:
        """
        Renames parent items based on the final assignments of their child items.

        :param items: List of text items to build the next level above.
        :param parent_items: List of proposed parent item names for the new level.
        :param parent_indices: List of indices into parent_items for each item in the input list, indicating which parent each item was assigned to.
        :param criteria: The criteria for renaming.
        :return: Updated list of parent item names, where each parent item has been renamed by the LLM based on the child items assigned to it
        """
        print(f"[Stage 4/4] Renaming parents based on final children assignments...")

        groups: dict[int, list[str]] = {i: [] for i in range(len(parent_items))}
        for child_idx, parent_idx in enumerate(parent_indices):
            groups[parent_idx].append(items[child_idx])

        rename_messages = []
        parents_to_rename_indices = []

        for i, parent in enumerate(parent_items):
            assigned_children = groups.get(i, [])
            if assigned_children:
                msgs = self._format_messages(
                    prompt_template=RENAME_PROMPT,
                    children="\n".join(f"{c}\n" for c in assigned_children),
                    criteria=criteria,
                )
                rename_messages.append(msgs)
                parents_to_rename_indices.append(i)

        if not rename_messages:
            return parent_items

        rename_batch_in = (
            self.batch_dir / "input" / f"rename_lv{self.current_level}_in.jsonl"
        )
        rename_batch_out = (
            self.batch_dir / "output" / f"rename_lv{self.current_level}_out.jsonl"
        )

        rename_results = submit_and_retrieve(
            client=self.client,
            messages=rename_messages,
            batch_file=rename_batch_in,
            output_file=rename_batch_out,
            word_id="rename",
            parse_func=_parse_batch_json_results,
        )

        updated_parent_items = copy.copy(parent_items)
        for req_idx, orig_parent_idx in enumerate(parents_to_rename_indices):
            res_dict = rename_results.get(f"rename_{req_idx}", {})
            new_name = res_dict.get("name")
            if new_name:
                updated_parent_items[orig_parent_idx] = new_name.strip()

        return updated_parent_items


# Example usage
if __name__ == "__main__":
    from openai import OpenAI

    tasks_path = (
        Path(__file__).parent.parent.parent / "data" / "input" / "TaskStatements.csv"
    )
    tasks_df = pd.read_csv(tasks_path)
    tasks = tasks_df["Task"].dropna().tolist()

    print(f"Loaded {len(tasks)} tasks from {tasks_path}")

    client = OpenAI()
    save_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "hierarchy"
        / "onet_hierarchy.json"
    )
    h = Hierarchizer(client=client)

    hierarchy = h.build(items=tasks, save_path=save_path, resume=True)

    print(f"Levels: {len(hierarchy.levels)}")
    for i, lv in enumerate(hierarchy.levels):
        print(f"Level {i}: {len(lv.items)} items")
