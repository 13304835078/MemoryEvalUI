from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from src.persistence import atomic_write_bytes
from src.ui.state_io import atomic_write_json


PARTITIONS = ("discovery", "validation", "locked_test")


def _is_session_start(row: dict[str, Any], index: int) -> bool:
    if index == 0:
        return True
    try:
        return int(float(str(row.get("轮次", "")).strip())) == 1
    except (TypeError, ValueError):
        return False


def _reviewer_filter_values(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").replace("，", ",").split(",") if item.strip()}


def _partition_counts(
    total: int,
    ratios: tuple[float, float, float],
    minimums: tuple[int, int, int] = (1, 1, 1),
) -> tuple[int, int, int]:
    minimums = tuple(max(1, int(value)) for value in minimums)
    if total < 3:
        raise ValueError(
            "可信闭环至少需要 3 位不同评测人的完整历史，才能切分为 Discovery、Validation 和 Locked Test。"
        )
    if total < sum(minimums):
        raise ValueError(
            "按当前统计验收门槛，可信闭环至少需要 "
            f"{sum(minimums)} 位不同评测人（Discovery {minimums[0]}、"
            f"Validation {minimums[1]}、Locked Test {minimums[2]}）。"
        )
    ratio_sum = sum(max(0.0, value) for value in ratios)
    if ratio_sum <= 0:
        ratios = (0.6, 0.2, 0.2)
        ratio_sum = 1.0
    normalized = tuple(max(0.0, value) / ratio_sum for value in ratios)
    validation = max(1, int(round(total * normalized[1])))
    locked_test = max(1, int(round(total * normalized[2])))
    while validation + locked_test >= total:
        if validation >= locked_test and validation > 1:
            validation -= 1
        elif locked_test > 1:
            locked_test -= 1
        else:
            break
    discovery = total - validation - locked_test
    if discovery < 1:
        raise ValueError("切分比例无法为 Discovery 保留至少 1 位评测人。")
    counts = [discovery, validation, locked_test]
    for target_index, minimum in enumerate(minimums):
        while counts[target_index] < minimum:
            donors = [
                index for index in range(3)
                if index != target_index and counts[index] > minimums[index]
            ]
            if not donors:
                raise ValueError("切分比例与统计验收门槛无法同时满足，请增加评测人或降低最少独立簇。")
            donor = max(donors, key=lambda index: (counts[index] - minimums[index], counts[index], -index))
            counts[donor] -= 1
            counts[target_index] += 1
    return counts[0], counts[1], counts[2]


def _excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return output.getvalue()


def split_excel_by_reviewer_session(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    sheet_name: str | int | None = 0,
    reviewer_filter: str = "",
    discovery_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    locked_test_ratio: float = 0.2,
    seed: str = "memory-eval-v1",
    min_discovery_reviewers: int = 1,
    min_validation_reviewers: int = 1,
    min_locked_test_reviewers: int = 1,
) -> dict[str, Any]:
    """Create a leakage-safe holdout split while preserving source chronology.

    A reviewer's complete cross-session history is assigned to one partition. Rows
    inside each partition remain in their original source order. The synthetic
    source segment column preserves LONG_MEMORY reset boundaries when reviewers
    removed by the split originally appeared between two sessions.
    """
    source = Path(input_path)
    df = pd.read_excel(source, sheet_name=sheet_name).fillna("")
    rows = df.to_dict("records")
    if not rows:
        raise ValueError("输入 Excel 为空，无法建立可信闭环切分。")

    starts = [index for index, row in enumerate(rows) if _is_session_start(row, index)]
    allowed_reviewers = _reviewer_filter_values(reviewer_filter)
    reviewer_groups: dict[str, dict[str, Any]] = {}
    source_segment = 0
    previous_reviewer = ""
    row_segments: dict[int, int] = {}

    for session_index, start in enumerate(starts):
        end = starts[session_index + 1] if session_index + 1 < len(starts) else len(rows)
        segment = rows[start:end]
        reviewer = next(
            (str(row.get("评测人", "")).strip() for row in segment if str(row.get("评测人", "")).strip()),
            "",
        )
        if not reviewer:
            raise ValueError(f"第 {session_index + 1} 个 session 缺少评测人，无法进行防泄漏切分。")
        if reviewer != previous_reviewer:
            source_segment += 1
            previous_reviewer = reviewer
        for row_index in range(start, end):
            row_segments[row_index] = source_segment
        if allowed_reviewers and reviewer not in allowed_reviewers:
            continue

        source_session_id = next(
            (str(row.get("session_id", "")).strip() for row in segment if str(row.get("session_id", "")).strip()),
            str(session_index + 1),
        )
        group = reviewer_groups.setdefault(
            reviewer,
            {
                "identity": reviewer,
                "group_id": hashlib.sha256(reviewer.encode("utf-8")).hexdigest()[:16],
                "reviewer_hash": hashlib.sha256(reviewer.encode("utf-8")).hexdigest()[:12],
                "source_session_ids": [],
                "row_ranges": [],
                "row_indexes": [],
                "first_row": start,
            },
        )
        group["source_session_ids"].append(source_session_id)
        group["row_ranges"].append([start, end])
        group["row_indexes"].extend(range(start, end))

    groups = list(reviewer_groups.values())
    for group in groups:
        group["session_count"] = len(group["source_session_ids"])
        group["row_count"] = len(group["row_indexes"])
        group["sort_key"] = hashlib.sha256(f"{seed}|{group['identity']}".encode("utf-8")).hexdigest()

    shuffled = sorted(groups, key=lambda item: item["sort_key"])
    discovery_count, validation_count, _locked_count = _partition_counts(
        len(shuffled),
        (discovery_ratio, validation_ratio, locked_test_ratio),
        (
            min_discovery_reviewers,
            min_validation_reviewers,
            min_locked_test_reviewers,
        ),
    )
    boundaries = (discovery_count, discovery_count + validation_count)
    for index, group in enumerate(shuffled):
        group["partition"] = (
            "discovery" if index < boundaries[0]
            else "validation" if index < boundaries[1]
            else "locked_test"
        )

    working_df = df.copy()
    working_df["__source_reviewer_segment"] = [row_segments.get(index, 0) for index in range(len(df))]
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    reviewer_counts: dict[str, int] = {}
    session_counts: dict[str, int] = {}
    row_counts: dict[str, int] = {}
    for partition in PARTITIONS:
        selected_indexes = sorted(
            row_index
            for group in groups
            if group["partition"] == partition
            for row_index in group["row_indexes"]
        )
        partition_df = working_df.iloc[selected_indexes].copy()
        path = target_dir / f"{partition}.xlsx"
        atomic_write_bytes(path, _excel_bytes(partition_df))
        paths[partition] = str(path)
        partition_groups = [group for group in groups if group["partition"] == partition]
        reviewer_counts[partition] = len(partition_groups)
        session_counts[partition] = sum(int(group["session_count"]) for group in partition_groups)
        row_counts[partition] = len(selected_indexes)

    public_groups = []
    for group in sorted(groups, key=lambda item: item["first_row"]):
        public_groups.append({
            key: value
            for key, value in group.items()
            if key not in {"identity", "sort_key", "row_indexes"}
        })

    manifest = {
        "version": "reviewer_history_split_v2",
        "source_file": source.name,
        "seed": seed,
        "split_unit": "reviewer_history",
        "preserves_source_row_order": True,
        "ratios": {
            "discovery": discovery_ratio,
            "validation": validation_ratio,
            "locked_test": locked_test_ratio,
        },
        "partition_paths": paths,
        "partition_group_counts": reviewer_counts,
        "partition_reviewer_counts": reviewer_counts,
        "minimum_partition_reviewers": {
            "discovery": max(1, int(min_discovery_reviewers)),
            "validation": max(1, int(min_validation_reviewers)),
            "locked_test": max(1, int(min_locked_test_reviewers)),
        },
        "partition_session_counts": session_counts,
        "partition_row_counts": row_counts,
        "group_count": len(groups),
        "groups": public_groups,
    }
    manifest_path = target_dir / "split_manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
