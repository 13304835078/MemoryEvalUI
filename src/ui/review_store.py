from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW_PATH = PROJECT_ROOT / "data" / "results" / "human_reviews.jsonl"


def review_key(case_id: str, model_name: str = "unknown", prompt_version: str = "unknown") -> str:
    return f"{case_id}||{model_name}||{prompt_version}"


def load_reviews(path: str | Path = DEFAULT_REVIEW_PATH) -> dict[str, dict[str, Any]]:
    path = Path(path)
    reviews: dict[str, dict[str, Any]] = {}

    if not path.exists():
        return reviews

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = review_key(
                str(item.get("case_id", "")),
                str(item.get("model_name", "unknown")),
                str(item.get("prompt_version", "unknown")),
            )
            reviews[key] = item

    return reviews


def save_all_reviews(reviews: dict[str, dict[str, Any]],
                     path: str | Path = DEFAULT_REVIEW_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in reviews.values():
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def upsert_review(review: dict[str, Any],
                  path: str | Path = DEFAULT_REVIEW_PATH) -> dict[str, dict[str, Any]]:
    reviews = load_reviews(path)

    review = dict(review)
    review.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    key = review_key(
        str(review.get("case_id", "")),
        str(review.get("model_name", "unknown")),
        str(review.get("prompt_version", "unknown")),
    )

    reviews[key] = review
    save_all_reviews(reviews, path)
    return reviews


def reviews_to_dataframe(reviews: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if not reviews:
        return pd.DataFrame()
    return pd.DataFrame(list(reviews.values()))