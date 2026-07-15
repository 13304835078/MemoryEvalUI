import threading

from src.ui.review_store import load_reviews, review_key, upsert_review


def test_review_store_roundtrip(tmp_path):
    path = tmp_path / "reviews.jsonl"

    upsert_review({
        "case_id": "case-1",
        "model_name": "model",
        "prompt_version": "v1",
        "comment": "人工确认",
    }, path)

    reviews = load_reviews(path)
    key = review_key("case-1", "model", "v1")
    assert reviews[key]["comment"] == "人工确认"
    assert reviews[key]["timestamp"]


def test_parallel_review_upserts_do_not_lose_rows(tmp_path):
    path = tmp_path / "reviews.jsonl"
    barrier = threading.Barrier(11)

    def worker(index: int) -> None:
        barrier.wait()
        upsert_review({
            "case_id": f"case-{index}",
            "model_name": "model",
            "prompt_version": "v1",
            "comment": str(index),
        }, path)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(10)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=3)

    assert all(not thread.is_alive() for thread in threads)
    assert len(load_reviews(path)) == 10
