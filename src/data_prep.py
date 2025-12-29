import json
import os
import random
import re
from typing import List

from datasets import load_from_disk


def set_seed(seed: int = 42) -> None:
    random.seed(seed)


def split_sentences(text: str) -> List[str]:
    # Simple sentence split for narrative text.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip().split()) >= 6]


def collect_distractors(seed: int = 42, max_sentences: int = 2000) -> List[str]:
    set_seed(seed)
    sentences: List[str] = []
    datasets = ["narrative_qa_helm", "booksum"]
    for name in datasets:
        path = os.path.join("datasets", name)
        if not os.path.isdir(path):
            continue
        ds = load_from_disk(path)
        split = ds["train"] if "train" in ds else list(ds.values())[0]
        for row in split:
            passage = row.get("passage") or row.get("chapter") or row.get("text") or ""
            if not passage:
                continue
            sentences.extend(split_sentences(passage))
            if len(sentences) >= max_sentences:
                break
        if len(sentences) >= max_sentences:
            break
    random.shuffle(sentences)
    return sentences[:max_sentences]


def main() -> None:
    output_path = os.path.join("results", "distractors.json")
    sentences = collect_distractors()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"sentences": sentences}, f, indent=2)
    print(f"Saved {len(sentences)} distractor sentences to {output_path}")


if __name__ == "__main__":
    main()
