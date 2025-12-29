import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class CharacterProfile:
    name: str
    role: str
    obj: str


def get_client() -> OpenAI:
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        return OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("Missing OPENAI_API_KEY or OPENROUTER_API_KEY")
    return OpenAI(api_key=openai_key)


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(5))
def call_model(client: OpenAI, model: str, prompt: str) -> Tuple[str, Dict]:
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=32,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content.strip()
    usage = response.usage.model_dump() if response.usage else {}
    return content, usage


def load_distractors(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sentences"]


def normalize_answer(text: str) -> str:
    return "".join(ch for ch in text.lower().strip() if ch.isalnum() or ch.isspace()).strip()


def is_correct(answer: str, expected: str) -> bool:
    answer_norm = f" {normalize_answer(answer)} "
    expected_norm = f" {normalize_answer(expected)} "
    return expected_norm in answer_norm


def build_story(
    characters: List[CharacterProfile],
    distractors: List[str],
    long_form: bool,
    objects: List[str],
) -> Tuple[str, Dict[str, str]]:
    parts: List[str] = []
    state = {c.name: c.obj for c in characters}
    distractor_count = 8 if long_form else 3

    for profile in characters:
        parts.append(f"{profile.name} is a {profile.role} who carries a {profile.obj}.")
        parts.extend(random.sample(distractors, k=distractor_count))

    update_steps = len(characters) * (2 if long_form else 1)
    for _ in range(update_steps):
        target = random.choice(characters)
        old_obj = state[target.name]
        new_obj = random.choice([o for o in objects if o != old_obj])
        state[target.name] = new_obj
        parts.append(
            f"Later, {target.name} traded the {old_obj} for a {new_obj}."
        )
        parts.extend(random.sample(distractors, k=distractor_count))

    return " ".join(parts), state


def build_prompt(story: str, target: CharacterProfile) -> str:
    return (
        "Read the story and answer the question with the single word object only.\n\n"
        f"Story: {story}\n\n"
        f"Question: At the end of the story, what object does {target.name} carry?\n"
        "Answer:"
    )


def load_cache(path: str) -> Dict[str, Dict]:
    cache: Dict[str, Dict] = {}
    if not os.path.exists(path):
        return cache
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            cache[row["prompt_hash"]] = row
    return cache


def save_row(path: str, row: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def run_experiment() -> None:
    set_seed(42)
    client = get_client()

    model = os.getenv("MODEL_NAME", "gpt-4.1")
    distractors = load_distractors(os.path.join("results", "distractors.json"))

    names = [
        "Avery", "Blake", "Casey", "Dakota", "Emerson", "Finley", "Gray", "Harper",
        "Indigo", "Jordan", "Kai", "Logan", "Morgan", "Nico", "Oakley", "Parker",
        "Quinn", "Riley", "Sawyer", "Taylor", "Val", "Winter", "Xen", "Yael",
        "Zane", "Alex", "Bailey", "Cameron", "Devon", "Elliot", "Frankie", "Gale",
        "Hayden", "Ira", "Jules", "Keegan", "Lane", "Marley", "Noel", "Orion",
        "Peyton", "Reese", "Shawn", "Teagan", "Ulric", "Vega", "Wren", "Xavier",
        "Yuri", "Zuri", "Alden", "Briar", "Clarke", "Drew", "Eden", "Flynn",
    ]
    roles = [
        "baker", "carpenter", "doctor", "engineer", "farmer", "gardener", "librarian",
        "musician", "nurse", "painter", "pilot", "teacher", "writer", "chef", "driver",
        "artist", "tailor", "sailor", "miner", "weaver", "coach", "merchant", "ranger",
        "plumber", "florist", "bartender", "detective", "biologist", "judge", "clerk",
        "editor", "farrier", "geologist", "host", "inspector", "jeweler", "knitter",
    ]
    objects = [
        "apple", "book", "coin", "drum", "emerald", "feather", "glove", "hammer",
        "ink", "jar", "key", "lantern", "map", "needle", "orb", "pen", "quill",
        "ring", "stone", "token", "umbrella", "vase", "whistle", "yarn", "compass",
        "crown", "dagger", "envelope", "flag", "goblet", "harp", "idol", "jewel",
        "kite", "lens", "mirror", "notebook", "oar", "pouch", "rope", "spear",
        "tablet", "urn", "violin", "wheel", "anchor", "badge", "candle", "flute",
        "gem", "helmet", "instrument", "journal", "kettle", "locket", "medal",
    ]

    n_values = [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
    lengths = ["short", "long"]
    trials_per_condition = 5

    output_path = os.path.join("results", "model_outputs.jsonl")
    cache = load_cache(output_path)

    run_id = datetime.utcnow().isoformat()
    meta_path = os.path.join("results", "metadata.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"model": model, "run_started": run_id}, f, indent=2)

    for n in n_values:
        for length in lengths:
            for trial in range(trials_per_condition):
                char_names = random.sample(names, k=n)
                char_roles = random.sample(roles, k=n)
                char_objs = random.sample(objects, k=n)
                characters = [
                    CharacterProfile(name=cn, role=cr, obj=co)
                    for cn, cr, co in zip(char_names, char_roles, char_objs)
                ]
                story, final_state = build_story(
                    characters, distractors, long_form=(length == "long"), objects=objects
                )
                target = random.choice(characters)
                target_obj = final_state[target.name]
                prompt = build_prompt(story, CharacterProfile(target.name, target.role, target_obj))
                prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

                if prompt_hash in cache:
                    continue

                answer, usage = call_model(client, model, prompt)
                correct = is_correct(answer, target_obj)

                recency_obj = final_state[characters[-1].name]
                recency_correct = recency_obj == target_obj
                random_guess = random.choice(objects)
                random_correct = random_guess == target_obj

                row = {
                    "prompt_hash": prompt_hash,
                    "run_id": run_id,
                    "model": model,
                    "n_characters": n,
                    "length": length,
                    "trial": trial,
                    "target_name": target.name,
                    "target_obj": target_obj,
                    "story": story,
                    "prompt": prompt,
                    "response": answer,
                    "correct": correct,
                    "recency_obj": recency_obj,
                    "recency_correct": recency_correct,
                    "random_guess": random_guess,
                    "random_correct": random_correct,
                    "usage": usage,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                save_row(output_path, row)
                time.sleep(0.2)


if __name__ == "__main__":
    run_experiment()
