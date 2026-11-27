"""VQA dataset access for Part 1.

All candidates are read from their official *validation* split (lmms-lab parquet
mirrors on the Hugging Face Hub). SmolVLM's training mix (The Cauldron) uses train
splits, so building both our fine-tuning set and our eval set from validation keeps
them unseen by the model. Later, train/eval are split by image (`group`) so that no
image appears on both sides.

Examples are streamed (no multi-GB download), shuffled with a fixed seed, and the
selected subset is cached with save_to_disk (on Drive under Colab) for later runs.
Images stay as compressed bytes while streaming and caching (decoded PIL images are
~10x larger and exhausted Colab's 12 GB RAM at 2.8k examples); they are decoded when a
row is accessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from common import get_out_dir


@dataclass(frozen=True)
class VQASpec:
    hf_id: str
    config: str | None
    split: str
    metric: str  # "vqa_acc" or "anls"


DATASETS: dict[str, VQASpec] = {
    "textvqa": VQASpec("lmms-lab/textvqa", None, "validation", "vqa_acc"),
    "docvqa": VQASpec("lmms-lab/DocVQA", "DocVQA", "validation", "anls"),
    "okvqa": VQASpec("lmms-lab/OK-VQA", None, "val2014", "vqa_acc"),
    "vqav2": VQASpec("lmms-lab/VQAv2", None, "validation", "vqa_acc"),
}

_ID_KEYS = ("question_id", "questionId", "id")
_GROUP_KEYS = ("image_id", "docId", "image_name", "image_path")


def _first(ex: dict, keys, default=None):
    for k in keys:
        if k in ex and ex[k] is not None:
            return ex[k]
    return default


def get_answers(ex: dict) -> list[str]:
    """Annotator answers as a list of strings, whatever the dataset's schema."""
    a = ex.get("answers")
    if a is None:
        a = ex.get("multiple_choice_answer")
    if a is None:
        return []
    if isinstance(a, str):
        return [a]
    if isinstance(a, dict):  # HF Sequence(dict) is returned as a dict of lists
        a = a.get("answer", a.get("answers", []))
    out = []
    for x in a:
        if isinstance(x, dict):
            x = x.get("answer")
        if x is not None:
            out.append(str(x))
    return out


def normalize_example(ex: dict, idx: int) -> dict:
    img = ex["image"]
    if hasattr(img, "mode") and img.mode != "RGB":  # decoded PIL image
        img = img.convert("RGB")
    # otherwise {"bytes", "path"} from Image(decode=False): decoded when the row is accessed
    return {
        "id": str(_first(ex, _ID_KEYS, idx)),
        "group": str(_first(ex, _GROUP_KEYS, f"row{idx}")),
        "image": img,
        "question": ex["question"].strip(),
        "answers": get_answers(ex),
    }


def stream_examples(name: str, seed: int = 42, buffer_size: int = 500, decode: bool = False) -> Iterator[dict]:
    from datasets import Image, load_dataset

    spec = DATASETS[name]
    ds = load_dataset(spec.hf_id, spec.config, split=spec.split, streaming=True)
    ds = ds.cast_column("image", Image(decode=decode))  # keep compressed bytes in RAM
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    for i, ex in enumerate(ds):
        ex = normalize_example(ex, i)
        if ex["answers"]:
            yield ex


def load_subset(name: str, n: int, seed: int = 42, buffer_size: int = 500, use_cache: bool = True):
    """First n examples of the seeded shuffled stream, as a HF Dataset (cached)."""
    from datasets import Dataset, Features, Image, Sequence, Value, load_from_disk

    spec = DATASETS[name]
    cache_dir = get_out_dir("data", f"{name}_{spec.split}_n{n}_s{seed}_b{buffer_size}")
    if use_cache and (cache_dir / "dataset_info.json").exists():
        return load_from_disk(str(cache_dir))

    rows = []
    for ex in stream_examples(name, seed=seed, buffer_size=buffer_size):
        rows.append(ex)
        if len(rows) >= n:
            break
    features = Features({
        "id": Value("string"), "group": Value("string"), "image": Image(),
        "question": Value("string"), "answers": Sequence(Value("string")),
    })
    ds = Dataset.from_list(rows, features=features)
    if use_cache:
        ds.save_to_disk(str(cache_dir))
    return ds


def majority_answer(answers: list[str]) -> str:
    """Training target: the most frequent annotator answer (ties -> first seen)."""
    from collections import Counter

    cleaned = [a.strip() for a in answers if a and a.strip()]
    return Counter(cleaned).most_common(1)[0][0] if cleaned else ""


def make_splits(name: str, n_train: int, n_eval: int, seed: int = 42, margin: int = 300):
    """Disjoint train / eval sets drawn from the validation pool, split by image so that no
    image (and none of its questions) appears on both sides."""
    import random

    pool = load_subset(name, n_train + n_eval + margin, seed=seed)
    groups = sorted(set(pool["group"]))
    random.Random(seed).shuffle(groups)
    by_group: dict[str, list[int]] = {}
    for i, g in enumerate(pool["group"]):
        by_group.setdefault(g, []).append(i)
    eval_idx, train_idx = [], []
    for g in groups:
        (eval_idx if len(eval_idx) < n_eval else train_idx).extend(by_group[g])
    if len(train_idx) < n_train:
        raise ValueError(f"pool too small: {len(train_idx)} train examples < {n_train}; raise margin")
    train = pool.select(sorted(train_idx)[:n_train])
    evl = pool.select(sorted(eval_idx)[:n_eval])
    assert not set(train["group"]) & set(evl["group"])
    return train, evl
