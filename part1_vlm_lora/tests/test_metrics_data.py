"""Offline checks for metrics and schema handling (no downloads)."""

import pytest
from PIL import Image

from part1_vlm_lora.src.data import get_answers, normalize_example
from part1_vlm_lora.src.metrics import anls, normalize_vqa, vqa_accuracy
from part1_vlm_lora.src.vqa import clean_answer


def test_normalize_vqa():
    assert normalize_vqa("Two dogs.") == "2 dogs"
    assert normalize_vqa("The  Cat") == "cat"
    assert normalize_vqa("dont") == "don't"
    assert normalize_vqa("3.5") == "3.5"          # decimals kept
    assert normalize_vqa("1,000") == "1000"       # digit commas stripped


def test_vqa_accuracy_official_values():
    # 3+ matching annotators -> 1.0; with leave-one-out averaging, exactly 3 -> 0.9
    assert vqa_accuracy("yes", ["yes"] * 10) == pytest.approx(1.0)
    assert vqa_accuracy("yes", ["yes"] * 3 + ["no"] * 7) == pytest.approx(0.9)
    assert vqa_accuracy("yes", ["yes"] * 2 + ["no"] * 8) == pytest.approx(0.6)
    assert vqa_accuracy("yes", ["yes"] + ["no"] * 9) == pytest.approx(0.3)
    assert vqa_accuracy("maybe", ["no"] * 10) == 0.0
    assert vqa_accuracy("two", ["2"] * 10) == pytest.approx(1.0)


def test_anls():
    assert anls("4512", ["4512"]) == 1.0
    assert anls("4512", ["ABC", "4512"]) == 1.0
    assert anls("45l2", ["4512"]) == pytest.approx(0.75)   # 1 edit / 4 chars
    assert anls("hello", ["world"]) == 0.0                  # NL >= 0.5 -> 0
    assert anls(" INVOICE ", ["invoice"]) == 1.0            # case/space insensitive


def test_get_answers_schemas():
    assert get_answers({"answers": ["a", "b"]}) == ["a", "b"]
    assert get_answers({"answers": [{"answer": "a"}, {"answer": "b"}]}) == ["a", "b"]
    assert get_answers({"answers": {"answer": ["a", "b"], "answer_id": [1, 2]}}) == ["a", "b"]
    assert get_answers({"multiple_choice_answer": "yes"}) == ["yes"]
    assert get_answers({}) == []


def test_normalize_example_ids_and_rgb():
    img = Image.new("L", (8, 8))
    ex = normalize_example({"image": img, "question": " q? ", "answers": ["x"], "questionId": "7", "docId": 3}, 0)
    assert ex["id"] == "7" and ex["group"] == "3" and ex["question"] == "q?"
    assert ex["image"].mode == "RGB"


def test_clean_answer():
    assert clean_answer(" 4512.") == "4512"
    assert clean_answer("Paris\nbecause...") == "Paris"


def test_subset_keeps_compressed_bytes_and_decodes_on_access(tmp_path, monkeypatch):
    """Regression: 2.8k decoded images OOM-killed Colab; rows must stay encoded until accessed."""
    import io

    import part1_vlm_lora.src.data as D
    from part1_vlm_lora.src.vqa import as_rgb

    monkeypatch.setenv("EN4554_OUT", str(tmp_path))

    def jpeg(mode):
        buf = io.BytesIO()
        Image.new(mode, (40, 30), 128).save(buf, format="JPEG")
        return {"bytes": buf.getvalue(), "path": None}

    raw = [{"image": jpeg("L" if i % 2 else "RGB"), "question": f"q{i}?", "answers": ["a"] * 10,
            "question_id": i, "image_id": f"img{i}"} for i in range(6)]
    monkeypatch.setattr(D, "stream_examples",
                        lambda name, seed=42, buffer_size=500: (D.normalize_example(e, i) for i, e in enumerate(raw)))
    norm = D.normalize_example(raw[1], 1)
    assert isinstance(norm["image"], dict)            # not decoded while streaming
    ds = D.load_subset("textvqa", 5, seed=3)
    img = ds[1]["image"]                              # decoded on access
    assert img.size == (40, 30) and img.mode == "L"
    assert as_rgb(img).mode == "RGB"
    assert D.majority_answer(["x", "y", "y"]) == "y"
