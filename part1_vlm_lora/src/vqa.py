"""Prompting and batched greedy generation for SmolVLM on short-answer VQA.

The same prompt is used for zero-shot evaluation and fine-tuning, so the measured
gains come from the adapters and not from a prompt change.
"""

from __future__ import annotations

import time

import torch

PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."


def build_messages(question: str, answer: str | None = None) -> list[dict]:
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": question.strip() + PROMPT_SUFFIX}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def set_resolution(processor, longest_edge: int | None) -> None:
    """Images are resized so the longest edge = longest_edge, then tiled into 512 px crops
    (64 tokens each for SmolVLM-256M) plus one downscaled global view. None keeps the
    processor default. Fewer tiles = shorter sequences = faster training."""
    if longest_edge:
        processor.image_processor.size = {"longest_edge": int(longest_edge)}


def as_rgb(img):
    return img if img.mode == "RGB" else img.convert("RGB")


def clean_answer(text: str) -> str:
    text = text.strip().split("\n")[0].strip()
    return text.rstrip(".").strip()


@torch.no_grad()
def generate_answers(model, processor, examples, batch_size: int = 4, max_new_tokens: int = 16):
    """Returns a list of dicts: pred, raw, n_input_tokens, n_image_tokens, seconds (per example)."""
    tok = processor.tokenizer
    tok.padding_side = "left"
    image_token_id = model.config.image_token_id
    use_amp = model.device.type == "cuda"
    results = []
    for start in range(0, len(examples), batch_size):
        batch = [examples[i] for i in range(start, min(start + batch_size, len(examples)))]
        prompts = [processor.apply_chat_template(build_messages(e["question"]), add_generation_prompt=True)
                   for e in batch]
        images = [[as_rgb(e["image"])] for e in batch]
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(model.device)

        if use_amp:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if use_amp:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / len(batch)

        n_in = inputs["input_ids"].shape[1]
        raw = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)
        n_tokens = inputs["attention_mask"].sum(dim=1).tolist()
        n_img = (inputs["input_ids"] == image_token_id).sum(dim=1).tolist()
        for r, nt, ni in zip(raw, n_tokens, n_img):
            results.append({"pred": clean_answer(r), "raw": r, "n_input_tokens": int(nt),
                            "n_image_tokens": int(ni), "seconds": dt})
    return results
