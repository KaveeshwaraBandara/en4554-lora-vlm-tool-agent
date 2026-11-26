"""A tiny, randomly initialised SmolVLM with the same module names as the real one.

Used for offline tests (CPU, no download): checks that adapter targeting, PEFT wrapping
and PiSSA init behave as expected before spending GPU time on the real model.
"""

import torch
from transformers import SmolVLMConfig, SmolVLMForConditionalGeneration

TINY_IMAGE_TOKEN_ID = 99
TINY_VOCAB = 128


def tiny_smolvlm(seed: int = 0) -> SmolVLMForConditionalGeneration:
    torch.manual_seed(seed)
    cfg = SmolVLMConfig(
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=32,
            patch_size=8,
        ),
        text_config=dict(
            model_type="llama",
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=TINY_VOCAB,
            max_position_embeddings=128,
        ),
        scale_factor=2,
        image_token_id=TINY_IMAGE_TOKEN_ID,
    )
    model = SmolVLMForConditionalGeneration(cfg)
    model.eval()
    return model


def tiny_batch(model, batch_size: int = 2, text_len: int = 6):
    """input_ids with one image's worth of image tokens + pixel_values for one image."""
    n_img_tokens = model.model.image_seq_len
    g = torch.Generator().manual_seed(1)
    text = torch.randint(0, TINY_IMAGE_TOKEN_ID - 1, (batch_size, text_len), generator=g)
    img = torch.full((batch_size, n_img_tokens), TINY_IMAGE_TOKEN_ID)
    input_ids = torch.cat([text[:, :2], img, text[:, 2:]], dim=1)
    vc = model.config.vision_config
    pixel_values = torch.randn(batch_size, 1, 3, vc.image_size, vc.image_size, generator=g)
    return dict(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), pixel_values=pixel_values)
