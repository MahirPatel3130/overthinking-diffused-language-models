"""
Step-logging version of the LLaDA sampler.

This is a light modification of ML-GSAI/LLaDA's generate.py
(https://github.com/ML-GSAI/LLaDA/blob/main/generate.py). The sampling math
(Gumbel-max token proposal + low-confidence remasking schedule) is untouched.
The only change: after every inner denoising step, we decode the current
generation buffer to text and append it to a trace list, so we can later ask
"at which step did the correct answer first appear".

Only batch size 1 is supported (one problem at a time) -- this keeps the
step-trace bookkeeping simple, and for this experiment we care about
per-problem step traces, not raw throughput.
"""

import torch
import numpy as np
import torch.nn.functional as F


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate_with_step_trace(
    model,
    tokenizer,
    prompt,                # tensor, shape (1, L)
    attention_mask=None,
    steps=64,
    gen_length=256,
    block_length=256,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    Same sampling loop as the official generate(), but returns a list of
    decoded strings -- one per global denoising step -- instead of only the
    final tensor.

    Returns:
        final_x: the finished token buffer (prompt + generation)
        step_texts: list[str] of length `steps`, the decoded generation
            region after each denoising step, in order.
    """
    assert prompt.shape[0] == 1, "generate_with_step_trace only supports batch size 1"
    device = model.device

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long
    ).to(device)
    x[:, : prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=-1,
        )

    prompt_index = x != mask_id
    prompt_len = prompt.shape[1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    step_texts = []

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt_len + num_block * block_length: prompt_len + (num_block + 1) * block_length] == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt_len + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = int(num_transfer_tokens[j, i])
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            # --- the only real addition vs. the original generate(): ---
            # decode the generation region *as it stands right now* (still-
            # masked positions decode to the mask token's string, which the
            # answer extractor will simply fail to parse, same as GSM8K/
            # MATH intermediate steps that don't have an answer yet).
            gen_ids = x[:, prompt_len:]
            text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            step_texts.append(text)

    return x, step_texts
