import logging
import numpy as np
import random
import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Set the random seed for NumPy and PyTorch for reproducibility.

    Args:
        seed (int): The random seed.
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper class for tokenized input IDs
class TokenizerWrapper:
    """
    Wrapper class for tokenized input IDs.

    Args:
        input_ids (tensor): The tokenized input IDs from the tokenizer.
    """

    def __init__(self, input_ids):
        self.input_ids = input_ids


def estimate_seqlen_arc_easy(ds, tokenizer, percentile=99, sample_size=1000, seed=0):
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    lengths = []
    if sample_size == 0:
        sample_size = len(idxs)
    for i in idxs[:sample_size]:
        ex = ds[i]
        q = ex["question"].strip()
        prompt_base = f"Question: {q}\nAnswer:"
        base_ids = tokenizer(prompt_base, add_special_tokens=False).input_ids
        Lb = len(base_ids)

        options = [t.strip() for t in ex["choices"]["text"]]
        for opt in options:
            opt_ids = tokenizer(opt, add_special_tokens=False).input_ids
            lengths.append(Lb + len(opt_ids))

    return int(np.percentile(lengths, percentile))


def get_arc_easy(max_tokens: int, seed: int, tokenizer):
    
    random.seed(seed)
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy",
                      split="train", trust_remote_code=True).shuffle(seed)

    seqlen = estimate_seqlen_arc_easy(ds, tokenizer,
                                      percentile=99,
                                      sample_size=2000,
                                      seed=seed)
    logger.info(f"[Auto] set seqlen={seqlen} (arc_easy 99th pct)")

    trainloader = []
    total_tokens = 0

    for ex in ds:
        q_text = ex["question"].strip()
        options = [t.strip() for t in ex["choices"]["text"]]
        correct_idx = ex["choices"]["label"].index(ex["answerKey"])

        prompt_base = f"Question: {q_text}\nAnswer:"
        q_enc = tokenizer(prompt_base, return_tensors="pt", add_special_tokens=False)
        q_ids = q_enc.input_ids  # [1, Lq_orig]
        Lq_orig = q_ids.size(1)

        per_opt_lengths = []
        trimmed_qs = []
        for opt in options:
            a_ids = tokenizer(opt, add_special_tokens=False).input_ids
            La = len(a_ids)
            if La >= seqlen:
                per_opt_lengths = None
                break
            max_q = seqlen - La
            Lq_eff = min(Lq_orig, max_q)
            per_opt_lengths.append(Lq_eff + La)
            trimmed_qs.append(q_ids[:, -Lq_eff:] if Lq_orig > max_q else q_ids)

        if per_opt_lengths is None:
            continue

        needed = sum(per_opt_lengths)
        if total_tokens + needed > max_tokens:
            break

        for idx, (opt, length_i, q_trim) in enumerate(zip(options, per_opt_lengths, trimmed_qs)):
            a_ids = tokenizer(opt, add_special_tokens=False).input_ids
            inp = torch.cat([q_trim, torch.tensor([a_ids], dtype=torch.long)], dim=1)
            labels = inp.clone()
            labels[0, :q_trim.size(1)] = -100

            pad_len = seqlen - length_i
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)
                pad_lbls = torch.full((1, pad_len), -100, dtype=torch.long)
                inp = torch.cat([inp, pad_ids], dim=1)
                labels = torch.cat([labels, pad_lbls], dim=1)

            is_positive = (idx == correct_idx)
            trainloader.append((inp, labels, is_positive))

        total_tokens += needed
    if total_tokens < max_tokens:
        logger.info(f"Warning: task arc_easy only generated {total_tokens} tokens, budget was {max_tokens}")
    val_prompts = [
        f"Question: {ex['question'].strip()}\nAnswer:"
        for ex in load_dataset("allenai/ai2_arc", "ARC-Easy",
                               split="validation", trust_remote_code=True)
    ]
    val_text = " ".join(val_prompts)
    val_enc = tokenizer(val_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256 * seqlen).input_ids
    valenc = TokenizerWrapper(val_enc)

    return trainloader, valenc, total_tokens


def get_loaders(name='arc_easy', seed=0, total_budget=512000, tokenizer=None, extra_config=None, model=None, full_config=None):
    if "arc_easy" in name:
        trainloader, valenc, total_tokens = get_arc_easy(total_budget, seed, tokenizer)
        return trainloader, valenc
    else:
        raise ValueError

