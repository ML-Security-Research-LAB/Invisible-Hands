import torch
from torch import nn
from tqdm import tqdm
from transformers import Cache
import lm_eval
from lm_eval import tasks
from modules.eval.setup_eval import eval_ppl, eval_lm_eval
from .layerwrapper import *
from .utils import *
import torch.nn.functional as F
from collections import defaultdict

#_________This function is for flipping one bit depending on the type of tensor___________________
def flip_bits(tensor: torch.Tensor, max_val, min_val):

    tensor = tensor.contiguous()
    device = tensor.device
    #_________________________________________________________
        #1- types + decode+ range  
    #_________________________________________________________
    if tensor.dtype == torch.int8:
        int_view = tensor.to(torch.int32).clone()
        def decode(bits):
            return bits.to(torch.int8)

        bit_range = range(7, -1, -1)

    elif tensor.dtype == torch.float16:
        int_view = tensor.view(torch.int16).to(torch.int32).clone()

        def decode(bits):
            return bits.to(torch.int16).view(torch.float16)

        bit_range = range(15, 10, -1)

    elif tensor.dtype == torch.bfloat16:
        int_view = tensor.view(torch.int16).to(torch.int32).clone()

        def decode(bits):
            return bits.to(torch.int16).view(torch.bfloat16)

        bit_range = range(15, 7, -1)

    else:
        raise TypeError(f"Unsupported dtype: {tensor.dtype}")

    flipped_tensor = tensor.clone()

    for bit_position in bit_range:

        bit_mask = torch.tensor(
            1 << bit_position,
            dtype=torch.int32,
            device=device
        )

        flipped_bits = int_view ^ bit_mask
        candidate = decode(flipped_bits)

        candidate_float = candidate.float()

        if (
            not torch.isnan(candidate_float).any()
            and torch.all(
                (candidate_float >= min_val)
                & (candidate_float <= max_val)
            )
        ):
            flipped_tensor = candidate
            break

    return flipped_tensor

