import logging
from collections import defaultdict
import torch
from bitsandbytes.nn import Linear8bitLt, Linear4bit
from torch import nn
from transformers import Conv1D
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def cosine_similarity(A, B):
    sim = F.cosine_similarity(A, B, dim=-1)
    token_mean = torch.mean(sim, dim=1)
    sample_mean = torch.mean(token_mean, dim=0)
    return sim, token_mean, sample_mean

def l2_distance(A, B):
    l2_distance = torch.norm(A - B)
    return l2_distance, l2_distance


def std(A, B):
    std_A = torch.std(A)
    std_B = torch.std(B)
    std_d = torch.abs(std_A - std_B)
    std_token_wise_A = torch.std(A, dim=(1, 2)).mean(dim=0)
    std_token_wise_B = torch.std(B, dim=(1, 2)).mean(dim=0)
    std_token_wise = torch.abs(std_token_wise_A - std_token_wise_B)
    return std_d, std_token_wise


def nested_getattr(obj, attr):
    for attr_part in attr.split('.'):
        obj = getattr(obj, attr_part)
    return obj


def prepare_calibration_input_qwen3(model, dataloader, nsamples, seqlen):
    """
    Prepare inputs for model calibration.

    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded.

    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        layers = model.model.layers
    except AttributeError:
        try:
            layers = model.base_model.layers
        except AttributeError:
            layers = model.base_model.decoder.layers
    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]
    else:
        device = model.device
    dtype = next(iter(model.parameters())).dtype
    # dtype = torch.float
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "cache_position": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "self_attn"):
                self.self_attn = module.self_attn
            elif hasattr(module, "attn"):
                self.attn = module.attn

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            if 'cache_position' in kwargs and 'position_embeddings' in kwargs:
                cache['cache_position'] = kwargs['cache_position']
                cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # model(torch.rand_like(batch[0].to(device)))
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    if 'cache_position' in cache and 'position_embeddings' in cache:
        cache_position = cache['cache_position']
        position_embeddings = cache['position_embeddings']
        return inps, outs, attention_mask, position_ids, cache_position, position_embeddings
    return inps, outs, attention_mask, position_ids

def prepare_calibration_input(model, dataloader, nsamples, seqlen):
    """
    Prepare inputs for model calibration.

    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded.

    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        layers = model.model.layers
    except AttributeError:
        try:
            layers = model.base_model.layers
        except AttributeError:
            layers = model.base_model.decoder.layers
    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]
    else:
        device = model.device
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, "cache_position": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            #self.module = module
            """if hasattr(module, "self_attn"):
                self.self_attn = module.self_attn
            elif hasattr(module, "attn"):
                self.attn = module.attn"""
            object.__setattr__(self, "module", module)

        def __getattr__(self, name):
            return getattr(self.module, name)


        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            if 'cache_position' in kwargs and 'position_embeddings' in kwargs:
                cache['cache_position'] = kwargs['cache_position']
                cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # model(torch.rand_like(batch[0].to(device)))
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    if 'cache_position' in cache and 'position_embeddings' in cache:
        cache_position = cache['cache_position']
        position_embeddings = cache['position_embeddings']
        return inps, outs, attention_mask, position_ids, cache_position, position_embeddings
    return inps, outs, attention_mask, position_ids


def find_layers(module, layers=[nn.Linear, Conv1D, Linear8bitLt, Linear4bit], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
