import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaConfig, \
    Qwen2Config, Qwen2ForCausalLM, MistralConfig, MistralForCausalLM, \
    Qwen3Config, Qwen3ForCausalLM,GenerationConfig
from .utils import str_to_torch_dtype, get_device_map
from utils import load_module
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)


def make_hf_model(model_config):
    if 'llama' in model_config.name or 'Llama' in model_config.name:
        bnb_config = None
        if model_config.load_in_8bit or model_config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=model_config.load_in_8bit,
                load_in_4bit=model_config.load_in_4bit,
            )
        config = LlamaConfig.from_pretrained(model_config.name, attn_implementation='eager')
        model = LlamaForCausalLM.from_pretrained(
            model_config.name,
            config=config,
            torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
            quantization_config=bnb_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_config.name)
        tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference
        
    elif 'Mistral' in model_config.name or 'mistral' in model_config.name:
            
        bnb_config = None
        if model_config.load_in_8bit or model_config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=model_config.load_in_8bit,
                load_in_4bit=model_config.load_in_4bit,
            )
        config = MistralConfig.from_pretrained(model_config.name, attn_implementation='eager')
        model = MistralForCausalLM.from_pretrained(
            model_config.name,
            config=config,
            torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
            quantization_config=bnb_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_config.name)
        tokenizer.pad_token_id = (
            0  
        )
        tokenizer.padding_side = "left"  # Allow batched inference

    elif 'deepseek' in model_config.name:
        
            bnb_config = None
            if model_config.load_in_8bit or model_config.load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=model_config.load_in_8bit,
                    load_in_4bit=model_config.load_in_4bit,
                )
            config = GenerationConfig.from_pretrained(model_config.name,trust_remote_code=True,)
            model = AutoModelForCausalLM.from_pretrained(
                model_config.name,
                generation_config = config,
                torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
                # device_map=get_device_map(),
                trust_remote_code=True,
                quantization_config=bnb_config,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.name,
                trust_remote_code=True,
                use_fast=False
            )
            if model.generation_config.pad_token_id is None:
                model.generation_config.pad_token_id = model.generation_config.eos_token_id

            tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
            tokenizer.padding_side = "left"  # Allow batched inference
            
    return model, tokenizer, config



