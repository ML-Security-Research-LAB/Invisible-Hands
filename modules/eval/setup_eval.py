import logging
import os
from transformers import AutoModelForCausalLM
import torch
from peft import PeftModel

logger = logging.getLogger(__name__)


def eval(c, model, tokenizer):
    with torch.no_grad():
        model.eval()
        if c.evaluation.ppl:
            from .ppl import eval
            logger.info(eval(model, tokenizer))
            logger.info(f"ppl eval complete.")
        if c.evaluation.lm_eval:
            from .lm_eval_main import lm_simple_eval
            lm_simple_eval(c.evaluation.lm_eval_options, model, tokenizer, "lm_eval_result")

def eval_ppl(model, tokenizer, save=False, save_path=None):
    from .ppl import eval
    result = eval(model, tokenizer)
    logger.info(result)
    logger.info(f"ppl eval complete.")
    if save:
        torch.save(result, save_path)
    return result


def eval_lm_eval(model, tokenizer, c, result_name, quick=False):
    from .lm_eval_main import lm_simple_eval
    lm_simple_eval(c.evaluation.lm_eval_options, model, tokenizer, result_name, quick=quick)



def save_and_eval(c, model, tokenizer, trainer=None):
    eval(c, model, tokenizer)
