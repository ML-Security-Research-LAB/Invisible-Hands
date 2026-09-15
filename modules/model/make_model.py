from .hf import make_hf_model


def make_model(model_config):
    model, tokenizer, config = make_hf_model(model_config)
    return model, tokenizer, config
