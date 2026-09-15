import os
import pickle
from . import data_ds, data

def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)


def make_data(config, model_config, seed, tokenizer, model=None):
    print("loading calibdation data")
    if config.flip_dataset.pickle_dump:
        if os.path.exists(config.flip_dataset.pickle.path):
            with open(config.flip_dataset.pickle.path, 'rb') as f:
                dataloader = pickle.load(f)
        else:
            raise FileNotFoundError
    else:
        if config.flip_dataset.type in ['downstream']:
            dataloader, _ = data_ds.get_loaders(config.flip_dataset.name,
                                                      seed=seed, tokenizer=tokenizer,
                                                      extra_config=config.flip_dataset.extra_config, model=model)
        else:
            dataloader, _ = data.get_loaders(config.flip_dataset.name,
                                                   tokenizer=tokenizer,
                                                   base_model=model_config.name)
        with open(f'{config.flip_dataset.name}.pkl', 'wb') as f:
            pickle.dump(dataloader, f)

    print("dataset loading complete")
    return dataloader
