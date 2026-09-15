import logging
import os
import transformers
from utils import load_module

logger = logging.getLogger(__name__)


def make_flipper(flip_config):
    if flip_config.custom_flip:
        if flip_config.custom_config.custom_package_location != "":
            custom_package_module = load_module(flip_config.custom_config.custom_package_location)
            flipper_class = getattr(custom_package_module, flip_config.flip_metric)
            return flipper_class
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
