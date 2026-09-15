import importlib.machinery
import importlib.util
import sys
import os
import torch.distributed as dist

def load_module(package_path):
    package_path = os.path.abspath(package_path)
    module_name = os.path.basename(package_path)
    init_file = os.path.join(package_path, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_file,
        submodule_search_locations=[package_path]
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module

import logging, traceback


def setup_logger(config, multi_process=False):
    if not multi_process:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  
        file_handler = logging.FileHandler(config.logger.log_file_path, mode='w')
        file_handler.setLevel(logging.INFO)
    else:
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"logging to rank: {rank}")
        log_file = config.logger.log_file_path.replace('.txt', f'_rank{rank}.txt')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
