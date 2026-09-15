import os
import random
import fire
import numpy as np
import psutil
import torch
import sys
import utils
from modules.config.config import Config
from pathlib import Path
import modules.system.system as system
from tasks.flipping.flip import flip_task
from tasks.test.test import test_task



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(config_path: str = ''):

    config = Config(config_path)
    c = config.get_config()
    Path(c.task.output_folder).mkdir(exist_ok=True)
    config.save_config(c.task.output_folder)
    system.init_system(c)
    logger = utils.setup_logger(c.report, multi_process=system.world_size > 1)
    logger.info(f"loaded config: {config_path}")
    logger.info(c)

    
    set_random_seed(config.task.seed)
    if c.task.task_mode == 'flip':
        flip_task(c)
    elif c.task.task_mode == 'test':
        test_task(c)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    print(" ".join(sys.argv))
    fire.Fire(main)
