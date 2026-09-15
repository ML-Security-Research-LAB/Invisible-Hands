import logging
import os
from modules.data.make_data import make_data
from modules.eval.setup_eval import eval, save_and_eval
from modules.model.make_model import make_model
import modules.system.system as system
from tasks.flipping.make_flipper import make_flipper

logger = logging.getLogger(__name__)


def flip_task(c):
    
    # load model
    model, tokenizer, config = make_model(c.model)

    model.config.use_cache = False

    flip_data = make_data(c.task.flip, c.model, c.task.seed, tokenizer, model)

    flipper_c = make_flipper(c.task.flip)

    model = system.setup_model(model, flip_data, tokenizer, c)

    flipper = flipper_c(model, c, flip_data)

    flipper.tokenizer = tokenizer

    flipper.flip()

    model = flipper.model

    if not system.ddp or system.rank == 0:
        save_and_eval(c, model, tokenizer)
