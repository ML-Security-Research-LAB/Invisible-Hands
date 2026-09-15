from abc import ABC, abstractmethod
import modules.system.system as system


class Flipper(ABC):
    @abstractmethod
    def __init__(self, model, config, data):
        self.model = model
        self.config = config
        self.data = data
        self.is_peft = config.model.model_peft

    @abstractmethod
    def flip(self):
        pass

    def get_model(self):
        if system.enable_deepspeed:
            return self.model.module
        elif self.is_peft:
            return self.model.model
        else:
            return self.model

    def get_wrapped_model(self):
        if self.is_peft:
            return self.model
        else:
            return self.model

    def get_model_config(self):
        if system.enable_deepspeed:
            return self.model.module.config
        else:
            return self.model.config

    def set_model_config(self, config):
        if system.enable_deepspeed:
            self.model.module.config = config
        else:
            self.model.config = config
