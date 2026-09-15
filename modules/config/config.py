import os
import yaml
from addict import Dict


class Config:
    _instance = None

    class SafeLoaderWithJoin(yaml.SafeLoader):
        pass

    @classmethod
    def join_constructor(cls, loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    def __new__(cls, path=None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.config = None
            if path:
                cls._instance._load_config(path)
        return cls._instance

    def _load_config(self, path):
        self.SafeLoaderWithJoin.add_constructor('!join', self.join_constructor)
        with open(path, 'r') as file:
            config = yaml.load(file, self.SafeLoaderWithJoin)

        self.config = Dict(config)

    def save_config(self, folder_path):
        
        if self.config is not None:
            
            file_path = os.path.join(folder_path, 'config.yml')

            config_dict = self._convert_to_dict(self.config)

            with open(file_path, 'w') as file:
                yaml.dump(config_dict, file, default_flow_style=False)

            return file_path
        else:
            raise ValueError("No configuration has been loaded yet")

    def _convert_to_dict(self, addict_obj):

        if isinstance(addict_obj, Dict):
            return {k: self._convert_to_dict(v) for k, v in addict_obj.items()}
        elif isinstance(addict_obj, list):
            return [self._convert_to_dict(item) for item in addict_obj]
        else:
            return addict_obj

    def get_config(self):
        return self.config

    def __getattr__(self, name):
        return getattr(self.config, name)


if __name__ == '__main__':
    config = Config('example_fuse.yml')
    c = config.get_config()
    print(c)
