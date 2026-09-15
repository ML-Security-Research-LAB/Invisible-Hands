import copy
import logging
import math
from abc import abstractmethod
from tqdm import tqdm
import torch
from .utils import *
from tasks.flipping.flippers import Flipper

LAYER_NAME_MAPPING = {
    'Llama': {
        'attn': {
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'o': 'self_attn.o_proj',
            'q_name': 'q_proj',
            'k_name': 'k_proj',
            'v_name': 'v_proj',
            'o_name': 'o_proj',
            'block': 'self_attn'
        },
        'mlp': {
            'd': 'mlp.down_proj',
            'g': 'mlp.gate_proj',
            'u': 'mlp.up_proj',
            'd_name': 'down_proj',
            'g_name': 'gate_proj',
            'u_name': 'up_proj',
            'block': 'mlp'
        },
        'layers': 'model.layers'
    },
    'Mistral': {
        'attn': {
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'o': 'self_attn.o_proj',
            'q_name': 'q_proj',
            'k_name': 'k_proj',
            'v_name': 'v_proj',
            'o_name': 'o_proj',
            'block': 'self_attn'
        },
        'mlp': {
            'd': 'mlp.down_proj',
            'g': 'mlp.gate_proj',
            'u': 'mlp.up_proj',
            'd_name': 'down_proj',
            'g_name': 'gate_proj',
            'u_name': 'up_proj',
            'block': 'mlp'
        },
        'layers': 'model.layers'
    },
    'Qwen': {
        'attn': {
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'o': 'self_attn.o_proj',
            'q_name': 'q_proj',
            'k_name': 'k_proj',
            'v_name': 'v_proj',
            'o_name': 'o_proj',
            'block': 'self_attn'
        },
        'mlp': {
            'd': 'mlp.down_proj',
            'g': 'mlp.gate_proj',
            'u': 'mlp.up_proj',
            'd_name': 'down_proj',
            'g_name': 'gate_proj',
            'u_name': 'up_proj',
            'block': 'mlp'
        },
        'layers': 'model.layers'
    },
    'phi-2': {
        'attn': {
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'o': 'self_attn.dense',
            'q_name': 'q_proj',
            'k_name': 'k_proj',
            'v_name': 'v_proj',
            'o_name': 'dense',
            'block': 'self_attn'
        },
        'mlp': {
            'u': 'mlp.fc1',
            'd': 'mlp.fc2',
            'u_name': 'fc1',
            'd_name': 'fc2',
            'block': 'mlp'
        },
        'layers': 'model.layers'
    },
    'phi-3': {
        'attn': {
            'qkv': 'self_attn.qkv_proj',  # fused QKV
            'o': 'self_attn.o_proj',
            'block': 'self_attn'
        },
        'mlp': {
            'u': 'mlp.gate_up_proj',  # merged gate+up
            'd': 'mlp.down_proj',
            'block': 'mlp'
        },
        'layers': 'model.layers'
    },
    'opt': {
        'attn': {
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'o': 'self_attn.out_proj',
            'block': 'self_attn'
        },
        'mlp': {
            'u': 'fc1',
            'd': 'fc2',
            'block': ''
        },
        'layers': 'base_model.decoder.layers'
    },
    'deepseek': {
        'attn': {
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'o': 'self_attn.o_proj',
            'q_name': 'q_proj',
            'k_name': 'k_proj',
            'v_name': 'v_proj',
            'o_name': 'o_proj',
            'block': 'self_attn'
        },
        'mlp': {
            'd': 'mlp.down_proj',
            'g': 'mlp.gate_proj',
            'u': 'mlp.up_proj',
            'd_name': 'down_proj',
            'g_name': 'gate_proj',
            'u_name': 'up_proj',
            'block': 'mlp'
        },
        'layers': 'model.layers'
    }
}


class Flip(Flipper):
    logger = logging.getLogger(__name__)

    def __init__(self, model, config, data):
        super().__init__(model, config, data)
        self.label_neg = None
        self.data_neg = None
        self.label_pos = None
        self.data_pos = None
        self.pos = None
        self.label = None
        self.use_cache = None
        print("model name:", config.model.name)   
        for k, _ in LAYER_NAME_MAPPING.items():
            if k in config.model.name:
                self.layer_mapping = LAYER_NAME_MAPPING[k]
                self.model_arch = k
                self.model_name = config.model.alias
                break
        if self.layer_mapping is None:
            raise Exception(f'model {config.model.name} is not supported yet')
        self.tokenizer = None
        self.data_processed = False
        self.is_gqa = False

    def obtain_information_critical(self, input=None, stop_index=None):
        n_samples = self.process_data()
        seq_len = self.config.task.flip.flip_dataset.seq_len
        self.model.eval()
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        with torch.no_grad():
            inps, outs, attention_mask, position_ids, cache_position, position_embeddings = prepare_calibration_input_qwen3(
                self.get_model(), self.data, n_samples,seq_len)
        layers = self.get_layers()
        if input is not None:
            inps = input
        if stop_index is None:
            stop_index = len(layers)
        elif stop_index == 0:
            return None, None, None

        def forward_layer(layer, inputs):
            with torch.no_grad():
                if isinstance(layer, nn.Identity):
                    outputs = layer(inputs)
                else:
                    outputs = inputs.detach().clone()
                    for j in range(n_samples):

                        outputs[j] = \
                            layer(inputs[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,
                                  cache_position=cache_position, position_embeddings=position_embeddings)[
                                0]
            return outputs

        current_cos_sim = []
        current_std_dis = []
        current_l2_dis = []
        inputs = inps.detach().clone()
        for j in tqdm(range(0, stop_index), desc="Obtaining following layers' cosine similarity"):
            current_layer = layers[j]
            outputs = forward_layer(current_layer, inputs)
            current_cos_sim.append(cosine_similarity(inputs, outputs)[2])
            l2, l2_t = l2_distance(inputs.float(), outputs.float())
            current_l2_dis.append(l2)
            std_v, std_t = std(inputs.float(), outputs.float())
            current_std_dis.append(std_v)
            inputs, outputs = outputs, inputs
        current_cos_sim = torch.tensor(current_cos_sim).float()
        current_l2_dis = torch.tensor(current_l2_dis).float()
        current_std_dis = torch.tensor(current_std_dis).float()
        self.model.config.use_cache = use_cache
        return current_cos_sim, current_std_dis, current_l2_dis

    def process_data(self):
            if not self.data_processed and self.config.task.flip.flip_dataset.type in ['downstream']:
                first_elems = []
                for tup in self.data:
                    flat = tup[0].view(-1)  
                    nonzero = flat[flat != 0]  
                    first_elems.append(nonzero)
                big_tensor = torch.cat(first_elems, dim=0)  # shape [L], L = sum_i len(tup_i[0])
    
                seq_len = self.config.task.flip.flip_dataset.seq_len
                chunked = big_tensor.split(seq_len)  
                new_data = []
                for i, chunk in enumerate(chunked):
                    if chunk.numel() < seq_len:
                        chunk = F.pad(chunk, (0, seq_len - chunk.numel()), value=0)
                    new_data.append((chunk.unsqueeze(0),))
                self.data = new_data
                self.data_processed = True
                
    
            return len(self.data)
    
    def get_layers(self):
        return nested_getattr(self.get_model(), self.layer_mapping['layers'])

    @abstractmethod
    def flip(self):
        pass
  