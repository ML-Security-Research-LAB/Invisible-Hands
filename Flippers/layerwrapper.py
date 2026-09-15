import torch
import torch.nn as nn
import bitsandbytes as bnb

class WrappedLayer:
    
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0
        self.feb_output = None
        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if self.scaler_row.numel() == 1:
            self.scaler_row = torch.zeros(inp.shape[0], device=inp.device)
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

    def free(self):
        self.scaler_row = None
        torch.cuda.empty_cache()
