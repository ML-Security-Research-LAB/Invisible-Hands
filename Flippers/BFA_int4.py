import torch
from tqdm import tqdm
import lm_eval
from lm_eval import tasks
from . import Flip
from .layerwrapper import *
from .utils import *
import bitsandbytes.functional as bnb_F
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def unpack_int4(qweight):
    """
    Unpack bnb.Params4bit into uint8 per weight.
    Each byte contains 2 weights: high nibble and low nibble.
    """
    data = qweight.data.view(-1)  # flatten 2D -> 1D
    n_weights = data.numel() * 2
    unpacked = torch.empty(n_weights, dtype=torch.uint8, device=data.device)

    unpacked[0::2] = (data >> 4) & 0xF   # high nibble
    unpacked[1::2] = data & 0xF          # low nibble

    return unpacked


def flip_q4_bit_inplace(qweight, weight_idx, bit_pos):
    """
    Flip one bit inside packed int4 tensor (bnb.Params4bit).
    Returns info needed to revert.
    """
    packed = qweight.data.view(-1)

    byte_idx = weight_idx // 2
    high = (weight_idx % 2 == 0)

    old_byte = packed[byte_idx].item()
    byte = packed[byte_idx]

    if high:
        nibble = (byte >> 4) & 0xF
        nibble ^= (1 << bit_pos)
        packed[byte_idx] = (byte & 0x0F) | (nibble << 4)
    else:
        nibble = byte & 0xF
        nibble ^= (1 << bit_pos)
        packed[byte_idx] = (byte & 0xF0) | nibble

    return byte_idx, old_byte


def revert_q4_flip(qweight, byte_idx, old_byte):
    """Restore original byte."""
    qweight.data.view(-1)[byte_idx] = old_byte


class BFA_int4(Flip):

    def __init__(self, model, config, data):
        super().__init__(model, config, data)

    def flip(self):
        func_name = self.config.task.flip.func_name
        if func_name in ['bfa']:
            self.bfa()
        else:
            raise Exception

    def bfa(self):
        log_file   = self.config.report.logger.log_file_path
        with open(log_file, "a") as f:
            f.write("========== Results of BFA (int4)==========\n")
        n_samples = self.process_data()
        seq_len = self.config.task.flip.flip_dataset.seq_len
       
        with torch.no_grad():
            inps, outs, attention_mask, position_ids, cache_position, position_embeddings = prepare_calibration_input(
                self.get_model(), self.data, n_samples, seq_len)

        #_______these are the parameters for the flipping process________
        max_iters = self.config.task.flip.max_iter
        task = self.config.task.flip.flip_dataset.name
        threshold = self.config.task.flip.threshold + 0.02
        layers = nested_getattr(self.model, self.layer_mapping['layers'])
        self.used_flips = set() #for traking and not flipping the same weight again, we store the tuple (layer_id, name, idx) in this set.
        #_________these are the parameters for the stagnation check________
        patience = 5                 # maximum stagnant iterations
        tolerance = 0.005            # accuracy similarity range
        stagnant_counter = 0
        prev_accuracy = None
        #__________start the flipping process for a maximum of max_iters iterations________
        wrapped_model = lm_eval.models.huggingface.HFLM(self.model, tokenizer=self.tokenizer,
                                                batch_size=64, max_length=None)
        results = lm_eval.simple_evaluate(  # call simple_evaluate
                model=wrapped_model,
                tasks=[task],
                num_fewshot=None,
                log_samples=False,
                limit=None,
            )
        intial_accuracy = results["results"][task]["acc,none"]

        for i in range(max_iters):
            torch.cuda.empty_cache()
            with open(log_file, "a") as f:
                f.write(f"Initial accuracy in {i} is {intial_accuracy}\n")
            # ---------- stagnation check ----------
            if prev_accuracy is not None:
                if abs(intial_accuracy - prev_accuracy) < tolerance:
                    stagnant_counter += 1
                    with open(log_file, "a") as f:
                        f.write(f"Accuracy stagnating ({stagnant_counter}/{patience})\n")
                else:
                    stagnant_counter = 0  # improvement happened

            prev_accuracy = intial_accuracy

            if stagnant_counter >= patience:
                with open(log_file, "a") as f:
                    f.write(f"Stopping early: accuracy stagnated for {patience} iterations.\n")
                break
            # --------------------------------------------
            
            best_flips = []
            
            for current_layer_id in tqdm(range(len(layers)), desc="Processing layers"):
                if current_layer_id not in self.config.task.flip.critical_layer_index:
                    inps,candidate = self.step(current_layer_id, inps.detach().clone(), attention_mask, position_ids, cache_position,
                                    position_embeddings, flip=False, acc = intial_accuracy, used_flips=self.used_flips, task=task  )
                
                elif current_layer_id in self.config.task.flip.critical_layer_index:
                    inps,candidate  = self.step(current_layer_id, inps.detach().clone(), attention_mask, position_ids, cache_position,
                                    position_embeddings, flip=True, acc = intial_accuracy, used_flips=self.used_flips, task=task  )
                
                if candidate is not None:
                    best_flips.append(candidate)

            # --- Apply the best flip ---
            if best_flips:
                best_flips.sort(key=lambda x: x['Acc'], reverse=False)
                top_flip = best_flips[0]
                layer_id = top_flip['layer_id']
                idx = top_flip['weight_idx']
                with open(log_file, "a") as f:
                    f.write(f"Applied flip: layer {layer_id} name {top_flip['name']}, index {idx}, ACC {top_flip['Acc']}\n")
                flip_key = (layer_id, top_flip['name'], idx)
                self.used_flips.add(flip_key) 
            intial_accuracy = top_flip['Acc'] if best_flips else intial_accuracy
            if top_flip['Acc'] < threshold:
                            break

    def step(self, layer_id, inps, attention_mask, position_ids, cache_position, position_embeddings,
            flip=True, acc = None, used_flips=None, task=None):

        n_samples = self.process_data()

        self.model.eval()
        self.model.config.use_cache = False
        
        layers = nested_getattr(self.model, self.layer_mapping['layers'])
        layer = layers[layer_id]
        subset = {}
        subset.update({self.layer_mapping['mlp']['d']: find_layers(layer)[self.layer_mapping['mlp']['d']]})
        
        wrapped_layers = {}
        for name in subset:
            dtype = subset[name].weight.data.dtype
            wrapped_layers[name] = WrappedLayer(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                # unwrap tuple input
                if isinstance(inp, (tuple, list)):
                    inp = inp[0]

                # unwrap tuple output (some layers return tuple)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                wrapped_layers[name].add_batch(inp.data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        outs = inps.detach().clone()
        for j in range(n_samples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,
                                cache_position=cache_position, position_embeddings=position_embeddings)[
                    0]
        for h in handles:
            h.remove()

        if not flip:
            return outs,None
        
        best_candidate = None
        optimal_accuracy = acc
        for name in subset:
            # ===== 1. Get quantized weight =====
            qweight = subset[name].weight  # This is a bnb.Params4bit tensor
            old_unpacked = unpack_int4(qweight).clone()
            # ===== 2. Dequantize to float16 on CUDA =====
            w = bnb_F.dequantize_4bit(
                qweight, 
                qweight.quant_state, 
                quant_type=qweight.quant_type, 
                blocksize=qweight.blocksize
                ).to(torch.bfloat16).to('cuda')
            # ===== 3. Compute vulnerability metric =====
            scaler_row = wrapped_layers[name].scaler_row.to(w.device, dtype=torch.bfloat16)
            W_metric = torch.abs(w) *torch.sqrt(scaler_row.reshape(1, -1))
            W_metric_flat = W_metric.view(-1)
            # ===== 4. Select top-k weights by metric =====
            torch.cuda.empty_cache()
            k = W_metric_flat.numel()
            _, topk_indices = torch.topk(W_metric_flat, k)
            
            # ===== 5. Flip bits =====
            for rank, idx in enumerate(topk_indices):
                flip_key = (layer_id, name, idx.item())
                # ---- SKIP previously used flips ----
                if used_flips is not None and flip_key in used_flips:
                    continue
                idx = idx.item()
                # ----- flip directly in Q4 -----
                byte_idx, old_byte = flip_q4_bit_inplace(
                    qweight,
                    weight_idx=idx,
                    bit_pos=3
                )
                
                # ===== Evaluate model =====
                wrapped_model = lm_eval.models.huggingface.HFLM(
                    self.model,
                    tokenizer=self.tokenizer,
                    batch_size=64,
                    max_length=None
                )

                results = lm_eval.simple_evaluate(
                    model=wrapped_model,
                    tasks=[task],
                    num_fewshot=None,
                    log_samples=False,
                    limit=None,
                )

                new_accuracy = results["results"][task]["acc,none"]

                # ===== Check improvement =====
                if new_accuracy < optimal_accuracy:

                    optimal_accuracy = new_accuracy

                    best_candidate = {
                        'layer_id': layer_id,
                        'name': name,
                        'weight_idx': idx,
                        'Acc': new_accuracy
                    }

                    # keep flip → stop searching
                    break

                else:
                    # ----- revert flip -----
                    revert_q4_flip(qweight, byte_idx, old_byte)

                # stop searching if we found good candidate
                if best_candidate is not None:
                    break
                
            # ===== 8. Cleanup =====
            del w, W_metric, W_metric_flat
            torch.cuda.empty_cache()
            wrapped_layers[name].free()
        
        for j in range(n_samples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,
                                cache_position=cache_position, position_embeddings=position_embeddings)[
                    0]
        return outs,best_candidate

