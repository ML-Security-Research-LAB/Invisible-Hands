import torch
from tqdm import tqdm
import lm_eval
from .Flip import Flip
from .layerwrapper import *
from .utils import *
from .BFA_utils import *
import logging


class BFA(Flip):

    def __init__(self, model, config, data):
        super().__init__(model, config, data)

    def flip(self):
        func_name = self.config.task.flip.func_name
        if func_name in ['bfa']:
            self.bfa()
        elif func_name in ['detect_critical_layer']:
            self.detect_critical_layer()
        else:
            raise Exception
        
    def bfa(self):
        log_file   = self.config.report.logger.log_file_path
        with open(log_file, "a") as f:
            f.write("========== Results of BFA ==========\n")
        n_samples = self.process_data()
        seq_len = self.config.task.flip.flip_dataset.seq_len

        with torch.no_grad():
            inps, outs, attention_mask, position_ids, cache_position, position_embeddings = prepare_calibration_input(
                self.get_model(), self.data, n_samples, seq_len)
        
        self.wrapped_model = lm_eval.models.huggingface.HFLM(self.model, tokenizer=self.tokenizer,batch_size=64, max_length=None)
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
        results = lm_eval.simple_evaluate(  
            model=self.wrapped_model,
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
                                    position_embeddings, flip=False, acc = intial_accuracy, used_flips=self.used_flips, task=task )
                
                elif current_layer_id in self.config.task.flip.critical_layer_index:
                    inps,candidate = self.step(current_layer_id, inps.detach().clone(), attention_mask, position_ids, cache_position,
                                    position_embeddings, flip=True, acc = intial_accuracy, used_flips=self.used_flips, task=task )
                
                if candidate is not None:
                    best_flips.append(candidate)

            # --- Apply the best flip ---
            if best_flips:
                best_flips.sort(key=lambda x: x['Acc'], reverse=False)
                top_flip = best_flips[0]
                layer_id = top_flip['layer_id']
                idx = top_flip['weight_idx']
                new_val = top_flip['new_value']
                layer = nested_getattr(self.model, self.layer_mapping['layers'])[layer_id]
                target_module = nested_getattr(layer, top_flip['name'])
                flat_weight = target_module.weight.data.view(-1)
                old_val = flat_weight[idx].clone()
                with torch.no_grad():
                    flat_weight[idx] = new_val
                with open(log_file, "a") as f:
                    f.write(f"Applied flip: layer {layer_id} name {top_flip['name']}, index {idx}, old {old_val.item()}, new {new_val.item()}, ACC {top_flip['Acc']}\n")
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
            wrapped_layers[name] = WrappedLayer(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

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
            max_val = subset[name].weight.max().item()
            min_val = subset[name].weight.min().item()

            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
            wrapped_layers[name].scaler_row.reshape((1, -1)))

            W_metric_flat = W_metric.view(-1)
            flat_weight = subset[name].weight.data.view(-1)
            k = flat_weight.numel()
            _, topk_indices = torch.topk(W_metric_flat, k)
            for rank, idx in enumerate(topk_indices):
                flip_key = (layer_id, name, idx.item())
                # ---- SKIP previously used flips ----
                if used_flips is not None and flip_key in used_flips:
                    continue
                idx = idx.item()
                old_val = flat_weight[idx].clone()
                new_val= flip_bits(old_val, max_val, min_val)

                with torch.no_grad():
                    flat_weight[idx] = new_val

                
                results = lm_eval.simple_evaluate(  # call simple_evaluate
                        model=self.wrapped_model,
                        tasks=[task],
                        num_fewshot=None,
                        log_samples=False,
                        limit=None,
                    )
                new_accuracy = results["results"][task]["acc,none"]

                if new_accuracy < optimal_accuracy: 
                    best_candidate = {
                        'layer_id': layer_id,
                        'name': name,
                        'weight_idx': idx,
                        'new_value': new_val,
                        'Acc': new_accuracy
                    }
                    optimal_accuracy = new_accuracy
                    with torch.no_grad():
                        flat_weight[idx] = old_val
                    break  
                elif rank == 49:
                    # revert the change
                    with torch.no_grad():
                        flat_weight[idx] = old_val
                    break 
                else: 
                    with torch.no_grad():
                        flat_weight[idx] = old_val
            
            torch.cuda.empty_cache()
            wrapped_layers[name].free()
        
        for j in range(n_samples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,
                                cache_position=cache_position, position_embeddings=position_embeddings)[
                    0]
        return outs,best_candidate

    def detect_critical_layer(self):
        layers = self.get_layers()
        total = len(layers)
        cos0, std0, _= self.obtain_information_critical()
        
        critical = int(torch.argmax(std0).item())
        print(f"critical layer index is: {critical}.")