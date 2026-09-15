import json
import logging
import types
from pathlib import Path
from typing import List, Dict
import lm_eval
import numpy as np
from datasets import DatasetDict, Dataset


logger = logging.getLogger(__name__)


def lm_simple_eval(lm_eval_options, model, tokenizer, result_name, quick=False):
    if not isinstance(lm_eval_options.num_fewshot, list):
        lm_eval_options.num_fewshot = [lm_eval_options.num_fewshot]
    if quick and not lm_eval_options.quick_tasks:
        return
    for num_fewshot in lm_eval_options.num_fewshot:
        wrapped_model = lm_eval.models.huggingface.HFLM(model, tokenizer=tokenizer,
                                                        batch_size=lm_eval_options.batch_size,
                                                        max_length=lm_eval_options.max_length if lm_eval_options.max_length else None)

        # monkey patch for qwen3
        if 'Qwen3' in model.config.name_or_path:
            import jinja2
            def apply_chat_template(
                    self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
            ) -> str:
                """
                Method to apply a chat template to a list of chat history between user and model.
                """
                try:
                    chat_templated = self.tokenizer.apply_chat_template(
                        chat_history,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        continue_final_message=not add_generation_prompt,
                        enable_thinking=lm_eval_options.enable_thinking if lm_eval_options.enable_thinking else False
                    )
                except jinja2.exceptions.TemplateError:
                    logger.warning(
                        "Failed to apply chat template. removing the system role in chat history."
                    )
                    chat_history = [msg for msg in chat_history if msg["role"] != "system"]
                    chat_templated = self.tokenizer.apply_chat_template(
                        chat_history,
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        continue_final_message=not add_generation_prompt,
                        enable_thinking=lm_eval_options.enable_thinking if lm_eval_options.enable_thinking else False
                    )
                return chat_templated

            wrapped_model.apply_chat_template = types.MethodType(apply_chat_template,
                                                                 wrapped_model)

        results = lm_eval.simple_evaluate(  # call simple_evaluate
            model=wrapped_model,
            tasks=lm_eval_options.tasks if not quick else lm_eval_options.quick_tasks,
            num_fewshot=num_fewshot if num_fewshot else None,
            log_samples=True,
            apply_chat_template=lm_eval_options.apply_chat_template if lm_eval_options.apply_chat_template else False,
            fewshot_as_multiturn=True,
        )

        def _handle_non_serializable(o):
            if isinstance(o, np.int64) or isinstance(o, np.int32):
                return int(o)
            elif isinstance(o, set):
                return list(o)
            else:
                return str(o)

        path = Path(lm_eval_options.output_path)
        if path.is_file():
            raise FileExistsError(f"File already exists at {path}")
        output_path_file = path.joinpath(f"{str(num_fewshot if num_fewshot else 'None')}_{result_name}.json")
        if path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
        dumped = json.dumps(
            results, indent=2, default=_handle_non_serializable, ensure_ascii=False
        )
        output_path_file.open("w", encoding="utf-8").write(dumped)
        logger.info(
            f"lm_eval complete. See report in {str(num_fewshot)}_{result_name}.json located in {lm_eval_options.output_path}")

