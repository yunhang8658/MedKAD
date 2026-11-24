# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import re
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", enable_format_reward=False, enable_think_reward=False) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            enable_format_reward: Whether to enable format reward. Defaults to False.
            enable_think_reward: Whether to enable the independent analysis-length
                reward before ``<answer>``. It does not require literal
                ``<think>`` tags and is independent from enable_format_reward.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.enable_format_reward = enable_format_reward
        self.enable_think_reward = enable_think_reward

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if "rm_scores" in data.batch.keys():
        #     if return_dict:
        #         reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
        #         reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
        #         return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
        #     else:
        #         return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        format_tensor = torch.zeros(data.batch["responses"].shape[0], dtype=torch.float32) # (batch_size, )
        think_tensor = torch.zeros(data.batch["responses"].shape[0], dtype=torch.float32) # (batch_size, )
        # Detached answer correctness is only a VA-selection mask, never a reward.
        answer_correct_tensor = torch.zeros(data.batch["responses"].shape[0], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            if self.enable_format_reward:
                # The answer tag is the only required structural delimiter.  The
                # model may write its reasoning as ordinary text before it; a
                # literal <think>...</think> wrapper is not required.
                has_answer_tag = bool(re.search(r"<answer>\s*.+?\s*</answer>", response_str, flags=re.DOTALL | re.IGNORECASE))
                format_score = 1.0 if has_answer_tag else 0.0
                format_tensor[i] = format_score

            if self.enable_think_reward:
                # Dense independent reasoning signal.  Reasoning is the non-empty
                # text before the answer tag; no literal think tags are required.
                answer_match = re.search(r"<answer>\s*.+?\s*</answer>", response_str, flags=re.DOTALL | re.IGNORECASE)
                has_answer = bool(answer_match)
                pre_answer = response_str.split("<answer>", 1)[0] if re.search(r"<answer>", response_str, flags=re.IGNORECASE) else response_str
                try:
                    pre_answer_tokens = len(self.tokenizer.encode(pre_answer, add_special_tokens=False))
                except Exception:
                    pre_answer_tokens = len(pre_answer.split())

                if has_answer and pre_answer_tokens >= 192:
                    think_score = 1.0
                elif has_answer and pre_answer_tokens >= 128:
                    think_score = 0.75
                elif has_answer and pre_answer_tokens >= 64:
                    think_score = 0.50
                elif has_answer and pre_answer_tokens >= 32:
                    think_score = 0.25
                else:
                    think_score = 0.0
                think_tensor[i] = think_score

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                # Accuracy is retained for evaluation and for the detached VA
                # selection mask. It is never converted into a training reward.
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                answer_correct_tensor[i] = 1.0 if score.get("acc", False) else 0.0
            else:
                answer_correct_tensor[i] = 0.0

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # rm_scores contains the teacher-distillation signal. Answer correctness
        # is never written into reward_tensor.
        if "rm_scores" in data.batch.keys():
            print(f"Now we are using rm_scores!")
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                if self.enable_format_reward:
                    print("Format mask has been added to reward_extra_info!")
                    reward_extra_info["format_mask"] = format_tensor
                if self.enable_think_reward:
                    print("Think mask has been added to reward_extra_info!")
                    reward_extra_info["think_mask"] = think_tensor
                print("Detached answer-correctness mask has been added to reward_extra_info!")
                reward_extra_info["answer_correct_mask"] = answer_correct_tensor
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        if return_dict:
            reward_extra_info["answer_correct_mask"] = answer_correct_tensor
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
