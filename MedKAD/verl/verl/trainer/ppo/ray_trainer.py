# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


def _json_default(value):
    """Convert tensor-like validation metadata into JSON-compatible values."""
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


@torch.no_grad()
def compute_overlap_token_advantage_metric(
    student_top_k_log_probs: torch.Tensor,         # (B, T, K)  log p_t on S_t^(p)
    teacher_on_student_log_probs: torch.Tensor,    # (B, T, K)  log q_t on S_t^(p) (same index order)
    overlap_mask: torch.Tensor,                    # (B, T, K)  float/bool, 1 if v ∈ S_t^(p) ∩ S_t^(q)
    response_mask: torch.Tensor,                   # (B, T)
) -> dict:
    """
    Overlap-Token Advantage (Li et al., 2026, Eq. 7):

        A_t(v) = p_bar_t(v) * ( log q_bar_t(v) - log p_bar_t(v) ),   v ∈ S_t^(p) ∩ S_t^(q)
        M_adv  = E_t [ (1 / |S_t^(p)∩S_t^(q)|) * sum_v A_t(v) ]

    p_bar, q_bar are p_t and q_t renormalized over the intersection.

    Notes:
      * This is a *distributional* metric, NOT the RL advantage from GAE/GRPO.
      * Paper Fig.6 shows successful OPD: M_adv climbs from ~-0.006 to ~0;
        failing OPD: stagnates around -0.003.
      * Returns 0.0 if no valid (response_mask, |intersection|>0) position exists.
    """
    inter = overlap_mask > 0.5  # (B, T, K) bool

    NEG_INF = torch.finfo(student_top_k_log_probs.dtype).min
    logp_m = torch.where(inter, student_top_k_log_probs,
                         torch.full_like(student_top_k_log_probs, NEG_INF))
    logq_m = torch.where(inter, teacher_on_student_log_probs,
                         torch.full_like(teacher_on_student_log_probs, NEG_INF))

    # Renormalize on the intersection
    log_Zp = torch.logsumexp(logp_m, dim=-1, keepdim=True)   # (B, T, 1)
    log_Zq = torch.logsumexp(logq_m, dim=-1, keepdim=True)
    log_pbar = logp_m - log_Zp                                # -inf outside intersection
    log_qbar = logq_m - log_Zq

    pbar = log_pbar.exp()  # 0 outside intersection
    a_v = pbar * (log_qbar - log_pbar)
    a_v = torch.where(inter, a_v, torch.zeros_like(a_v))      # kill any 0*inf -> nan
    sum_a = a_v.sum(dim=-1)                                    # (B, T)
    inter_sz = inter.sum(dim=-1).to(sum_a.dtype)               # (B, T)

    rmask = response_mask.bool()
    valid = rmask & (inter_sz > 0)
    m_adv = (sum_a[valid] / inter_sz[valid]).mean().item() if valid.any() else 0.0

    # Bonus: overlap mass (paper §B.1 Fig.18) — cheap to add, important sanity check
    overlap_f = inter.to(student_top_k_log_probs.dtype)
    p = student_top_k_log_probs.exp()
    q = teacher_on_student_log_probs.exp()
    if rmask.any():
        p_topk_sum = p.sum(dim=-1).clamp(min=1e-12)
        q_topk_sum = q.sum(dim=-1).clamp(min=1e-12)
        p_overlap_mass = ((p * overlap_f).sum(dim=-1) / p_topk_sum)[rmask].mean().item()
        q_overlap_mass = ((q * overlap_f).sum(dim=-1) / q_topk_sum)[rmask].mean().item()
    else:
        p_overlap_mass = 0.0
        q_overlap_mass = 0.0

    return {
        "opd/overlap_token_advantage": m_adv,
        "opd/student_overlap_mass": p_overlap_mass,
        "opd/teacher_overlap_mass": q_overlap_mass,
    }

def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute uniform Standard OPD or KAlign token weights."""
    del gamma, lam, num_repeat, norm_adv_by_std_in_grpo
    estimator = str(adv_estimator)
    if estimator not in ("STANDARD_OPD", "KALIGN_OPD"):
        raise ValueError(f"Isolated runtime does not support {adv_estimator!r}")
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    kalign_kwargs = {}
    if estimator == "KALIGN_OPD":
        score_mode = str(data.meta_info.get("kalign_score_mode", os.getenv("KALIGN_SCORE_MODE", "cosine")))
        if score_mode not in ("cosine", "sampled_logprob_diff"):
            raise ValueError(f"Unsupported KALIGN_SCORE_MODE={score_mode!r}")
        kalign_kwargs["score_mode"] = score_mode
        if data.meta_info.get("kalign_union_topk", False):
            source_keys = {
                "student_top_k_log_probs": "kalign_similarity_student_log_probs",
                "answer_teacher_on_student_log_probs": "kalign_similarity_answer_log_probs",
                "teacher_on_student_log_probs": "kalign_similarity_knowledge_log_probs",
                "candidate_mask": "kalign_similarity_mask",
            }
        else:
            source_keys = {key: key for key in (
                "student_top_k_log_probs",
                "answer_teacher_on_student_log_probs",
                "teacher_on_student_log_probs",
            )}
        for argument, key in source_keys.items():
            if key not in data.batch:
                raise RuntimeError(f"KAlign requires {key}")
            kalign_kwargs[argument] = data.batch[key]
        if score_mode == "sampled_logprob_diff":
            for argument, key in (
                ("answer_teacher_sampled_log_probs", "answer_teacher_sampled_log_probs"),
                ("knowledge_teacher_sampled_log_probs", "teacher_sampled_log_probs"),
            ):
                if key not in data.batch:
                    raise RuntimeError(f"KAlign direct-difference mode requires {key}")
                kalign_kwargs[argument] = data.batch[key]
        position_mask_mode = os.getenv("POSITION_MASK_MODE")
        if position_mask_mode:
            if position_mask_mode not in {"none", "random", "high_m", "low_m", "high_c", "low_c"}:
                raise ValueError(f"Unsupported POSITION_MASK_MODE={position_mask_mode!r}")
            if not data.meta_info.get("kalign_union_topk", False):
                raise RuntimeError("Position masking requires deduplicated student/knowledge Union Top-K")
            responses = data.batch["responses"]
            eligible = data.batch["response_mask"].bool().clone()
            eos_ids = data.meta_info.get("eos_token_id")
            if eos_ids is None:
                raise RuntimeError("Position masking needs eos_token_id to exclude EOS from selection")
            if isinstance(eos_ids, (list, tuple, set)):
                for eos_id in eos_ids:
                    eligible &= responses != int(eos_id)
            else:
                eligible &= responses != int(eos_ids)
            sample_keys = data.non_tensor_batch.get("index", data.non_tensor_batch["uid"])
            kalign_kwargs.update(
                position_mask_mode=position_mask_mode,
                position_mask_eligible=eligible,
                position_mask_sample_keys=np.asarray(sample_keys, dtype=object),
                position_mask_global_step=int(data.meta_info.get("global_steps", 0)),
                position_mask_seed=int(os.getenv("POSITION_MASK_SEED", os.getenv("EXPERIMENT_SEED", "42"))),
                position_mask_fraction=float(os.getenv("POSITION_MASK_FRACTION", "0.1")),
            )
    weights, returns, extra = core_algos.get_adv_estimator_fn(adv_estimator)(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=data.batch["response_mask"],
        index=data.non_tensor_batch["uid"],
        config=config,
        **kalign_kwargs,
    )
    data.batch["advantages"] = weights
    data.batch["returns"] = returns
    for key, value in extra.items():
        data.batch[key] = value
    return data


def _clean_ds(name: str) -> str:
    """剥掉已知的冗余前缀，让 SwanLab 指标更短。"""
    if isinstance(name, str) and name.startswith("OmniMedVQA_"):
        return name[len("OmniMedVQA_"):]
    return name

class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False, default=_json_default))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set(
            {"data_source", "reward_model", "extra_info", "knowledge_hint", "uid"}
        ) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )

            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")

            def _run_val_generation(gen_batch):
                gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
                if not self.async_rollout_mode:
                    out_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                else:
                    out_padded = self.async_rollout_manager.generate_sequences(gen_batch_padded)
                return unpad_dataproto(out_padded, pad_size=pad_size)

            def _score_val(meta_batch, out_batch):
                # Union onto the freshly generated batch so meta_batch (whose prompt
                # tensors were popped by _get_gen_batch) is left unmutated and can be
                # reused for retry rounds.
                merged = out_batch.union(meta_batch)
                merged.meta_info["validate"] = True
                res = self.val_reward_fn(merged, return_dict=True)
                s = res["reward_tensor"].sum(-1).cpu().tolist()
                ei = {k: list(v) for k, v in res.get("reward_extra_info", {}).items()}
                out_ids = out_batch.batch["responses"]
                texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in out_ids]
                return s, ei, texts

            # ---- round 1: full generation + scoring ----
            test_output_gen_batch = _run_val_generation(test_gen_batch)
            print("validation generation end")
            scores, batch_extra, output_texts = _score_val(test_batch, test_output_gen_batch)

            # ---- retry answer-format-failed generations ----
            # Only the answer tag is required. Reasoning may be ordinary text and
            # does not need literal <think> wrappers.
            # Each retry is a fresh stochastic sample with the same prompt/config.
            val_max_rounds = int(
                getattr(self.config.actor_rollout_ref.rollout.val_kwargs, "format_max_rounds", 1) or 1
            )
            preds = batch_extra.get("pred", None)
            if preds is not None and val_max_rounds > 1:
                round_num = 1
                while round_num < val_max_rounds:
                    has_answer = batch_extra.get("has_answer_tag", [False] * len(preds))
                    failed = [
                        i for i, p in enumerate(preds)
                        if p is None or str(p) == ""
                        or not bool(has_answer[i])
                    ]
                    if not failed:
                        break
                    round_num += 1
                    print(
                        f"[val-format] round {round_num}: re-generating {len(failed)} "
                        f"samples missing a complete <answer> tag"
                    )
                    sub_gen = test_gen_batch.select_idxs(failed)
                    sub_meta = test_batch.select_idxs(failed)
                    sub_out = _run_val_generation(sub_gen)
                    sub_scores, sub_extra, sub_texts = _score_val(sub_meta, sub_out)
                    sub_preds = sub_extra.get("pred", [None] * len(failed))
                    for j, idx in enumerate(failed):
                        scores[idx] = sub_scores[j]
                        output_texts[idx] = sub_texts[j]
                        preds[idx] = sub_preds[j] if j < len(sub_preds) else None
                        for key, lst in sub_extra.items():
                            if key in batch_extra and j < len(lst):
                                batch_extra[key][idx] = lst[j]

            sample_outputs.extend(output_texts)
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, lst in batch_extra.items():
                reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(scores)))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            ds_display = _clean_ds(data_source) 
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    # pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    pfx = f"{metric_sec}/{ds_display}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        # ---- validation-level counters ----
        # num_test_generations: total generations scored (repeated, i.e. n per question).
        # num_test_samples: unique test questions = total generations / val_kwargs.n.
        val_n = int(self.config.actor_rollout_ref.rollout.val_kwargs.n or 1)
        total_gens = len(sample_scores)
        metric_dict["val-aux/num_test_generations"] = total_gens
        metric_dict["val-aux/num_test_samples"] = (total_gens / val_n) if val_n else total_gens

        # format_accuracy: fraction of generations that yielded a parseable answer
        # (pred != ""). Mirrors eval_omnimedvqa.py's format_pass_rate and lets us track
        # whether the retry-based validation improves format compliance.
        val_preds = reward_extra_infos_dict.get("pred", [])
        if val_preds:
            fmt_ok = sum(1 for p in val_preds if p is not None and str(p) != "")
            metric_dict["val-core/format_accuracy"] = fmt_ok / len(val_preds)

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        skip_dataloader_state = os.getenv(
            "VERL_SKIP_DATALOADER_STATE_LOAD", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        if skip_dataloader_state:
            print(
                "VERL_SKIP_DATALOADER_STATE_LOAD=1: keeping the current "
                "diagnostic dataloader instead of restoring data.pt"
            )
        elif os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if self.config.algorithm.adv_estimator == "KALIGN_OPD":
            # Keep all sibling rollouts on one rank and in one actor mini-batch.
            # Dynamic micro-batches can then split tokens without splitting a
            # question-level normalization group.
            groups = defaultdict(list)
            for row, uid in enumerate(batch.non_tensor_batch["uid"]):
                groups[str(uid)].append(row)
            rollout_n = int(self.config.actor_rollout_ref.rollout.n)
            mini_size = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
            if any(len(rows) != rollout_n for rows in groups.values()):
                raise RuntimeError("KAlign question group does not contain all rollouts")
            if mini_size % rollout_n or len(groups) % world_size:
                raise RuntimeError("KAlign needs complete question groups per rank and mini-batch")
            target_groups = len(groups) // world_size
            group_rows = list(groups.values())
            workloads = [sum(int(global_seqlen_lst[row]) for row in rows) for rows in group_rows]
            partitions = [[] for _ in range(world_size)]
            totals = [0] * world_size
            for group_idx in sorted(range(len(group_rows)), key=lambda i: workloads[i], reverse=True):
                rank = min((r for r in range(world_size) if len(partitions[r]) < target_groups), key=lambda r: totals[r])
                partitions[rank].append(group_idx)
                totals[rank] += workloads[group_idx]
            order = [row for partition in partitions for group_idx in partition for row in group_rows[group_idx]]
            batch.reorder(torch.tensor(order, dtype=torch.long))
            metrics["kalign/question_groups_per_rank"] = target_groups
            return
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """Train Standard OPD with student rollouts and frozen-teacher top-k KL."""
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        if not self.use_rm:
            raise RuntimeError("Standard OPD requires a frozen teacher reward-model worker")
        if self.use_critic or self.config.actor_rollout_ref.actor.use_kl_loss:
            raise RuntimeError("Standard OPD does not use a critic or auxiliary KL loss")
        if self.config.algorithm.adv_estimator not in ("STANDARD_OPD", "KALIGN_OPD"):
            raise RuntimeError("This isolated trainer supports only Standard OPD and KAlign OPD")
        if os.getenv("KALIGN_ENABLED", "0") == "1" and self.config.algorithm.adv_estimator != "KALIGN_OPD":
            raise RuntimeError("KALIGN_ENABLED requires algorithm.adv_estimator=KALIGN_OPD")
        if os.getenv("KALIGN_UNION_TOPK", "0") == "1" and self.config.algorithm.adv_estimator != "KALIGN_OPD":
            raise RuntimeError("KALIGN_UNION_TOPK requires algorithm.adv_estimator=KALIGN_OPD")
        kalign_score_mode = os.getenv("KALIGN_SCORE_MODE", "cosine")
        if kalign_score_mode not in ("cosine", "sampled_logprob_diff"):
            raise RuntimeError(f"Unsupported KALIGN_SCORE_MODE={kalign_score_mode!r}")
        if kalign_score_mode != "cosine" and self.config.algorithm.adv_estimator != "KALIGN_OPD":
            raise RuntimeError("Non-cosine KALIGN_SCORE_MODE requires algorithm.adv_estimator=KALIGN_OPD")
        if self.config.algorithm.adv_estimator == "KALIGN_OPD":
            if not self.config.trainer.balance_batch:
                raise RuntimeError("KAlign requires question-group-preserving batch balancing")
            if self.config.reward_model.inject_mode != "knowledge_gt_contrastive":
                raise RuntimeError("KAlign requires the knowledge-and-answer teacher prompt")
            rollout_n = int(self.config.actor_rollout_ref.rollout.n)
            mini_size = int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size)
            if mini_size % rollout_n:
                raise RuntimeError("KAlign actor mini-batch must contain complete rollout groups")
            position_mask_mode = os.getenv("POSITION_MASK_MODE")
            if position_mask_mode:
                valid_position_mask_modes = {"none", "random", "high_m", "low_m", "high_c", "low_c"}
                if position_mask_mode not in valid_position_mask_modes:
                    raise RuntimeError(f"Unsupported POSITION_MASK_MODE={position_mask_mode!r}")
                if os.getenv("KALIGN_UNION_TOPK", "0") != "1":
                    raise RuntimeError("POSITION_MASK_MODE requires KALIGN_UNION_TOPK=1")
                if kalign_score_mode != "cosine":
                    raise RuntimeError("Position masking requires cosine knowledge diagnostics")
        self.global_steps = 0
        self._load_checkpoint()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            initial_metrics = self._validate()
            if initial_metrics:
                logger.log(data=initial_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Standard OPD Training",
        )
        self.global_steps += 1
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {"train/epoch": epoch}
                timing_raw = {}
                batch = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        if self.async_rollout_mode:
                            gen_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        else:
                            gen_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_output.meta_info.get("timing", {}))
                        gen_output.meta_info.pop("timing", None)
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    batch = batch.union(gen_output)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    with marked_timer("student_log_prob", timing_raw, color="blue"):
                        student_data = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(student_data)
                    top_k = int(self.config.actor_rollout_ref.rollout.get("log_prob_top_k", 0))
                    if top_k <= 0 or "student_top_k_ids" not in batch.batch:
                        raise RuntimeError("Standard OPD requires student top-k candidate IDs")
                    batch.meta_info["global_steps"] = self.global_steps
                    batch.meta_info["eos_token_id"] = self.tokenizer.eos_token_id
                    batch.meta_info["log_prob_top_k"] = top_k
                    batch.meta_info["top_k_strategy"] = self.config.actor_rollout_ref.rollout.get(
                        "top_k_strategy", "only_stu"
                    )
                    batch.meta_info["teacher_temperature"] = self.config.actor_rollout_ref.rollout.get(
                        "teacher_temperature", 1.0
                    )
                    batch.meta_info["kalign_mode"] = self.config.algorithm.adv_estimator == "KALIGN_OPD"
                    batch.meta_info["kalign_union_topk"] = (
                        batch.meta_info["kalign_mode"] and os.getenv("KALIGN_UNION_TOPK", "0") == "1"
                    )
                    batch.meta_info["kalign_score_mode"] = (
                        os.getenv("KALIGN_SCORE_MODE", "cosine") if batch.meta_info["kalign_mode"] else "cosine"
                    )
                    with marked_timer("teacher_log_prob", timing_raw, color="magenta"):
                        teacher_data = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(teacher_data)
                    if batch.meta_info["kalign_union_topk"]:
                        candidate_data = DataProto.from_dict(
                            tensors={
                                key: batch.batch[key] for key in (
                                    "input_ids", "attention_mask", "position_ids", "responses"
                                )
                            } | {"target_ids": batch.batch["teacher_top_k_ids"]},
                            non_tensors=dict(batch.non_tensor_batch),
                            meta_info=dict(batch.meta_info),
                        )
                        student_on_knowledge = self.actor_rollout_wg.compute_log_probs_for_ids(candidate_data)
                        batch = batch.union(student_on_knowledge)
                    with marked_timer("distillation_reward", timing_raw, color="orange"):
                        distillation_data = self.actor_rollout_wg.compute_distillation_reward(batch)
                        batch = batch.union(distillation_data)
                    if batch.meta_info["kalign_union_topk"]:
                        support = core_algos.build_kalign_similarity_union(
                            student_ids=batch.batch["student_top_k_ids"],
                            knowledge_teacher_ids=batch.batch["teacher_top_k_ids"],
                            student_on_student_ids=batch.batch["student_top_k_log_probs"],
                            student_on_knowledge_ids=batch.batch["student_log_probs_on_teacher_ids"],
                            answer_on_union_ids=batch.batch["answer_teacher_on_union_log_probs"],
                            knowledge_on_student_ids=batch.batch["teacher_on_student_log_probs"],
                            knowledge_on_knowledge_ids=batch.batch["teacher_top_k_log_probs"],
                        )
                        for key, value in support.items():
                            batch.batch[key] = value
                        valid_positions = batch.batch["response_mask"].bool().to(
                            support["kalign_similarity_mask"].device
                        )
                        metrics["kalign/union_support_size"] = (
                            support["kalign_similarity_mask"].sum(dim=-1)[valid_positions].float().mean().item()
                            if valid_positions.any() else 0.0
                        )

                    if self.reward_fn is not None:
                        reward_tensor, _ = compute_reward(batch, self.reward_fn)
                    else:
                        reward_tensor = torch.zeros_like(batch.batch["response_mask"])
                    batch.batch["token_level_scores"] = reward_tensor
                    batch.batch["token_level_rewards"] = reward_tensor
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        config=self.config.algorithm,
                    )
                    if self.config.algorithm.adv_estimator == "KALIGN_OPD":
                        valid_rollouts = batch.batch["kalign_valid_rollout"]
                        metrics["kalign/valid_rollout_ratio"] = valid_rollouts.float().mean().item()
                        metrics["kalign/positive_positions_per_rollout"] = (
                            batch.batch["kalign_positive_position_count"].float().mean().item()
                        )
                        batch.batch.pop("kalign_positive_position_count")
                        batch.batch.pop("kalign_valid_rollout")
                        position_metric_keys = {
                            "position_mask_eligible_count": "mask/eligible_tokens_per_rollout",
                            "position_mask_selected_count": "mask/masked_tokens_per_rollout",
                            "position_mask_selected_m_mean": "mask/selected_m_mean",
                            "position_mask_selected_c_mean": "mask/selected_c_mean",
                            "position_mask_selected_positive_ratio": "mask/selected_positive_c_ratio",
                            "position_mask_selected_zero_ratio": "mask/selected_zero_c_ratio",
                            "position_mask_high_m_high_c_overlap": "mask/high_m_high_c_overlap",
                            "position_mask_magnitude_mean": "mask/all_m_mean",
                            "position_mask_alignment_mean": "mask/all_c_mean",
                        }
                        for key, metric_name in position_metric_keys.items():
                            if key in batch.batch:
                                metrics[metric_name] = batch.batch[key].float().mean().item()
                                batch.batch.pop(key)
                        if "mask/masked_tokens_per_rollout" in metrics:
                            eligible_mean = max(metrics["mask/eligible_tokens_per_rollout"], 1e-12)
                            metrics["mask/fraction"] = metrics["mask/masked_tokens_per_rollout"] / eligible_mean
                        if batch.meta_info["kalign_union_topk"]:
                            for key in (
                                "kalign_similarity_ids",
                                "kalign_similarity_mask",
                                "kalign_similarity_student_log_probs",
                                "kalign_similarity_answer_log_probs",
                                "kalign_similarity_knowledge_log_probs",
                            ):
                                batch.batch.pop(key)
                    for key in (
                        "teacher_top_k_ids",
                        "teacher_top_k_log_probs",
                        "teacher_entropy",
                        "teacher_valid_counts",
                        "overlap_mask",
                        "teacher_in_student_mask",
                        "student_log_probs_on_teacher_ids",
                        "teacher_sampled_log_probs",
                        "answer_teacher_sampled_log_probs",
                        "answer_teacher_on_union_log_probs",
                    ):
                        if key in batch.batch:
                            batch.batch.pop(key)
                    if batch.meta_info["kalign_union_topk"]:
                        if any(key in batch.batch for key in ("union_top_k_ids", "union_top_k_log_probs")):
                            raise RuntimeError("KAlign v2 union candidates must not enter the actor loss")
                        if batch.batch["student_top_k_ids"].shape[-1] != top_k or batch.batch[
                            "teacher_on_student_log_probs"
                        ].shape[-1] != top_k:
                            raise RuntimeError("KAlign v2 reverse-KL must stay on student Top-K")
                    with marked_timer("update_actor", timing_raw, color="red"):
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                logger.log(data=metrics, step=self.global_steps)
                is_last_step = self.global_steps >= self.total_training_steps
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    val_metrics = self._validate()
                    if val_metrics:
                        logger.log(data=val_metrics, step=self.global_steps)
                if (
                    self.config.trainer.save_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0)
                ):
                    self._save_checkpoint()
                progress.update(1)
                if is_last_step:
                    progress.close()
                    return
                self.global_steps += 1
        progress.close()
