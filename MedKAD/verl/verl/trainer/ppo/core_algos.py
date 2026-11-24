# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
import hashlib
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | AlgoConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    STANDARD_OPD = "STANDARD_OPD"
    KALIGN_OPD = "KALIGN_OPD"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est("STANDARD_OPD")
def compute_standard_opd_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Uniform per-question, per-rollout, per-response-token OPD loss weight.

    The historical launcher used a shared estimator with visual weighting
    disabled. This isolated implementation retains only that uniform branch:
    every sibling rollout has weight 1/K and every valid token has weight 1/T.
    The actor consumes va_loss_weight as its reverse-KL loss mask.
    """
    del token_level_rewards, config, kwargs
    mask = response_mask.float()
    counts = mask.sum(dim=-1).clamp(min=1.0)
    groups = defaultdict(list)
    for row, prompt_id in enumerate(index):
        groups[prompt_id].append(row)
    rollout_weight = torch.zeros(mask.shape[0], device=mask.device, dtype=mask.dtype)
    for rows in groups.values():
        rollout_weight[rows] = 1.0 / len(rows)
    loss_weight = rollout_weight[:, None] * mask / counts[:, None]
    return loss_weight, loss_weight.clone(), {"va_loss_weight": loss_weight}


@torch.no_grad()
def compute_kalign_token_weights(
    student_log_probs: torch.Tensor,
    answer_teacher_log_probs: torch.Tensor,
    knowledge_teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    delta: float = 1e-6,
    candidate_mask: Optional[torch.Tensor] = None,
    alignment_fraction: float = 0.5,
    score_mode: str = "cosine",
    answer_teacher_sampled_log_probs: Optional[torch.Tensor] = None,
    knowledge_teacher_sampled_log_probs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Blend uniform OPD weights with a nonnegative token score, per rollout.

    Every valid response token receives at least (1-alignment_fraction)/T.
    ``cosine`` is MedKAD. ``sampled_logprob_diff`` replaces its vector
    similarity with relu(log P_K(y_t) - log P_A(y_t)) for the actual student
    token. A rollout with no positive score receives uniform OPD weights.
    """
    if not 0 <= alignment_fraction < 1:
        raise ValueError("KAlign alignment_fraction must satisfy 0 <= lambda < 1")
    student = student_log_probs.float()
    answer = answer_teacher_log_probs.float()
    knowledge = knowledge_teacher_log_probs.float()
    if student.shape != answer.shape or student.shape != knowledge.shape or student.ndim != 3:
        raise ValueError("KAlign needs three aligned [rollout, position, top_k] arrays")
    if response_mask.shape != student.shape[:2]:
        raise ValueError("KAlign response mask shape differs from log probabilities")
    if candidate_mask is not None and candidate_mask.shape != student.shape:
        raise ValueError("KAlign candidate mask shape differs from log probabilities")
    if not (torch.isfinite(student).all() and torch.isfinite(answer).all() and torch.isfinite(knowledge).all()):
        raise ValueError("KAlign received non-finite log probabilities")
    if score_mode == "cosine":
        knowledge_direction = knowledge - answer
        learning_direction = knowledge - student
        if candidate_mask is not None:
            candidate_mask = candidate_mask.to(device=student.device, dtype=torch.bool)
            knowledge_direction = knowledge_direction.masked_fill(~candidate_mask, 0.0)
            learning_direction = learning_direction.masked_fill(~candidate_mask, 0.0)
        raw_inner = (knowledge_direction * learning_direction).sum(dim=-1)
        knowledge_norm = torch.linalg.vector_norm(knowledge_direction, dim=-1)
        learning_norm = torch.linalg.vector_norm(learning_direction, dim=-1)
        eligible = response_mask.bool() & (raw_inner > delta) & (knowledge_norm > 0) & (learning_norm > 0)
        cosine = (raw_inner / (knowledge_norm * learning_norm).clamp_min(1e-30)).clamp(0.0, 1.0)
        positive_score = torch.where(eligible, cosine, torch.zeros_like(cosine))
    elif score_mode == "sampled_logprob_diff":
        if answer_teacher_sampled_log_probs is None or knowledge_teacher_sampled_log_probs is None:
            raise ValueError("sampled_logprob_diff requires answer and knowledge sampled-token log probabilities")
        answer_sampled = answer_teacher_sampled_log_probs.float()
        knowledge_sampled = knowledge_teacher_sampled_log_probs.float()
        if answer_sampled.shape != response_mask.shape or knowledge_sampled.shape != response_mask.shape:
            raise ValueError("sampled-token log probabilities must match the response mask")
        if not (torch.isfinite(answer_sampled).all() and torch.isfinite(knowledge_sampled).all()):
            raise ValueError("sampled_logprob_diff received non-finite log probabilities")
        positive_score = (knowledge_sampled - answer_sampled).clamp_min(0.0)
        positive_score = positive_score * response_mask.to(positive_score.dtype)
        eligible = response_mask.bool() & (positive_score > 0)
    else:
        raise ValueError(f"Unsupported KAlign score mode: {score_mode!r}")
    normalizer = positive_score.sum(dim=-1, keepdim=True)
    token_count = response_mask.bool().sum(dim=-1, keepdim=True)
    uniform = response_mask.to(positive_score.dtype) / token_count.clamp_min(1)
    aligned = torch.where(normalizer > 0, positive_score / normalizer.clamp_min(1e-30), uniform)
    weights = (1 - alignment_fraction) * uniform + alignment_fraction * aligned
    valid_rollout = token_count.squeeze(-1) > 0
    return weights, {
        "kalign_valid_rollout": valid_rollout,
        "kalign_positive_position_count": eligible.sum(dim=-1),
    }


@torch.no_grad()
def compute_position_mask_token_weights(
    student_log_probs: torch.Tensor,
    answer_teacher_log_probs: torch.Tensor,
    knowledge_teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    mask_eligible: torch.Tensor,
    *,
    mode: str,
    sample_keys: np.ndarray,
    global_step: int,
    seed: int,
    mask_fraction: float = 0.1,
    delta: float = 1e-6,
    candidate_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Uniform OPD after masking positions ranked by knowledge diagnostics.

    The union support is used only for ``m_t`` and ``c_t``. Returned weights
    are response-normalized and are later applied to reverse KL on the
    student's Top-K support. EOS/special positions remain in the KL loss but
    must be false in ``mask_eligible`` so they cannot be selected for masking.
    """
    valid_modes = {"none", "random", "high_m", "low_m", "high_c", "low_c"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported POSITION_MASK_MODE={mode!r}")
    if not 0.0 <= mask_fraction < 1.0:
        raise ValueError("POSITION_MASK_FRACTION must satisfy 0 <= fraction < 1")
    student = student_log_probs.float()
    answer = answer_teacher_log_probs.float()
    knowledge = knowledge_teacher_log_probs.float()
    if student.shape != answer.shape or student.shape != knowledge.shape or student.ndim != 3:
        raise ValueError("Position masking needs three aligned [rollout, position, candidate] arrays")
    if response_mask.shape != student.shape[:2] or mask_eligible.shape != response_mask.shape:
        raise ValueError("Position masking received incompatible response masks")
    if candidate_mask is not None and candidate_mask.shape != student.shape:
        raise ValueError("Position masking candidate mask shape differs from log probabilities")
    if len(sample_keys) != student.shape[0]:
        raise ValueError("Position masking needs one stable sample key per rollout")
    if not (torch.isfinite(student).all() and torch.isfinite(answer).all() and torch.isfinite(knowledge).all()):
        raise ValueError("Position masking received non-finite log probabilities")

    knowledge_direction = knowledge - answer
    learning_direction = knowledge - student
    if candidate_mask is not None:
        candidate_mask = candidate_mask.to(device=student.device, dtype=torch.bool)
        knowledge_direction = knowledge_direction.masked_fill(~candidate_mask, 0.0)
        learning_direction = learning_direction.masked_fill(~candidate_mask, 0.0)
    inner = (knowledge_direction * learning_direction).sum(dim=-1)
    magnitude = torch.linalg.vector_norm(knowledge_direction, dim=-1)
    learning_norm = torch.linalg.vector_norm(learning_direction, dim=-1)
    direction_valid = (inner > delta) & (magnitude > 0) & (learning_norm > 0)
    cosine = (inner / (magnitude * learning_norm).clamp_min(1e-30)).clamp(0.0, 1.0)
    alignment = torch.where(direction_valid, cosine, torch.zeros_like(cosine))

    valid = response_mask.bool()
    eligible = mask_eligible.bool() & valid
    selected = torch.zeros_like(valid)
    batch_size = student.shape[0]
    selected_m_mean = torch.zeros(batch_size, dtype=torch.float32, device=student.device)
    selected_c_mean = torch.zeros_like(selected_m_mean)
    selected_positive_ratio = torch.zeros_like(selected_m_mean)
    selected_zero_ratio = torch.zeros_like(selected_m_mean)
    high_overlap = torch.zeros_like(selected_m_mean)

    # Rollout ordinals are stable within each question group even after the
    # central trainer reorders complete groups for balanced data parallelism.
    seen: dict[str, int] = defaultdict(int)
    rollout_ordinals: list[int] = []
    for key in sample_keys:
        key_s = str(key)
        rollout_ordinals.append(seen[key_s])
        seen[key_s] += 1

    magnitude_cpu = magnitude.detach().cpu().numpy()
    alignment_cpu = alignment.detach().cpu().numpy()
    eligible_cpu = eligible.detach().cpu().numpy()
    masked_counts = torch.zeros(batch_size, dtype=torch.long, device=student.device)
    eligible_counts = eligible.sum(dim=-1).to(torch.long)

    for row in range(batch_size):
        positions = np.flatnonzero(eligible_cpu[row])
        k = int(np.floor(mask_fraction * len(positions)))
        if k <= 0:
            continue
        material = f"{seed}|{global_step}|{sample_keys[row]}|{rollout_ordinals[row]}".encode("utf-8")
        row_seed = int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "little")
        tie_rng = np.random.default_rng(row_seed)
        tie = tie_rng.random(len(positions))
        m_values = magnitude_cpu[row, positions]
        c_values = alignment_cpu[row, positions]
        high_m_order = np.lexsort((tie, -m_values))
        high_c_order = np.lexsort((tie, -c_values))
        high_m = positions[high_m_order[:k]]
        high_c = positions[high_c_order[:k]]
        high_overlap[row] = len(set(high_m.tolist()) & set(high_c.tolist())) / float(k)

        if mode == "none":
            continue
        if mode == "random":
            chosen = positions[np.argsort(tie, kind="stable")[:k]]
        elif mode == "high_m":
            chosen = high_m
        elif mode == "low_m":
            chosen = positions[np.lexsort((tie, m_values))[:k]]
        elif mode == "high_c":
            chosen = high_c
        else:  # low_c
            chosen = positions[np.lexsort((tie, c_values))[:k]]
        chosen_t = torch.as_tensor(chosen, dtype=torch.long, device=student.device)
        selected[row, chosen_t] = True
        masked_counts[row] = k
        selected_m = magnitude[row, chosen_t]
        selected_c = alignment[row, chosen_t]
        selected_m_mean[row] = selected_m.mean()
        selected_c_mean[row] = selected_c.mean()
        selected_positive_ratio[row] = (selected_c > 0).float().mean()
        selected_zero_ratio[row] = (selected_c == 0).float().mean()

    remaining = valid & ~selected
    remaining_count = remaining.sum(dim=-1, keepdim=True)
    if (remaining_count == 0).any():
        raise RuntimeError("Position masking removed every KL position from a response")
    weights = remaining.to(torch.float32) / remaining_count.to(torch.float32)
    return weights, {
        "kalign_valid_rollout": valid.sum(dim=-1) > 0,
        "kalign_positive_position_count": (direction_valid & valid).sum(dim=-1),
        "position_mask_eligible_count": eligible_counts,
        "position_mask_selected_count": masked_counts,
        "position_mask_selected_m_mean": selected_m_mean,
        "position_mask_selected_c_mean": selected_c_mean,
        "position_mask_selected_positive_ratio": selected_positive_ratio,
        "position_mask_selected_zero_ratio": selected_zero_ratio,
        "position_mask_high_m_high_c_overlap": high_overlap,
        "position_mask_magnitude_mean": (magnitude * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1),
        "position_mask_alignment_mean": (alignment * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1),
    }


@register_adv_est("KALIGN_OPD")
def compute_kalign_opd_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Optional[AlgoConfig] = None,
    *,
    student_top_k_log_probs: torch.Tensor,
    answer_teacher_on_student_log_probs: torch.Tensor,
    teacher_on_student_log_probs: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
    score_mode: str = "cosine",
    answer_teacher_sampled_log_probs: Optional[torch.Tensor] = None,
    knowledge_teacher_sampled_log_probs: Optional[torch.Tensor] = None,
    position_mask_mode: Optional[str] = None,
    position_mask_eligible: Optional[torch.Tensor] = None,
    position_mask_sample_keys: Optional[np.ndarray] = None,
    position_mask_global_step: int = 0,
    position_mask_seed: int = 42,
    position_mask_fraction: float = 0.1,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    del token_level_rewards, index, config, kwargs
    if position_mask_mode is not None:
        if position_mask_eligible is None or position_mask_sample_keys is None:
            raise ValueError("Position masking requires eligibility and stable sample keys")
        weights, diagnostics = compute_position_mask_token_weights(
            student_top_k_log_probs,
            answer_teacher_on_student_log_probs,
            teacher_on_student_log_probs,
            response_mask,
            position_mask_eligible,
            mode=position_mask_mode,
            sample_keys=position_mask_sample_keys,
            global_step=position_mask_global_step,
            seed=position_mask_seed,
            mask_fraction=position_mask_fraction,
            candidate_mask=candidate_mask,
        )
    else:
        weights, diagnostics = compute_kalign_token_weights(
            student_top_k_log_probs,
            answer_teacher_on_student_log_probs,
            teacher_on_student_log_probs,
            response_mask,
            candidate_mask=candidate_mask,
            score_mode=score_mode,
            answer_teacher_sampled_log_probs=answer_teacher_sampled_log_probs,
            knowledge_teacher_sampled_log_probs=knowledge_teacher_sampled_log_probs,
        )
    return weights, weights.clone(), {"va_loss_weight": weights, **diagnostics}


@torch.no_grad()
def build_kalign_similarity_union(
    student_ids: torch.Tensor,
    knowledge_teacher_ids: torch.Tensor,
    student_on_student_ids: torch.Tensor,
    student_on_knowledge_ids: torch.Tensor,
    answer_on_union_ids: torch.Tensor,
    knowledge_on_student_ids: torch.Tensor,
    knowledge_on_knowledge_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Align S/A/K for similarity on the deduplicated S-top16 ∪ K-top16 set.

    Fixed-width tensors retain 32 slots for batching. The mask removes repeated
    knowledge-teacher candidates from the direction and cosine calculations.
    """
    if student_ids.shape != knowledge_teacher_ids.shape or student_ids.ndim != 3:
        raise ValueError("KAlign v2 needs matching [rollout, position, 16] ID tensors")
    shape = student_ids.shape
    if any(value.shape != shape for value in (
        student_on_student_ids, student_on_knowledge_ids,
        knowledge_on_student_ids, knowledge_on_knowledge_ids,
    )) or answer_on_union_ids.shape != (*shape[:2], 2 * shape[2]):
        raise ValueError("KAlign v2 candidate log-probability shapes are not aligned")
    keep_knowledge = (knowledge_teacher_ids.unsqueeze(-1) != student_ids.unsqueeze(-2)).all(dim=-1)
    mask = torch.cat((torch.ones_like(student_ids, dtype=torch.bool), keep_knowledge), dim=-1)
    return {
        "kalign_similarity_ids": torch.cat((student_ids, knowledge_teacher_ids), dim=-1),
        "kalign_similarity_mask": mask,
        "kalign_similarity_student_log_probs": torch.cat((student_on_student_ids, student_on_knowledge_ids), dim=-1),
        "kalign_similarity_answer_log_probs": answer_on_union_ids,
        "kalign_similarity_knowledge_log_probs": torch.cat(
            (knowledge_on_student_ids, knowledge_on_knowledge_ids), dim=-1
        ),
    }


def compute_kalign_rollout_coefficients(
    uids: np.ndarray,
    token_weights: torch.Tensor,
    global_valid_questions: int,
    world_size: int = 1,
) -> torch.Tensor:
    """Per-rollout factors for an exact question/valid-rollout mean.

    The world-size factor cancels distributed gradient averaging. Callers sum
    weighted token losses across microbatches without another normalization.
    """
    if token_weights.ndim != 2 or len(uids) != token_weights.shape[0]:
        raise ValueError("KAlign expects one token-weight row per rollout UID")
    if global_valid_questions < 0 or world_size < 1:
        raise ValueError("Invalid KAlign distributed reduction dimensions")
    result = torch.zeros(token_weights.shape[0], dtype=token_weights.dtype, device=token_weights.device)
    if global_valid_questions == 0:
        return result
    groups: dict[str, list[int]] = {}
    for row, uid in enumerate(uids):
        groups.setdefault(str(uid), []).append(row)
    valid = token_weights.sum(dim=-1) > 0
    for rows in groups.values():
        active = [row for row in rows if bool(valid[row])]
        if active:
            result[active] = world_size / (global_valid_questions * len(active))
    return result

def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        loss = verl_F.masked_mean(seq_losses, seq_mask)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        loss = verl_F.masked_mean(seq_losses, seq_mask)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training when using this aggregation mode.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@register_policy_loss("vanilla")
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[DictConfig | AlgoConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    format_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Generic clipped policy loss retained by the underlying training runtime."""
    if config is None or isinstance(config, AlgoConfig):
        raise TypeError("vanilla policy loss requires the actor configuration")

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    if clip_ratio_c <= 1.0:
        raise ValueError(f"clip_ratio_c must be greater than 1.0, got {clip_ratio_c}")

    negative_approx_kl = torch.clamp(log_prob - old_log_prob, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - clip_ratio_low, 1 + clip_ratio_high
    )
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(
        torch.gt(pg_losses2, pg_losses1).float(), response_mask
    )

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(),
        response_mask,
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    loss_mask = response_mask
    if format_mask is not None:
        loss_mask = loss_mask * format_mask.unsqueeze(-1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)

    metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, metrics


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


