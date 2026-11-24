"""CPU checks for the 50/50 uniform-plus-alignment KAlign weights."""

import numpy as np
import torch

from verl.trainer.ppo.core_algos import (
    build_kalign_similarity_union,
    compute_kalign_rollout_coefficients,
    compute_kalign_token_weights,
)


def test_soft_weights_keep_all_valid_tokens_and_fallback_to_uniform():
    student = torch.tensor([[[-2., -3.], [-1., -3.], [-1., -3.]],
                            [[-1., -3.], [-1., -3.], [-1., -3.]]])
    answer = torch.tensor([[[-2., -3.], [-3., -1.], [-3., -1.]],
                           [[-3., -1.], [-3., -1.], [-3., -1.]]])
    knowledge = torch.tensor([[[-1., -3.], [-2., -2.], [-2., -2.]],
                              [[-2., -2.], [-2., -2.], [-2., -2.]]])
    mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
    weights, diagnostics = compute_kalign_token_weights(student, answer, knowledge, mask)
    torch.testing.assert_close(weights[0], torch.tensor([.75, .25, 0.]))
    torch.testing.assert_close(weights[1], torch.full((3,), 1 / 3))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
    assert diagnostics["kalign_valid_rollout"].tolist() == [True, True]
    assert diagnostics["kalign_positive_position_count"].tolist() == [1, 0]


def test_fraction_endpoints_and_empty_response():
    s = torch.tensor([[[-2., -3.], [-1., -3.]], [[-1., -3.], [-1., -3.]]])
    a = torch.tensor([[[-2., -3.], [-3., -1.]], [[-3., -1.], [-3., -1.]]])
    k = torch.tensor([[[-1., -3.], [-2., -2.]], [[-2., -2.], [-2., -2.]]])
    mask = torch.tensor([[1., 1.], [0., 0.]])
    weights, diagnostics = compute_kalign_token_weights(s, a, k, mask, alignment_fraction=0)
    torch.testing.assert_close(weights, torch.tensor([[.5, .5], [0., 0.]]))
    assert diagnostics["kalign_valid_rollout"].tolist() == [True, False]
    for bad in (-0.1, 1.0):
        try:
            compute_kalign_token_weights(s, a, k, mask, alignment_fraction=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid alignment fraction was accepted")


def test_sampled_logprob_difference_replaces_cosine_and_falls_back_to_uniform():
    student = torch.zeros(2, 3, 2)
    answer = torch.zeros(2, 3, 2)
    knowledge = torch.zeros(2, 3, 2)
    mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
    answer_sampled = torch.tensor([[-2., -2., -2.], [-1., -2., -3.]])
    knowledge_sampled = torch.tensor([[-1., -3., -1.], [-2., -3., -4.]])
    weights, diagnostics = compute_kalign_token_weights(
        student,
        answer,
        knowledge,
        mask,
        score_mode="sampled_logprob_diff",
        answer_teacher_sampled_log_probs=answer_sampled,
        knowledge_teacher_sampled_log_probs=knowledge_sampled,
    )
    torch.testing.assert_close(weights[0], torch.tensor([.75, .25, 0.]))
    torch.testing.assert_close(weights[1], torch.full((3,), 1 / 3))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
    assert diagnostics["kalign_positive_position_count"].tolist() == [1, 0]


def test_question_mean_and_microbatch_split_with_uniform_fallback():
    uids = np.array(["q1", "q1", "q2", "q2"], dtype=object)
    weights = torch.tensor([[.75, .25], [.5, .5], [.5, .5], [.6, .4]])
    coeff = compute_kalign_rollout_coefficients(uids, weights, global_valid_questions=2)
    torch.testing.assert_close(coeff, torch.full((4,), .25))
    values = torch.arange(8, dtype=torch.float32).reshape(4, 2).requires_grad_()
    whole = (weights * coeff[:, None] * values).sum()
    split = sum((weights[part] * coeff[part, None] * values[part]).sum()
                for part in ([0, 2], [1, 3]))
    torch.testing.assert_close(whole, split)
    torch.testing.assert_close(torch.autograd.grad(whole, values, retain_graph=True)[0],
                               torch.autograd.grad(split, values)[0])


def test_union_mask_changes_alignment_only():
    student_ids = torch.tensor([[[1, 2]]])
    teacher_ids = torch.tensor([[[2, 3]]])
    support = build_kalign_similarity_union(
        student_ids, teacher_ids,
        student_on_student_ids=torch.tensor([[[-2., -3.]]]),
        student_on_knowledge_ids=torch.tensor([[[-3., -4.]]]),
        answer_on_union_ids=torch.tensor([[[-2., -3., -3., -4.]]]),
        knowledge_on_student_ids=torch.tensor([[[-1., -3.]]]),
        knowledge_on_knowledge_ids=torch.tensor([[[-3., -4.]]]),
    )
    assert support["kalign_similarity_mask"].tolist() == [[[True, True, False, True]]]
    weights, _ = compute_kalign_token_weights(
        support["kalign_similarity_student_log_probs"],
        support["kalign_similarity_answer_log_probs"],
        support["kalign_similarity_knowledge_log_probs"],
        torch.ones(1, 1),
        candidate_mask=support["kalign_similarity_mask"],
    )
    torch.testing.assert_close(weights, torch.ones(1, 1))


def test_trainer_advantage_keeps_student_loss_support_for_both_modes():
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import compute_advantage

    student = torch.tensor([[[-2., -3.], [-1., -3.]]])
    answer = torch.tensor([[[-2., -3.], [-3., -1.]]])
    knowledge = torch.tensor([[[-1., -3.], [-2., -2.]]])
    for union_mode in (False, True):
        tensors = {
            "response_mask": torch.ones(1, 2),
            "token_level_rewards": torch.zeros(1, 2),
            "student_top_k_log_probs": student,
            "answer_teacher_on_student_log_probs": answer,
            "teacher_on_student_log_probs": knowledge,
        }
        if union_mode:
            tensors.update({
                "kalign_similarity_student_log_probs": torch.cat((student, student), dim=-1),
                "kalign_similarity_answer_log_probs": torch.cat((answer, answer), dim=-1),
                "kalign_similarity_knowledge_log_probs": torch.cat((knowledge, knowledge), dim=-1),
                "kalign_similarity_mask": torch.tensor([[[True, True, False, False]] * 2]),
            })
        data = DataProto.from_dict(
            tensors=tensors,
            non_tensors={"uid": np.array(["q1"], dtype=object)},
            meta_info={"kalign_union_topk": union_mode},
        )
        result = compute_advantage(data, adv_estimator="KALIGN_OPD")
        torch.testing.assert_close(result.batch["va_loss_weight"], torch.tensor([[.75, .25]]))
        assert result.batch["teacher_on_student_log_probs"].shape[-1] == 2
        assert "union_top_k_ids" not in result.batch
