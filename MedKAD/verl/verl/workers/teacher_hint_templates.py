"""Teacher-only prompt suffixes for Standard OPD distillation."""

GT_CONTRASTIVE_HINT_TEMPLATE = (
    "\n\n["
    "Use the privileged answer ({gt}: {gt_answer_text}) only as a correctness reference. "
    "Analyze the image and question, identify the visual evidence or attributes relevant to this task, and "
    "explain why the referenced option is supported and the other options are less appropriate. "
    "Write several substantive reasoning sentences as ordinary text, then put ONLY the final answer {gt} "
    "inside <answer>...</answer>. The required ending is <answer>{gt}</answer>.]\n"
)

KNOWLEDGE_GT_CONTRASTIVE_HINT_TEMPLATE = (
    "\n\n[The following relevant medical knowledge is provided as reference:\n"
    "{knowledge_hint}\n\n"
    "Use the privileged answer ({gt}: {gt_answer_text}) only as a correctness reference. "
    "Analyze the image and question with the relevant medical knowledge provided above, identify the visual "
    "evidence or attributes relevant to this task, and explain why the referenced option is supported and "
    "the other options are less appropriate. Write several substantive reasoning sentences as ordinary "
    "text, then put ONLY the final answer {gt} inside <answer>...</answer>. The required ending is "
    "<answer>{gt}</answer>.]\n"
)


def format_teacher_contrastive_hint(
    gt: str,
    gt_answer_text: str,
    knowledge_hint: str | None = None,
) -> str:
    """Format the teacher-only prompt suffix documented in README_KNOWLEDGE_HINT.md."""
    gt = str(gt).strip()
    gt_answer_text = str(gt_answer_text).strip()
    if not gt:
        raise ValueError("Teacher contrastive hint requires a non-empty ground-truth answer")
    if not gt_answer_text:
        raise ValueError("Teacher contrastive hint requires extra_info.gt_answer_text")
    if isinstance(knowledge_hint, str) and knowledge_hint.strip():
        return KNOWLEDGE_GT_CONTRASTIVE_HINT_TEMPLATE.format(
            knowledge_hint=knowledge_hint.strip(),
            gt=gt,
            gt_answer_text=gt_answer_text,
        )
    return GT_CONTRASTIVE_HINT_TEMPLATE.format(gt=gt, gt_answer_text=gt_answer_text)
