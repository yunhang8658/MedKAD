# Copyright 2024 PRIME team and/or its affiliates
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

"""Medical VQA 答案打分函数（使用 <answer> 标签格式）。

约定的模型输出格式（由 prompt 强制要求）:
    任意可选的分析文本
    <answer>X</answer>

格式奖励只检查完整且非空的 ``<answer>...</answer>``。``<think>``
标签不再是格式奖励的要求；如果启用了独立的 think loss，它使用
答案标签之前的有效分析长度作为另一项信号。

兼容 2/3/4/5/6 选项；选项数由 extra_info 提供，按以下优先级解析：
    extra_info["valid_letters"]   ->  "AB" / "ABC" / "ABCD" ...
    extra_info["num_options"]     ->  整数 2/3/4/...
    extra_info["options"]         ->  选项列表（取长度）
    extra_info["question"]        ->  从题面文本扫描 "A." / "B)" / "C:" 推断
    fallback                       ->  "ABCD"
"""

import re
import string
import traceback

# 最多支持 A-F（OmniMedVQA collect_options 也只到 F）
ALL_LETTERS = string.ascii_uppercase[:6]  # "ABCDEF"


# =========================================================
#  选项集合解析（未改动）
# =========================================================

def _detect_letters_from_question(text: str) -> str:
    """从题面扫描 'A:xxx' / 'B. xxx' / 'C) xxx' 等选项标号。"""
    if not text:
        return ""
    found = set()
    for L in ALL_LETTERS:
        pat = rf"(?:^|[\s\(\[])({L})\s*[\.\)\:、:]"
        if re.search(pat, text, flags=re.MULTILINE):
            found.add(L)
    return "".join(L for L in ALL_LETTERS if L in found)


def _resolve_valid_letters(extra_info, gt_answer: str) -> str:
    """确定本题合法的字母集合。保证 gt_answer 一定在结果里。"""
    valid = ""

    if isinstance(extra_info, dict):
        v = extra_info.get("valid_letters")
        if v:
            valid = "".join(ch for ch in str(v).upper() if ch in ALL_LETTERS)

        if not valid:
            n = extra_info.get("num_options") or extra_info.get("num_choices")
            if n is not None:
                try:
                    n = int(n)
                    if 2 <= n <= len(ALL_LETTERS):
                        valid = ALL_LETTERS[:n]
                except (ValueError, TypeError):
                    pass

        if not valid and extra_info.get("options") is not None:
            try:
                n = len(extra_info["options"])
                if 2 <= n <= len(ALL_LETTERS):
                    valid = ALL_LETTERS[:n]
            except TypeError:
                pass

        if not valid:
            q_text = (
                extra_info.get("question")
                or extra_info.get("prompt")
                or extra_info.get("query")
                or ""
            )
            detected = _detect_letters_from_question(q_text)
            if len(detected) >= 2:
                valid = detected

    if not valid:
        valid = "ABCD"

    # 强制把 gt 字母放进合法集合（防数据预处理漏字段）
    gt_letter = (gt_answer or "").strip().upper()
    if gt_letter and gt_letter in ALL_LETTERS and gt_letter not in valid:
        valid = "".join(L for L in ALL_LETTERS if L in set(valid) | {gt_letter})

    return valid


# =========================================================
#  标签检测（新增）
# =========================================================

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)


def has_answer_tag(passage: str) -> bool:
    match = _ANSWER_TAG_RE.search(passage) if passage else None
    return bool(match and match.group(1).strip())


def has_think_tag(passage: str) -> bool:
    match = _THINK_TAG_RE.search(passage) if passage else None
    return bool(match and match.group(1).strip())


# =========================================================
#  答案抽取
# =========================================================

def _pick_letter_from_text(text: str, valid_letters: str) -> str | None:
    """在一段文本里找合法字母：优先独立字母，其次任意字符。"""
    if not text:
        return None
    text_upper = text.strip().upper()
    # 1) 独立字母（如 "B"、"B."、"B)"、"B is correct"）
    m = re.search(rf"\b([{valid_letters}])\b", text_upper)
    if m:
        return m.group(1)
    # 2) 兜底：任意位置的合法字母（如 "BA" -> "B"，"选B" -> "B"）
    for ch in text_upper:
        if ch in valid_letters:
            return ch
    return None


def extract_answer(passage: str, valid_letters: str = "ABCD") -> str | None:
    """
    从模型输出中提取答案字母；仅接受 valid_letters 中的字母。
    优先级:
      1. <answer>...</answer>   ← 最高（符合 prompt 约定）
      2. \\boxed{}
      3. 常见模板（"the answer is X" 等）
      4. 尾部独立字母
    """
    if passage is None:
        return None

    letter_class = f"[{valid_letters}{valid_letters.lower()}]"

    # ---- 1. <answer>...</answer>（取最后一个标签，避免误抽 <think> 中的演练答案） ----
    answer_matches = _ANSWER_TAG_RE.findall(passage)
    for content in reversed(answer_matches):
        cand = _pick_letter_from_text(content, valid_letters)
        if cand is not None:
            return cand
    # 如果 <answer> 存在但内容里没合法字母，继续走兜底（不直接返回 None，留点训练信号）

    # ---- 2. \boxed{...} ----
    if "\\boxed" in passage:
        boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", passage)
        for content in reversed(boxed_matches):
            cand = _pick_letter_from_text(content, valid_letters)
            if cand is not None:
                return cand

    # ---- 3. 常见模板 ----
    patterns = [
        rf"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:\s]*\(?({letter_class})\)?",
        rf"[Aa]nswer\s*[:\s]+\(?({letter_class})\)?",
        rf"[Oo]ption\s+({letter_class})\s+is\s+correct",
        rf"[Cc]hoose\s+({letter_class})",
        rf"(?:选|答案)[是为：:]\s*({letter_class})",
        rf"\b({letter_class})\s*[\.\)]\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, passage)
        if m:
            cand = m.group(1).upper()
            if cand in valid_letters:
                return cand

    # ---- 4. 兜底：尾部 50 字符 ----
    tail = passage[-50:] if len(passage) > 50 else passage
    tail_matches = re.findall(rf"\b({letter_class})\b", tail)
    for c in reversed(tail_matches):
        c = c.upper()
        if c in valid_letters:
            return c

    return None


# =========================================================
#  打分
# =========================================================

def _compute_format_score(has_ans: bool, has_thk: bool, is_correct: bool) -> float:
    """Return format reward for a complete, non-empty ``<answer>`` pair.

    Correctness is deliberately not part of this signal: answer accuracy is
    reported separately, while the format reward should measure only whether
    the answer delimiter protocol was followed.  ``has_thk`` is retained in
    the signature for compatibility and diagnostics, but is intentionally not
    consulted here.
    """
    return 1.0 if has_ans else 0.0


def compute_score(model_response: str, gt_answer: str, valid_letters: str = "ABCD"):
    """
    Returns:
        dict: score / format_score / acc / extracted_gt / pred / valid_letters
              + has_answer_tag / has_think_tag (用于格式达成率统计)
    """
    has_ans = has_answer_tag(model_response)
    has_thk = has_think_tag(model_response)
    model_answer = extract_answer(model_response, valid_letters=valid_letters)
    gt = (gt_answer or "").strip().upper()
    is_correct = model_answer is not None and model_answer == gt
    format_score = _compute_format_score(has_ans, has_thk, is_correct)

    if model_answer is None:
        return {
            "score": 0.0,
            "format_score": format_score,
            "acc": False,
            "extracted_gt": gt,
            "pred": "",
            "valid_letters": valid_letters,
            "has_answer_tag": has_ans,
            "has_think_tag": has_thk,
        }

    return {
        "score": 1.0 if is_correct else 0.0,
        "format_score": format_score,
        "acc": is_correct,
        "extracted_gt": gt,
        "pred": model_answer,
        "valid_letters": valid_letters,
        "has_answer_tag": has_ans,
        "has_think_tag": has_thk,
    }


# =========================================================
#  reward function 入口
# =========================================================

def reward_func(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
):
    """奖励函数入口。"""
    try:
        valid_letters = _resolve_valid_letters(extra_info, str(ground_truth))
        res = compute_score(solution_str, str(ground_truth), valid_letters=valid_letters)

        if isinstance(res, dict):
            return res
        if isinstance(res, (int, float, bool)):
            return float(res)
        return float(res[0])
    except Exception as e:
        print(f"[ERROR] Error in reward_func for Medical VQA task: {str(e)}")
        traceback.print_exc()
        raise
