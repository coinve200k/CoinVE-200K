#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_coinbench_gemini_public.py
============================================================
Evaluate CoinBench checklist with the public Google Gemini API.

Key design:
    - Uses the `google-genai` SDK to call the public Gemini API.
    - Videos are uploaded via `client.files.upload()` and referenced as File objects.
    - Authentication is via the `GEMINI_API_KEY` environment variable.
    - The Gemini model can be selected via `--model` (default gemini-3.6-flash).

Scoring, batching, aggregation, I/O, and multi-round retry logic:
    - Single-TF / AB-MCQ  : each case uploads only Video B
    - Dual-TF / Score-MCQ : each case uploads Video A + Video B
    - Each question is scored independently with its own model_answer / model_reasoning

The checklist is model-agnostic:
    - Video A (original)      = case["src_video"]
    - Video B (model output)  = <gen_video_dir>/<case["id"]>.mp4

Usage:
    export GEMINI_API_KEY=your_key_here
    python eval_coinbench_gemini_public.py \
        --input /path/to/eval_subset_xxx_checklist.jsonl \
        --gen-video-dir /path/to/model_videos \
        --output-name my_model_run \
        --workers 8 \
        --model gemini-3.6-flash
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
MODEL_DEFAULT = "gemini-3.6-flash"
# Batched output needs a larger token budget; empirically one question's
# reasoning + answer is ~250 tokens, so 4096 is insufficient for a 20+ question
# batch. Relaxed to 65k.
MAX_TOKENS = 65000
TEMPERATURE = 0.0
MAX_RETRIES = 5

QUESTION_TYPES = ("Single-TF", "AB-MCQ", "Dual-TF", "Score-MCQ")
DUAL_VIDEO_TYPES = {"Dual-TF", "Score-MCQ"}
PROMPT_FILES = {
    "AB-MCQ": "AB-MCQ.txt",
    "Single-TF": "Single-TF.txt",
    "Dual-TF": "Dual-TF.txt",
    "Score-MCQ": "Score-MCQ-0-10.txt",
}

METRIC_BUCKETS = {
    "Editing Accuracy": ["Semantic Accuracy", "Scope Accuracy", "Edit Persistence"],
    "Physical Naturalness": ["Appearance Naturalness", "Motion Naturalness", "Scale Consistency"],
    "Semantic Preservation": ["Content Preservation"],
}
ALL_METRICS: list[str] = [m for lst in METRIC_BUCKETS.values() for m in lst]
DIMENSION_OF_METRIC: dict[str, str] = {m: d for d, ms in METRIC_BUCKETS.items() for m in ms}
ACC_METRICS = set(METRIC_BUCKETS["Editing Accuracy"]) | set(METRIC_BUCKETS["Physical Naturalness"])
SCORE_METRICS = set(METRIC_BUCKETS["Semantic Preservation"])

BOOL_TRUE = {"yes", "true", "1", "correct"}
BOOL_FALSE = {"no", "false", "0", "incorrect"}

JSONL_SUFFIXES = {".jsonl", ".ndjson"}


# ─────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class LRUFileRefCache:
    """Thread-safe LRU cache of uploaded Gemini File references.

    Caches the File object returned by `client.files.upload()` so the same
    video is not re-uploaded by different batch tasks.
    """

    def __init__(self, client: genai.Client, capacity: int = 16):
        self.client = client
        self.capacity = capacity
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, path: str):
        with self._lock:
            if path in self._data:
                self._data.move_to_end(path)
                return self._data[path]
        # Upload outside the lock (network I/O)
        file_ref = self.client.files.upload(file=path)
        # Wait for the file to become ACTIVE
        while getattr(file_ref, "state", None) == "PROCESSING":
            time.sleep(2)
            file_ref = self.client.files.get(name=file_ref.name)
        with self._lock:
            self._data[path] = file_ref
            self._data.move_to_end(path)
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)
        return file_ref


def norm_text(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def norm_ab(value: Any) -> str:
    """Normalize AB-MCQ answers: 'A', 'B', 'A and B'."""
    s = str(value).strip().upper()
    s = re.sub(r"\s+", " ", s)
    if s in {"A AND B", "B AND A", "A, B", "B, A", "A,B", "B,A", "A&B", "B&A", "BOTH"}:
        return "A AND B"
    if s in {"A", "B"}:
        return s
    return s


def extract_json_from_text(text: str) -> str:
    if not text:
        return ""
    if "```" in text:
        text = text.split("```")[-1].strip()
    m = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Prefer matching an array for batched output
    m = re.search(r"(\[.*\])", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def extract_answer_from_item(item: dict) -> Any:
    for key in ("final_answer", "final_score", "answer", "score"):
        v = item.get(key)
        if v is not None:
            return v
    return None


def score_question(question: dict, model_answer: Any) -> float | None:
    """
    Single-TF / Dual-TF : 1.0 / 0.0
    AB-MCQ              : 1.0 / 0.0
    Score-MCQ           : float in [0, 10]
    """
    if model_answer is None:
        return None
    q_type = question.get("type")

    if q_type == "Score-MCQ":
        try:
            v = float(model_answer)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(10.0, v))

    expected = question.get("expected_answer")
    if expected is None:
        return None

    if q_type in {"Single-TF", "Dual-TF"}:
        pred = norm_text(model_answer)
        gold = norm_text(expected)
        if pred in BOOL_TRUE:
            pred = "yes"
        elif pred in BOOL_FALSE:
            pred = "no"
        if gold in BOOL_TRUE:
            gold = "yes"
        elif gold in BOOL_FALSE:
            gold = "no"
        return 1.0 if pred == gold else 0.0

    if q_type == "AB-MCQ":
        return 1.0 if norm_ab(model_answer) == norm_ab(expected) else 0.0

    return None


# ─────────────────────────────────────────────────────────────
# Gemini API call (batched)
# ─────────────────────────────────────────────────────────────
def build_contents_batch(questions: list[dict], batch_ids: list[str], q_type: str,
                         video_b_ref, video_a_ref) -> list:
    """Build batched contents for one (case, q_type): upload videos once and
    present all questions as a single JSON array.

    Videos are passed as already-uploaded File object references.

    The `id` uses the caller-supplied **batch-unique synthetic id** (`batch_ids`),
    not the question's own `Q1/Q2/...` — because different groups within the same
    case can share the same `q_id`, and using `q_id` directly as the key would
    cause answer misalignment (same-named questions across groups get overwritten).

    The editing instruction is intentionally NOT injected, to avoid
    confirmation bias / leaking the target state.
    """
    assert len(questions) == len(batch_ids), "questions and batch_ids length mismatch"
    llm_input: list[dict] = []
    for bid, q in zip(batch_ids, questions):
        item = {"id": bid, "question": q["question"]}
        if "options" in q and q["options"]:
            item["options"] = q["options"]
        llm_input.append(item)
    qs_text = json.dumps(llm_input, ensure_ascii=False, indent=2)

    contents: list = []
    if q_type in DUAL_VIDEO_TYPES:
        contents.append("Video A (Original Source):\n")
        contents.append(video_a_ref)
        contents.append("\nVideo B (Edited/Generated):\n")
        contents.append(video_b_ref)
    else:
        contents.append("Video for evaluation (this is Video B):\n")
        contents.append(video_b_ref)

    n = len(questions)
    tail = (
        f"\nBelow is a list of {n} independent questions you need to evaluate based "
        "on the video(s). Each question has a unique `id` (an opaque token like "
        "`i0`, `i1`, ...). Treat each question completely independently — do NOT "
        "let one answer bias another.\n\n"
        f"Questions:\n{qs_text}\n\n"
        "CRITICAL OUTPUT REQUIREMENTS:\n"
        f"1. Output STRICTLY as a valid JSON array of EXACTLY {n} objects, "
        "one per input question, in the SAME order as the input.\n"
        "2. Every output object MUST carry the same `id` (verbatim, e.g. `i0`) as "
        "its corresponding input question so we can match them.\n"
        "3. Every output object MUST contain the fields required by the system prompt "
        "(e.g. `reasoning` and `final_answer` / `final_score`).\n"
        "4. Do NOT include any markdown formatting, prose, or explanations outside the JSON array."
    )
    contents.append(tail)
    return contents


def call_gemini(client: genai.Client, model: str,
                system_prompt: str, contents: list) -> str:
    """Call the public Gemini API and return the model response text.

    Uses the `google-genai` SDK's `client.models.generate_content()`.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
    )
    last_err: str = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            text = response.text
            if not text:
                raise RuntimeError("Empty response from Gemini")
            return text
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(min(30, 5 * attempt))
    raise RuntimeError(last_err or "Unknown gemini error")


def parse_answers_batch(raw_text: str, expected_ids: list[str]) -> dict[str, dict]:
    """
    Parse a batch of answers from raw_text and return
    {batch_id: {"answer": ..., "reasoning": ...}}.

    `expected_ids` must be the **batch-unique synthetic id** (e.g. `i0`, `i1`, ...),
    never the question's own `Q1/Q2/...`, otherwise same-named questions across
    groups within a case will overwrite each other.
    IDs that cannot be parsed will not appear in the returned dict; the caller
    uses this to identify failed questions.
    """
    js_text = extract_json_from_text(raw_text)
    try:
        parsed = json.loads(js_text)
    except Exception as e:
        raise ValueError(f"JSON parse failed: {e}; head={js_text[:200]!r}")

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"expected JSON array/object, got: {type(parsed).__name__}")

    # Index by id first; fall back to positional order for unlabeled items
    by_id: dict[str, dict] = {}
    unlabeled: list[dict] = []
    for it in parsed:
        if not isinstance(it, dict):
            continue
        rid = it.get("id")
        if rid is None:
            unlabeled.append(it)
        else:
            by_id[str(rid).strip()] = it

    result: dict[str, dict] = {}
    for i, qid in enumerate(expected_ids):
        item = by_id.get(str(qid).strip())
        if item is None and i < len(unlabeled):
            # Some models occasionally omit the id; fill in by order
            item = unlabeled[i]
        if item is None:
            continue
        ans = extract_answer_from_item(item)
        if ans is None:
            continue
        result[str(qid).strip()] = {
            "answer": ans,
            "reasoning": item.get("reasoning", ""),
        }
    return result


# ─────────────────────────────────────────────────────────────
# I/O: checklist / prompts
# ─────────────────────────────────────────────────────────────
def load_prompts(prompt_dir: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for q_type, fname in PROMPT_FILES.items():
        p = prompt_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"system prompt not found: {p}")
        prompts[q_type] = p.read_text(encoding="utf-8").strip()
    return prompts


def atomic_save_json(data: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_save_jsonl(items: list, path: Path) -> None:
    """Atomically write a list of dicts as JSONL (one object per line, no indent)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False))
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"invalid JSONL at {path}:{lineno}: {e}; head={s[:200]!r}"
                ) from e


def load_checklist(path: Path) -> tuple[list, bool]:
    """Load a checklist, auto-detecting JSON / JSONL. Returns (checklist_list, is_jsonl).

    - `.jsonl` / `.ndjson` suffix: read as JSONL
    - `.json` suffix: try as a single JSON array first; if not a list or parse
      fails, fall back to JSONL
    - Other suffixes: try JSON first, then JSONL on failure
    """
    suffix = path.suffix.lower()
    if suffix in JSONL_SUFFIXES:
        items = list(_iter_jsonl(path))
        return items, True

    # Try as a single JSON document
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data, False
        # A single dict is also supported
        if isinstance(data, dict):
            return [data], False
        raise ValueError(f"top-level JSON must be list/dict, got {type(data).__name__}")
    except json.JSONDecodeError:
        # Fall back to JSONL
        items = list(_iter_jsonl(path))
        return items, True


class Store:
    """Thread-safe incremental checklist storage. Supports JSON or JSONL output."""

    def __init__(self, checklist: list, out_path: Path, flush_every: int = 20,
                 as_jsonl: bool = False):
        self.checklist = checklist
        self.out_path = out_path
        self.flush_every = flush_every
        self.as_jsonl = as_jsonl
        self._lock = threading.Lock()
        self._dirty = 0

    def _dump(self):
        if self.as_jsonl:
            atomic_save_jsonl(self.checklist, self.out_path)
        else:
            atomic_save_json(self.checklist, self.out_path)

    def mark_dirty(self, n: int = 1):
        with self._lock:
            self._dirty += n
            if self._dirty >= self.flush_every:
                self._dump()
                self._dirty = 0

    def flush(self):
        with self._lock:
            self._dump()
            self._dirty = 0


class ErrorLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._items: list[dict] = []
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    self._items = json.load(f)
            except Exception:
                self._items = []

    def add(self, rec: dict):
        with self._lock:
            self._items.append(rec)
            try:
                atomic_save_json(self._items, self.path)
            except Exception:
                pass

    def add_many(self, recs: list[dict]):
        if not recs:
            return
        with self._lock:
            self._items.extend(recs)
            try:
                atomic_save_json(self._items, self.path)
            except Exception:
                pass

    @property
    def items(self):
        return list(self._items)


# ─────────────────────────────────────────────────────────────
# Task construction & execution (batched)
# ─────────────────────────────────────────────────────────────
def build_batch_tasks(checklist: list, overwrite: bool,
                      gen_video_dir: Path, gen_video_ext: str) -> list[dict]:
    """
    Aggregate tasks by (case, q_type). Each task contains all not-yet-evaluated
    question references for that case and question type.
    """
    tasks: list[dict] = []
    for case_idx, case in enumerate(checklist):
        case_id = case.get("id", case_idx)
        video_a = case.get("src_video")
        video_b = str(gen_video_dir / f"{case_id}{gen_video_ext}")

        # Aggregate pending question references by q_type (with group/q indices for error tracing)
        by_type: dict[str, list[dict]] = defaultdict(list)
        for gi, group in enumerate(case.get("evaluation_groups", []) or []):
            for qi, q in enumerate(group.get("questions", []) or []):
                q_type = q.get("type")
                if q_type not in QUESTION_TYPES:
                    continue
                if not overwrite and q.get("model_answer") is not None:
                    continue
                by_type[q_type].append({
                    "group_idx": gi,
                    "q_idx": qi,
                    "q_id": q.get("id"),
                    "question_ref": q,
                })

        for q_type, items in by_type.items():
            if not items:
                continue
            tasks.append({
                "case_idx": case_idx,
                "case_id": case_id,
                "q_type": q_type,
                "video_a": video_a,
                "video_b": video_b,
                "items": items,  # list of {group_idx, q_idx, q_id, question_ref}
            })
    return tasks


def clear_model_answers(checklist: list):
    for case in checklist:
        for g in case.get("evaluation_groups", []) or []:
            for q in g.get("questions", []) or []:
                q.pop("model_answer", None)
                q.pop("model_reasoning", None)


def eval_batch(task: dict, prompts: dict[str, str],
               client: genai.Client, model: str,
               cache: LRUFileRefCache) -> dict:
    """
    Execute one (case, q_type) batch task. Returns:
        {
          "task": task,
          "results": [ {q_id, ok, model_answer?, model_reasoning?, error?} ... ],
          "call_error": str | None,   # non-empty when the whole batch call failed
          "raw_head": str,            # for debugging
        }
    """
    q_type = task["q_type"]
    items = task["items"]
    # Different groups within the same case can share the same q_id (e.g. group0/Q2
    # and group1/Q2), so never use q_id directly as the batch key. Generate a
    # batch-unique synthetic id `i0/i1/...` for Gemini; map back to
    # (group_idx, q_idx) precisely when writing answers.
    batch_ids = [f"i{k}" for k in range(len(items))]

    # Video path checks (the whole batch shares the same video pair)
    video_b = task["video_b"]
    video_a = task["video_a"] if q_type in DUAL_VIDEO_TYPES else None
    if not video_b or not Path(video_b).is_file():
        return {"task": task, "results": [],
                "call_error": f"video B missing: {video_b}", "raw_head": ""}
    if q_type in DUAL_VIDEO_TYPES and (not video_a or not Path(video_a).is_file()):
        return {"task": task, "results": [],
                "call_error": f"video A missing: {video_a}", "raw_head": ""}

    try:
        vb_ref = cache.get(video_b)
        va_ref = cache.get(video_a) if video_a else None
    except Exception as e:  # noqa: BLE001
        return {"task": task, "results": [],
                "call_error": f"file upload failed: {e}", "raw_head": ""}

    questions = [it["question_ref"] for it in items]
    contents = build_contents_batch(questions, batch_ids, q_type, vb_ref, va_ref)

    try:
        raw = call_gemini(client, model, prompts[q_type], contents)
    except Exception as e:  # noqa: BLE001
        return {"task": task, "results": [],
                "call_error": f"gemini call failed: {e}", "raw_head": ""}

    try:
        parsed_map = parse_answers_batch(raw, batch_ids)
    except Exception as e:  # noqa: BLE001
        return {"task": task, "results": [],
                "call_error": f"parse failed: {e}", "raw_head": raw[:400]}

    # Build per-question results: map batch_id strictly to (group_idx, q_idx)
    per_q: list[dict] = []
    for bid, it in zip(batch_ids, items):
        got = parsed_map.get(bid)
        if got is None:
            per_q.append({
                "group_idx": it["group_idx"],
                "q_idx": it["q_idx"],
                "q_id": it["q_id"],
                "ok": False,
                "error": "missing in batch response",
            })
        else:
            per_q.append({
                "group_idx": it["group_idx"],
                "q_idx": it["q_idx"],
                "q_id": it["q_id"],
                "ok": True,
                "model_answer": got["answer"],
                "model_reasoning": got["reasoning"],
            })
    return {"task": task, "results": per_q,
            "call_error": None, "raw_head": raw[:400]}


# ─────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────
def aggregate(checklist: list) -> tuple[dict, list[dict]]:
    metric_scores: dict[str, list[float]] = {m: [] for m in ALL_METRICS}
    metric_correct: dict[str, int] = {m: 0 for m in ALL_METRICS}
    metric_total: dict[str, int] = {m: 0 for m in ALL_METRICS}

    total_questions = 0
    answered_questions = 0
    per_case: list[dict] = []
    cases_all_correct = 0
    total_cases = 0

    for case in checklist:
        total_cases += 1
        case_id = case.get("id")
        case_metric_scores: dict[str, list[float]] = defaultdict(list)
        case_metric_correct: dict[str, int] = defaultdict(int)
        case_metric_total: dict[str, int] = defaultdict(int)
        ea_pn_total = 0
        ea_pn_correct = 0

        for g in case.get("evaluation_groups", []) or []:
            for q in g.get("questions", []) or []:
                q_type = q.get("type")
                if q_type not in QUESTION_TYPES:
                    continue
                metric = q.get("metric")
                if metric not in ALL_METRICS:
                    continue
                total_questions += 1

                model_answer = q.get("model_answer")
                if model_answer is None:
                    continue
                score = score_question(q, model_answer)
                if score is None:
                    continue
                answered_questions += 1

                metric_scores[metric].append(score)
                metric_total[metric] += 1
                if metric in ACC_METRICS and score == 1.0:
                    metric_correct[metric] += 1

                case_metric_scores[metric].append(score)
                case_metric_total[metric] += 1
                if metric in ACC_METRICS:
                    if score == 1.0:
                        case_metric_correct[metric] += 1
                    ea_pn_total += 1
                    if score == 1.0:
                        ea_pn_correct += 1

        all_correct = 1 if (ea_pn_total > 0 and ea_pn_correct == ea_pn_total) else 0
        if all_correct:
            cases_all_correct += 1

        per_case.append({
            "id": case_id,
            "all_correct": all_correct,
            "ea_pn_correct": ea_pn_correct,
            "ea_pn_total": ea_pn_total,
            "metrics": {
                m: (
                    {
                        "num_questions": case_metric_total.get(m, 0),
                        "num_correct": case_metric_correct.get(m, 0),
                        "score_raw": (
                            case_metric_correct.get(m, 0) / case_metric_total[m]
                            if case_metric_total.get(m) else None
                        ),
                    } if m in ACC_METRICS else
                    {
                        "num_questions": case_metric_total.get(m, 0),
                        "score_raw": (
                            sum(case_metric_scores.get(m, [])) / case_metric_total[m]
                            if case_metric_total.get(m) else None
                        ),
                    }
                )
                for m in ALL_METRICS
            },
        })

    metrics_out: dict[str, dict] = {}
    for m in ALL_METRICS:
        n = metric_total[m]
        if m in ACC_METRICS:
            raw = (metric_correct[m] / n) if n else None
            metrics_out[m] = {
                "score_raw": None if raw is None else round(raw, 6),
                "score_100": None if raw is None else round(raw * 100.0, 4),
                "num_questions": n,
                "num_correct": metric_correct[m],
            }
        else:
            raw = (sum(metric_scores[m]) / n) if n else None
            metrics_out[m] = {
                "score_raw": None if raw is None else round(raw, 6),
                "score_100": None if raw is None else round(raw * 10.0, 4),
                "num_questions": n,
            }

    dimensions_out: dict[str, dict] = {}
    for dim, ms in METRIC_BUCKETS.items():
        vals = [metrics_out[m]["score_100"] for m in ms
                if metrics_out[m].get("score_100") is not None]
        dimensions_out[dim] = {
            "score_100": round(sum(vals) / len(vals), 4) if vals else None,
            "num_metrics": len(vals),
            "metrics": ms,
        }

    stats = {
        "generated_at": _now(),
        "summary": {
            "total_cases": total_cases,
            "total_questions": total_questions,
            "answered_questions": answered_questions,
            "failed_questions": total_questions - answered_questions,
        },
        "metrics": metrics_out,
        "dimensions": dimensions_out,
        "case_all_correct": {
            "definition": "All Editing Accuracy + Physical Naturalness questions in "
                          "a case are correct (Content Preservation excluded)",
            "num_cases_all_correct": cases_all_correct,
            "total_cases": total_cases,
            "ratio": round(cases_all_correct / total_cases, 6) if total_cases else None,
            "ratio_100": round(cases_all_correct / total_cases * 100.0, 4) if total_cases else None,
        },
    }
    return stats, per_case


# ─────────────────────────────────────────────────────────────
# Single eval round (called by main for multi-round retry)
# ─────────────────────────────────────────────────────────────
def run_eval_round(checklist: list, prompts: dict[str, str],
                   gen_video_dir: Path, gen_video_ext: str,
                   client: genai.Client, model: str,
                   cache: LRUFileRefCache, store: Store, errors: ErrorLog,
                   workers: int, round_idx: int,
                   evaluated_path: Path | None = None) -> dict:
    """Run one evaluation round.

    build_batch_tasks is re-invoked each round — it automatically picks only
    questions with `model_answer is None`, so multi-round retry just repeats
    this call to make progress.

    Args:
        evaluated_path: only used for logging; persistence is handled by store.flush().

    Returns:
        {
          "round": round_idx,
          "ok_batches": int, "fail_batches": int,
          "ok_qs": int, "fail_qs": int,
          "pending_batches": int,   # remaining unanswered batches after this round
          "pending_questions": int, # remaining unanswered questions after this round
        }
    """
    tasks = build_batch_tasks(checklist, overwrite=False,
                              gen_video_dir=gen_video_dir,
                              gen_video_ext=gen_video_ext)
    total_pending_q = sum(len(t["items"]) for t in tasks)
    print(f"[round {round_idx}] pending_batches={len(tasks)}  "
          f"pending_questions={total_pending_q}  workers={workers}")
    if not tasks:
        # No pending questions; return 0 pending directly
        return {
            "round": round_idx, "ok_batches": 0, "fail_batches": 0,
            "ok_qs": 0, "fail_qs": 0,
            "pending_batches": 0, "pending_questions": 0,
        }

    ok_batches, fail_batches = 0, 0
    ok_qs, fail_qs = 0, 0
    pbar = tqdm(total=len(tasks), desc=f"batch-eval-r{round_idx}", ncols=110)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(eval_batch, t, prompts, client, model, cache)
                   for t in tasks]
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                fail_batches += 1
                pbar.update(1)
                pbar.set_postfix_str(f"ok_b={ok_batches} fail_b={fail_batches} "
                                     f"ok_q={ok_qs} fail_q={fail_qs}")
                tqdm.write(f"[fail-batch] unexpected: {e}")
                continue

            task = res["task"]
            call_error = res.get("call_error")
            if call_error:
                # Whole batch failed: record an error for every question (with round field)
                fail_batches += 1
                err_recs = []
                for it in task["items"]:
                    fail_qs += 1
                    err_recs.append({
                        "round": round_idx,
                        "case_idx": task["case_idx"],
                        "case_id": task["case_id"],
                        "group_idx": it["group_idx"],
                        "q_idx": it["q_idx"],
                        "q_id": it["q_id"],
                        "q_type": task["q_type"],
                        "video_a": task["video_a"],
                        "video_b": task["video_b"],
                        "error": call_error,
                        "raw_head": res.get("raw_head", ""),
                        "time": _now(),
                    })
                errors.add_many(err_recs)
                tqdm.write(f"[fail-batch] case={task['case_id']} "
                           f"type={task['q_type']} n={len(task['items'])} "
                           f":: {call_error[:160]}")
            else:
                ok_batches += 1
                # Write back per-question results
                batch_ok = 0
                batch_fail_recs = []
                for r in res["results"]:
                    it_ref = None
                    for it in task["items"]:
                        if it["group_idx"] == r["group_idx"] and it["q_idx"] == r["q_idx"]:
                            it_ref = it
                            break
                    if it_ref is None:
                        continue
                    q = it_ref["question_ref"]
                    if r["ok"]:
                        q["model_answer"] = r["model_answer"]
                        q["model_reasoning"] = r.get("model_reasoning", "")
                        batch_ok += 1
                        ok_qs += 1
                    else:
                        fail_qs += 1
                        batch_fail_recs.append({
                            "round": round_idx,
                            "case_idx": task["case_idx"],
                            "case_id": task["case_id"],
                            "group_idx": r["group_idx"],
                            "q_idx": r["q_idx"],
                            "q_id": r["q_id"],
                            "q_type": task["q_type"],
                            "video_a": task["video_a"],
                            "video_b": task["video_b"],
                            "error": r.get("error", ""),
                            "raw_head": res.get("raw_head", ""),
                            "time": _now(),
                        })
                errors.add_many(batch_fail_recs)
                if batch_ok:
                    store.mark_dirty(batch_ok)

            pbar.update(1)
            pbar.set_postfix_str(f"ok_b={ok_batches} fail_b={fail_batches} "
                                 f"ok_q={ok_qs} fail_q={fail_qs}")
    pbar.close()
    store.flush()
    if evaluated_path is not None:
        print(f"[round {round_idx}] done  batches ok={ok_batches} fail={fail_batches}  |  "
              f"questions ok={ok_qs} fail={fail_qs}  -> {evaluated_path}")

    # Re-count pending (build_batch_tasks automatically skips answered questions)
    remaining_tasks = build_batch_tasks(checklist, overwrite=False,
                                        gen_video_dir=gen_video_dir,
                                        gen_video_ext=gen_video_ext)
    pending_batches = len(remaining_tasks)
    pending_questions = sum(len(t["items"]) for t in remaining_tasks)
    return {
        "round": round_idx,
        "ok_batches": ok_batches, "fail_batches": fail_batches,
        "ok_qs": ok_qs, "fail_qs": fail_qs,
        "pending_batches": pending_batches,
        "pending_questions": pending_questions,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CoinBench checklist with public Gemini API "
                    "(BATCHED per case x question type, JSON/JSONL input)."
    )
    parser.add_argument("--input", required=True,
                        help="Checklist path, supports .json (list) or .jsonl (one case per line)")
    parser.add_argument("--gen-video-dir", default="",
                        help="Output video directory of the model to evaluate "
                             "(videos named {id}.mp4). Required unless --only-aggregate.")
    parser.add_argument("--gen-video-ext", default=".mp4",
                        help="Generated video extension, default .mp4")
    parser.add_argument("--prompt-dir", default="",
                        help="system_prompts directory, defaults to ./system_prompts next to this script")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default=MODEL_DEFAULT,
                        help=f"Gemini model name, default {MODEL_DEFAULT}")
    parser.add_argument("--overwrite", action="store_true",
                        help="Clear existing model_answer and rerun everything")
    parser.add_argument("--flush-every", type=int, default=20,
                        help="Flush to disk every N answered questions")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N cases (0 = all)")
    parser.add_argument("--only-aggregate", action="store_true",
                        help="Do not call Gemini; only aggregate existing model_answer")
    parser.add_argument("--outputs-root",
                        default="/cfs/cfs-ho82q7ml/Benchmark/CoVEBench/outputs",
                        help="Root directory for all evaluation outputs, default CoVEBench/outputs")
    parser.add_argument("--output-name", default="",
                        help="Output subdirectory name; final path is <outputs-root>/<output-name>/. "
                             "Defaults to the last component of --gen-video-dir.")
    parser.add_argument("--out-dir", default="",
                        help="Directly specify an absolute output directory (highest priority, "
                             "overrides --output-name / --outputs-root).")
    parser.add_argument("--evaluated-format", choices=("auto", "json", "jsonl"),
                        default="auto",
                        help="evaluated output format: auto=follow input; json=force JSON; "
                             "jsonl=force JSONL. Default auto.")
    parser.add_argument("--evaluated-suffix", default="",
                        help="evaluated file suffix. Empty auto-selects by --evaluated-format: "
                             "JSON -> '_evaluated.json', JSONL -> '_evaluated.jsonl'.")
    parser.add_argument("--stats-suffix", default="_stats.json")
    parser.add_argument("--per-case-suffix", default="_per_case.json")
    parser.add_argument("--errors-suffix", default="_eval.errors.json")
    parser.add_argument("--timestamp", action="store_true",
                        help="Add a timestamp suffix to stats / per_case / errors report files "
                             "(the evaluated file is not suffixed, to keep resume working).")
    parser.add_argument("--timestamp-tag", default="",
                        help="Custom timestamp tag; defaults to YYYYmmdd_HHMMSS.")
    parser.add_argument("--max-rounds", type=int, default=10,
                        help="Maximum number of internal retry rounds, default 10. After each round, "
                             "if there are still pending questions, sleep --round-sleep seconds and continue; "
                             "exit immediately when pending=0.")
    parser.add_argument("--round-sleep", type=float, default=30.0,
                        help="Sleep seconds between rounds, default 30. Used to wait for API rate-limit recovery.")
    args = parser.parse_args()

    # ─── Auth: GEMINI_API_KEY ─────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.only_aggregate:
        raise SystemExit(
            "GEMINI_API_KEY environment variable is required. "
            "Set it via: export GEMINI_API_KEY=your_key"
        )

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        raise SystemExit(f"input not found: {in_path}")

    stem = in_path.stem

    gen_video_dir: Path | None = None
    if args.gen_video_dir:
        gen_video_dir = Path(args.gen_video_dir).resolve()
        if not gen_video_dir.is_dir():
            raise SystemExit(f"--gen-video-dir not a directory: {gen_video_dir}")
    elif not args.only_aggregate:
        raise SystemExit(
            "--gen-video-dir is required (unless --only-aggregate)."
        )

    if args.output_name:
        output_name = args.output_name
    elif gen_video_dir is not None:
        output_name = gen_video_dir.name
    else:
        output_name = ""

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    elif output_name:
        out_dir = (Path(args.outputs_root).resolve() / output_name).resolve()
    else:
        out_dir = in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out_dir] {out_dir}")
    if gen_video_dir is not None:
        print(f"[gen_video_dir] {gen_video_dir}")

    if args.timestamp or args.timestamp_tag:
        ts = args.timestamp_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_infix = f"_{ts}"
        print(f"[timestamp] {ts}")
    else:
        ts_infix = ""

    def _with_ts(suffix: str) -> str:
        if not ts_infix:
            return suffix
        p = Path(suffix)
        return f"{p.stem}{ts_infix}{p.suffix}"

    # ─── Load checklist (auto-detect JSON / JSONL) ────────────
    print(f"[load] {in_path}")
    input_checklist, input_is_jsonl = load_checklist(in_path)
    print(f"[load] format={'jsonl' if input_is_jsonl else 'json'}  "
          f"cases_loaded={len(input_checklist)}")

    # ─── Decide evaluated output format ───────────────────────
    if args.evaluated_format == "json":
        evaluated_is_jsonl = False
    elif args.evaluated_format == "jsonl":
        evaluated_is_jsonl = True
    else:  # auto
        evaluated_is_jsonl = input_is_jsonl

    if args.evaluated_suffix:
        evaluated_suffix = args.evaluated_suffix
    else:
        evaluated_suffix = "_evaluated.jsonl" if evaluated_is_jsonl else "_evaluated.json"

    evaluated_path = out_dir / f"{stem}{evaluated_suffix}"
    stats_path = out_dir / f"{stem}{_with_ts(args.stats_suffix)}"
    per_case_path = out_dir / f"{stem}{_with_ts(args.per_case_suffix)}"
    errors_path = out_dir / f"{stem}{_with_ts(args.errors_suffix)}"

    prompt_dir = (Path(args.prompt_dir).resolve()
                  if args.prompt_dir else Path(__file__).resolve().parent / "system_prompts")

    # Resume: prefer loading an already-generated evaluated file
    if evaluated_path.exists() and not args.overwrite:
        print(f"[load] resume from {evaluated_path}")
        checklist, _ = load_checklist(evaluated_path)
    else:
        checklist = input_checklist

    if args.overwrite:
        clear_model_answers(checklist)
    if args.limit > 0:
        checklist = checklist[: args.limit]

    total_q_all = sum(
        1
        for c in checklist
        for g in c.get("evaluation_groups", []) or []
        for q in g.get("questions", []) or []
        if q.get("type") in QUESTION_TYPES
    )
    already_done = sum(
        1
        for c in checklist
        for g in c.get("evaluation_groups", []) or []
        for q in g.get("questions", []) or []
        if q.get("type") in QUESTION_TYPES and q.get("model_answer") is not None
    )
    print(f"[stats] cases={len(checklist)}  total_q={total_q_all}  already_done={already_done}")
    print(f"[stats] evaluated_out={'jsonl' if evaluated_is_jsonl else 'json'} -> {evaluated_path}")

    if not args.only_aggregate:
        prompts = load_prompts(prompt_dir)
        assert gen_video_dir is not None

        # Inject the edit_video field (same placement strategy as the original)
        for case in checklist:
            case_id = case.get("id")
            if case_id is None:
                continue
            new_edit_video = str(gen_video_dir / f"{case_id}{args.gen_video_ext}")
            anchor = "src_video" if "src_video" in case else None
            if anchor is None:
                case["edit_video"] = new_edit_video
                continue
            reordered = {}
            for k, v in case.items():
                if k == "edit_video":
                    continue
                reordered[k] = v
                if k == anchor:
                    reordered["edit_video"] = new_edit_video
            case.clear()
            case.update(reordered)

        # Pre-check pending (logging only)
        pre_tasks = build_batch_tasks(checklist, overwrite=False,
                                      gen_video_dir=gen_video_dir,
                                      gen_video_ext=args.gen_video_ext)
        total_pending_q = sum(len(t["items"]) for t in pre_tasks)
        print(f"[stats] pending_batches={len(pre_tasks)}  "
              f"pending_questions={total_pending_q}  workers={args.workers}  "
              f"model={args.model}  max_rounds={args.max_rounds} round_sleep={args.round_sleep}")
        if not pre_tasks:
            print("[stats] nothing to do; go aggregate.")
        else:
            client = genai.Client(api_key=api_key)
            cache = LRUFileRefCache(client, capacity=16)
            store = Store(checklist, evaluated_path, flush_every=args.flush_every,
                          as_jsonl=evaluated_is_jsonl)
            errors = ErrorLog(errors_path)

            # ─── Multi-round retry loop ─────────────────────────
            # After each round, if there are still pending questions, sleep
            # round_sleep seconds and continue; exit immediately when pending=0;
            # otherwise run up to max_rounds rounds.
            round_idx = 0
            while round_idx < args.max_rounds:
                round_idx += 1
                r = run_eval_round(checklist=checklist, prompts=prompts,
                                   gen_video_dir=gen_video_dir,
                                   gen_video_ext=args.gen_video_ext,
                                   client=client, model=args.model,
                                   cache=cache, store=store, errors=errors,
                                   workers=args.workers, round_idx=round_idx,
                                   evaluated_path=evaluated_path)
                print(f"[round {round_idx}/{args.max_rounds}] "
                      f"ok_batches={r['ok_batches']} fail_batches={r['fail_batches']} "
                      f"ok_q={r['ok_qs']} fail_q={r['fail_qs']} "
                      f"pending={r['pending_questions']}")
                if r["pending_questions"] == 0:
                    print(f"[round {round_idx}] all questions answered; exit retry loop")
                    break
                if round_idx < args.max_rounds:
                    print(f"[round {round_idx}] sleep {args.round_sleep}s before next round ...")
                    time.sleep(args.round_sleep)
                else:
                    print(f"[round {round_idx}] reached max rounds; stopping retry. "
                          f"remaining unanswered: {r['pending_questions']}")

    print("[aggregate] computing metrics …")
    stats, per_case = aggregate(checklist)
    atomic_save_json(stats, stats_path)
    atomic_save_json({"generated_at": _now(),
                      "total_cases": len(per_case),
                      "cases": per_case}, per_case_path)

    if not evaluated_path.exists():
        if evaluated_is_jsonl:
            atomic_save_jsonl(checklist, evaluated_path)
        else:
            atomic_save_json(checklist, evaluated_path)

    print("\n" + "=" * 72)
    print(f"stats -> {stats_path}")
    print(f"per_case -> {per_case_path}")
    print(f"evaluated -> {evaluated_path}")
    print(f"errors -> {errors_path}")
    print("-" * 72)
    print(f"{'Metric':32s} {'raw':>10s}  {'/100':>8s}  {'n':>6s}")
    for m in ALL_METRICS:
        v = stats["metrics"][m]
        raw_s = "-" if v["score_raw"] is None else f"{v['score_raw']:.4f}"
        s100 = "-" if v["score_100"] is None else f"{v['score_100']:.2f}"
        print(f"  {m:30s} {raw_s:>10s}  {s100:>8s}  {v['num_questions']:>6d}")
    print("-" * 72)
    for dim, v in stats["dimensions"].items():
        s = "-" if v["score_100"] is None else f"{v['score_100']:.2f}"
        print(f"  [{dim:22s}] score_100 = {s}")
    cac = stats["case_all_correct"]
    print(f"  case_all_correct: {cac['num_cases_all_correct']}/{cac['total_cases']} "
          f"= {cac['ratio_100']}%  (definition: all EA+PN correct)")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupt] user cancelled", file=sys.stderr)
        sys.exit(130)
