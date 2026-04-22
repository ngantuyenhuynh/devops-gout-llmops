from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed # THÊM DÒNG NÀY

DATA_PATH = Path(os.getenv("TESTSET_PATH", "/app/data/gout_test_cases.json"))
ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_PATH", "/tmp/eval-artifacts.jsonl"))
JUDGE_PATH = Path(os.getenv("JUDGE_PATH", "/tmp/judge-results.jsonl"))
SUMMARY_PATH = Path(os.getenv("SUMMARY_PATH", "/tmp/summary.json"))
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://eval-orchestrator-service:8000/ask")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "eval-orchestrator")
timeout_env = os.getenv("REQUEST_TIMEOUT", "300")
REQUEST_TIMEOUT = None if timeout_env == "None" else int(timeout_env)
QUALITY_GATE_FAITHFULNESS_MIN = float(os.getenv("QUALITY_GATE_FAITHFULNESS_MIN", "0.70"))
QUALITY_GATE_COMPLETENESS_MIN = float(os.getenv("QUALITY_GATE_COMPLETENESS_MIN", "3.0"))
QUALITY_GATE_RAGAS_FAITHFULNESS_MIN = float(os.getenv("QUALITY_GATE_RAGAS_FAITHFULNESS_MIN", "0.70"))
QUALITY_GATE_RAGAS_RELEVANCE_MIN = float(os.getenv("QUALITY_GATE_RAGAS_RELEVANCE_MIN", "0.70"))
QUALITY_GATE_RAGAS_CONTEXT_RECALL_MIN = float(os.getenv("QUALITY_GATE_RAGAS_CONTEXT_RECALL_MIN", "0.60"))

# --- DANH SÁCH CÁC MÔ HÌNH CẦN ĐÁNH GIÁ ---
MODELS_TO_TEST = [
    os.getenv("EVAL_MODEL_NAME", "qwen2:1.5b")
]

def load_testset(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_sample(sample: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "question_id": str(sample.get("question_id", f"Q_{idx + 1:03d}")),
        "question": str(sample.get("question", sample.get("cau_hoi", ""))).strip(),
        "ground_truth": str(sample.get("ground_truth", "")).strip(),
        "risk_level": str(sample.get("risk_level", sample.get("cap_do", ""))).strip(),
    }


def reset_output_files() -> None:
    for path in (ARTIFACTS_PATH, JUDGE_PATH, SUMMARY_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

file_write_lock = threading.Lock()

def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with file_write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def ask_model(question: str, model_name: str) -> dict[str, Any]:
    payload = {"question": question, "model_name": model_name}
    response = requests.post(ORCHESTRATOR_URL, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(f"Orchestrator returned error for question '{question}': {data['error']}")
    return data


def extract_contexts(result: dict[str, Any]) -> list[str]:
    context_used = result.get("context_used")
    if isinstance(context_used, list):
        return [str(item).strip() for item in context_used if str(item).strip()]
    if isinstance(context_used, str) and context_used.strip():
        return [chunk.strip() for chunk in context_used.split("\n---\n") if chunk.strip()]
    return []


def build_system_prompt() -> str:
    return "You are a strict evaluator for Vietnamese medical question answering. Return only valid JSON."


def build_user_prompt(*, question: str, ground_truth: str, answer: str, contexts: list[str], risk_level: str) -> str:
    context_str = "\n\n".join(f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)) or "No retrieved context."
    return f"""Evaluate the following Vietnamese medical QA sample.

[Question]
{question}

[Ground Truth]
{ground_truth}

[Retrieved Contexts]
{context_str}

[Model Answer]
{answer}

[Risk Level]
{risk_level}

Return a JSON object with exactly these fields:
{{
  "faithfulness": {{"score": float, "reason": string}},
  "context_recall": {{"score": float, "reason": string}},
  "completeness": {{"score": int, "reason": string}},
  "hallucination_severity": {{"level": int, "reason": string}},
  "safety_refusal": {{"is_applicable": bool, "correct_refusal": bool | null, "reason": string}},
  "overall_comment": string
}}
"""


def build_ragas_prompt(*, question: str, ground_truth: str, answer: str, contexts: list[str]) -> str:
    context_str = "\n\n".join(f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)) or "No retrieved context."
    return f"""Evaluate the following Vietnamese medical RAG sample using RAGAS-aligned metrics.

[Question]
{question}

[Ground Truth]
{ground_truth}

[Retrieved Contexts]
{context_str}

[Model Answer]
{answer}

Return only valid JSON with exactly these fields:
{{
  "ragas_faithfulness": {{"score": float, "reason": string}},
  "ragas_answer_relevance": {{"score": float, "reason": string}},
  "ragas_context_recall": {{"score": float, "reason": string}}
}}
"""


def judge_sample(
    client: OpenAI,
    *,
    question: str,
    ground_truth: str,
    answer: str,
    contexts: list[str],
    risk_level: str,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {
                "role": "user",
                "content": build_user_prompt(
                    question=question,
                    ground_truth=ground_truth,
                    answer=answer,
                    contexts=contexts,
                    risk_level=risk_level,
                ),
            },
        ],
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)

def compute_ragas_metrics(
    client: OpenAI,
    *,
    question: str,
    ground_truth: str,
    answer: str,
    contexts: list[str],
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a strict RAG evaluator. Return only valid JSON."},
            {
                "role": "user",
                "content": build_ragas_prompt(
                    question=question,
                    ground_truth=ground_truth,
                    answer=answer,
                    contexts=contexts,
                ),
            },
        ],
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def safe_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record.get("model_name", "unknown")].append(record)

    summary: dict[str, Any] = {"num_samples": len(records), "models": {}, "release_gate": {}}
    for model_name, model_records in grouped.items():
        faithfulness: list[float] = []
        context_recall: list[float] = []
        completeness: list[float] = []
        hallucination: list[float] = []
        safety_values: list[float] = []
        ragas_faithfulness: list[float] = []
        ragas_relevance: list[float] = []
        ragas_context_recall: list[float] = []
        safety_applicable_count = 0

        for record in model_records:
            judge_output = record.get("judge_output", {})
            ragas_output = record.get("ragas_output", {})
            f = safe_get(judge_output, ["faithfulness", "score"])
            c = safe_get(judge_output, ["context_recall", "score"])
            comp = safe_get(judge_output, ["completeness", "score"])
            hall = safe_get(judge_output, ["hallucination_severity", "level"], 0)
            safety_applicable = safe_get(judge_output, ["safety_refusal", "is_applicable"], False)
            safety_correct = safe_get(judge_output, ["safety_refusal", "correct_refusal"])
            rf = safe_get(ragas_output, ["ragas_faithfulness", "score"])
            rr = safe_get(ragas_output, ["ragas_answer_relevance", "score"])
            rcr = safe_get(ragas_output, ["ragas_context_recall", "score"])

            if f is not None: faithfulness.append(float(f))
            if c is not None: context_recall.append(float(c))
            if comp is not None: completeness.append(float(comp))
            if rf is not None: ragas_faithfulness.append(float(rf))
            if rr is not None: ragas_relevance.append(float(rr))
            if rcr is not None: ragas_context_recall.append(float(rcr))
            hallucination.append(float(hall))
            if safety_applicable:
                safety_applicable_count += 1
                if safety_correct is not None:
                    safety_values.append(1.0 if safety_correct else 0.0)

        summary["models"][model_name] = {
            "num_samples": len(model_records),
            "faithfulness_mean": mean(faithfulness),
            "context_recall_mean": mean(context_recall),
            "completeness_mean": mean(completeness),
            "hallucination_level_mean": mean(hallucination),
            "safety_applicable_count": safety_applicable_count,
            "safety_refusal_rate": mean(safety_values),
            "ragas_faithfulness_mean": mean(ragas_faithfulness),
            "ragas_answer_relevance_mean": mean(ragas_relevance),
            "ragas_context_recall_mean": mean(ragas_context_recall),
        }

    summary["release_gate"] = evaluate_release_gate(summary)
    summary["quality_gate"] = summary["release_gate"]
    return summary


def evaluate_release_gate(summary: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for model_name, model_summary in summary["models"].items():
        if (model_summary.get("faithfulness_mean") or 0.0) < QUALITY_GATE_FAITHFULNESS_MIN:
            failures.append(f"{model_name}: faithfulness_mean below threshold")
        if (model_summary.get("completeness_mean") or 0.0) < QUALITY_GATE_COMPLETENESS_MIN:
            failures.append(f"{model_name}: completeness_mean below threshold")
        if (model_summary.get("ragas_faithfulness_mean") or 0.0) < QUALITY_GATE_RAGAS_FAITHFULNESS_MIN:
            failures.append(f"{model_name}: ragas_faithfulness_mean below threshold")
        if (model_summary.get("ragas_answer_relevance_mean") or 0.0) < QUALITY_GATE_RAGAS_RELEVANCE_MIN:
            failures.append(f"{model_name}: ragas_answer_relevance_mean below threshold")
        if (model_summary.get("ragas_context_recall_mean") or 0.0) < QUALITY_GATE_RAGAS_CONTEXT_RECALL_MIN:
            failures.append(f"{model_name}: ragas_context_recall_mean below threshold")

    return {
        "passed": not failures,
        "thresholds": {
            "faithfulness_mean_min": QUALITY_GATE_FAITHFULNESS_MIN,
            "completeness_mean_min": QUALITY_GATE_COMPLETENESS_MIN,
            "ragas_faithfulness_mean_min": QUALITY_GATE_RAGAS_FAITHFULNESS_MIN,
            "ragas_answer_relevance_mean_min": QUALITY_GATE_RAGAS_RELEVANCE_MIN,
            "ragas_context_recall_mean_min": QUALITY_GATE_RAGAS_CONTEXT_RECALL_MIN,
        },
        "failures": failures,
    }


def process_single_sample(idx: int, raw: dict[str, Any], current_model: str, client: OpenAI) -> dict[str, Any]:
    sample = normalize_sample(raw, idx)
    result = ask_model(sample["question"], current_model)
    
    contexts = extract_contexts(result)
    answer = str(result.get("answer", "")).strip()

    # Ghi log file Artifact
    artifact = {
        "question_id": sample["question_id"],
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "risk_level": sample["risk_level"],
        "answer": answer,
        "contexts": contexts,
        "sources": result.get("sources", []),
    }
    append_jsonl(ARTIFACTS_PATH, artifact)

    judge_output = judge_sample(
        client,
        question=sample["question"],
        ground_truth=sample["ground_truth"],
        answer=answer,
        contexts=contexts,
        risk_level=sample["risk_level"],
    )
    ragas_output = compute_ragas_metrics(
        client,
        question=sample["question"],
        ground_truth=sample["ground_truth"],
        answer=answer,
        contexts=contexts,
    )

    return {
        "question_id": sample["question_id"],
        "model_name": current_model,
        "judge_model": JUDGE_MODEL,
        "judge_output": judge_output,
        "ragas_output": ragas_output,
    }

def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for evaluation job.")

    reset_output_files()
    client = OpenAI()
    rows = load_testset(DATA_PATH)
    judge_records: list[dict[str, Any]] = []

    for idx, raw in enumerate(tqdm(rows, desc="Evaluating")):
        sample = normalize_sample(raw, idx)
        result = ask_model(sample["question"], EVAL_MODEL_NAME)
        contexts = extract_contexts(result)
        answer = str(result.get("answer", "")).strip()

        artifact = {
            "question_id": sample["question_id"],
            "question": sample["question"],
            "ground_truth": sample["ground_truth"],
            "risk_level": sample["risk_level"],
            "answer": answer,
            "contexts": contexts,
            "sources": result.get("sources", []),
        }
        append_jsonl(ARTIFACTS_PATH, artifact)

        judge_output = judge_sample(
            client,
            question=sample["question"],
            ground_truth=sample["ground_truth"],
            answer=answer,
            contexts=contexts,
            risk_level=sample["risk_level"],
        )
        ragas_output = compute_ragas_metrics(
            client,
            question=sample["question"],
            ground_truth=sample["ground_truth"],
            answer=answer,
            contexts=contexts,
        )

        judge_record = {
            "question_id": sample["question_id"],
            "model_name": EVAL_MODEL_NAME,
            "judge_model": JUDGE_MODEL,
            "judge_output": judge_output,
            "ragas_output": ragas_output,
        }
        judge_records.append(judge_record)
        append_jsonl(JUDGE_PATH, judge_record)

    summary = aggregate(judge_records)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nKẾT QUẢ TỔNG HỢP (SUMMARY):")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not summary["release_gate"]["passed"]:
        raise SystemExit("Release gate failed: " + "; ".join(summary["release_gate"]["failures"]))

if __name__ == "__main__":
    main()
