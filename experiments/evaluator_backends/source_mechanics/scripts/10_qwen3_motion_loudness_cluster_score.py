from __future__ import annotations

"""Judge two visual action clusters with the released motion-only Qwen policy."""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


def configure_triton_ptxas() -> None:
    """Use the venv-bundled ptxas when it is available."""
    python_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    configured = os.environ.get("ACOUSTITRACE_TRITON_PTXAS_PATH", "").strip()
    candidate = Path(configured) if configured else (
        Path(sys.prefix)
        / "lib"
        / python_tag
        / "site-packages"
        / "triton"
        / "backends"
        / "nvidia"
        / "bin"
        / "ptxas"
    )
    if candidate.is_file():
        os.environ["TRITON_PTXAS_PATH"] = str(candidate.resolve())


configure_triton_ptxas()
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "你是一个严格的视频物理动作强度评估助手。你只根据视觉画面判断动作强弱，"
    "不能使用音频、文件名、prompt 先验或任何外部信息。你必须只输出 JSON。"
)

USER_PROMPT = (
    "下面给出同一个生成视频中切出的两个视觉片段：视频1对应 cluster_1，视频2对应 cluster_2。\n"
    "请判断哪个 cluster 的可见动作强度更高。\n\n"
    "判断标准：\n"
    "1. 只比较动作本身的可见运动幅度、速度、运动范围、持续性和清晰度。\n"
    "2. 手、身体、工具或主体的可见运动越大、越快、越清楚，视觉动作强度越高。\n"
    "3. 不要使用两个物体是否发生物理交互、目标物状态变化或结果性画面证据作为判断依据。\n"
    "4. reasoning 只能描述动作幅度、速度、运动范围、持续性和清晰度，不要描述结果性证据。\n"
    "5. 如果两个片段动作强弱无法可靠判断，输出 unclear。\n"
    "6. 如果两个片段视觉强度非常接近，输出 tie。\n\n"
    "严格 JSON 输出，格式如下：\n"
    "{\n"
    "  \"visual_stronger_cluster\": \"cluster_1\",\n"
    "  \"confidence\": 0.85,\n"
    "  \"reasoning\": \"简短说明视觉证据\"\n"
    "}\n"
    "visual_stronger_cluster 只能是 cluster_1、cluster_2、tie、unclear。"
)

RETRY_USER_SUFFIX = (
    "\n\n重新回答。上一版说明使用了本任务禁止的结果性证据。"
    "请只依据动作幅度、速度、运动范围、持续性和清晰度判断，"
    "reasoning 中不要出现任何物理交互或结果性画面证据。"
)

CONTACT_REASONING_BANNED_TERMS = [
    "接触", "击打", "击中", "冲击", "空挥", "被击打", "震动", "振动",
    "变形", "碎屑", "弹跳", "hit", "strike", "impact", "contact",
    "reaction", "deform", "vibrate",
]


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text.strip()


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_message(cluster_1_clip: str, cluster_2_clip: str, retry: bool = False) -> list[dict[str, Any]]:
    prompt = USER_PROMPT + (RETRY_USER_SUFFIX if retry else "")
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "视频1（cluster_1）："},
                {"type": "video", "video": f"file://{cluster_1_clip}"},
                {"type": "text", "text": "视频2（cluster_2）："},
                {"type": "video", "video": f"file://{cluster_2_clip}"},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def prepare_inputs(messages: list[dict[str, Any]], processor: AutoProcessor) -> dict[str, Any]:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data: dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def parse_json_object(text: str) -> dict[str, Any]:
    raw = clean(text)
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


def parse_stronger(raw_text: str) -> tuple[str, float, str, str]:
    obj = parse_json_object(raw_text)
    stronger = clean(obj.get("visual_stronger_cluster")).lower()
    reasoning = clean(obj.get("reasoning"))
    confidence = as_float(obj.get("confidence"))
    stronger = stronger.replace("-", "_").replace(" ", "_")
    if stronger in {"1", "video_1", "视频1", "cluster1"}:
        stronger = "cluster_1"
    if stronger in {"2", "video_2", "视频2", "cluster2"}:
        stronger = "cluster_2"
    if stronger not in {"cluster_1", "cluster_2", "tie", "unclear"}:
        low = raw_text.lower()
        if "cluster_1" in low or "video 1" in low or "视频1" in raw_text:
            stronger = "cluster_1"
        elif "cluster_2" in low or "video 2" in low or "视频2" in raw_text:
            stronger = "cluster_2"
        else:
            stronger = "unparseable"
    status = "valid" if stronger in {"cluster_1", "cluster_2"} else stronger
    return stronger, confidence, reasoning, status


def contact_reasoning_terms(reasoning: str) -> list[str]:
    low = clean(reasoning).lower()
    return [term for term in CONTACT_REASONING_BANNED_TERMS if term.lower() in low]


def run_qwen_pair(
    cluster_1_clip: str,
    cluster_2_clip: str,
    processor: AutoProcessor,
    llm: LLM,
    sampling_params: SamplingParams,
    *,
    retry: bool = False,
) -> tuple[str, str, float, str, str]:
    inputs = [prepare_inputs(build_message(cluster_1_clip, cluster_2_clip, retry), processor)]
    outputs = llm.generate(inputs, sampling_params, use_tqdm=False)
    raw_text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    stronger, confidence, reasoning, status = parse_stronger(raw_text)
    return raw_text, stronger, confidence, reasoning, status


def compare_audio(row: dict[str, Any], stronger: str, metric: str) -> dict[str, Any]:
    c1 = as_float(row.get(f"cluster_1_{metric}"))
    c2 = as_float(row.get(f"cluster_2_{metric}"))
    prefix = f"qwen_dynamic_pairwise_{metric}"
    if stronger not in {"cluster_1", "cluster_2"} or not math.isfinite(c1) or not math.isfinite(c2):
        return {
            f"qwen_stronger_minus_weaker_{metric}": float("nan"),
            f"{prefix}_valid": False,
            f"{prefix}_no_margin": "",
            f"{prefix}_margin_1db": "",
            f"{prefix}_margin_3db": "",
        }
    stronger_val = c1 if stronger == "cluster_1" else c2
    weaker_val = c2 if stronger == "cluster_1" else c1
    margin = stronger_val - weaker_val
    return {
        f"qwen_stronger_minus_weaker_{metric}": float(margin),
        f"{prefix}_valid": True,
        f"{prefix}_no_margin": bool(margin > 0.0),
        f"{prefix}_margin_1db": bool(margin >= 1.0),
        f"{prefix}_margin_3db": bool(margin >= 3.0),
    }


def bool_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows if isinstance(row.get(field), bool)]
    return float(sum(values) / len(values)) if values else None


def numeric_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [as_float(row.get(field)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return float(sum(values) / len(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=32768)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.pairs_jsonl)
    if args.limit > 0:
        rows = rows[: args.limit]
    results_jsonl = args.output_dir / "qwen_cluster_visual_strength_results.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if args.skip_existing and results_jsonl.exists():
        for row in read_jsonl(results_jsonl):
            sample_id = clean(row.get("generated_sample_id"))
            if sample_id:
                existing[sample_id] = row

    pending = [row for row in rows if clean(row.get("generated_sample_id")) not in existing]
    processor = llm = sampling_params = None
    if pending:
        processor = AutoProcessor.from_pretrained(str(args.checkpoint_path))
        llm = LLM(
            model=str(args.checkpoint_path),
            mm_encoder_tp_mode="data",
            tensor_parallel_size=max(1, args.tensor_parallel_size),
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=0,
        )
        sampling_params = SamplingParams(
            temperature=args.temperature, max_tokens=args.max_tokens, top_k=-1
        )

    output_rows: list[dict[str, Any]] = []
    with results_jsonl.open("w", encoding="utf-8") as out_f:
        for index, row in enumerate(rows):
            sample_id = clean(row.get("generated_sample_id"))
            if sample_id in existing:
                result = existing[sample_id]
            else:
                c1 = clean(row.get("cluster_1_clip_path"))
                c2 = clean(row.get("cluster_2_clip_path"))
                result = dict(row)
                if not Path(c1).exists() or not Path(c2).exists():
                    result.update(
                        {
                            "qwen_visual_judgment_status": "missing_cluster_clip",
                            "qwen_visual_stronger_cluster": "unverifiable",
                            "qwen_confidence": float("nan"),
                            "qwen_reasoning": "",
                            "qwen_raw_output": "",
                            "qwen_retry_count": 0,
                            "qwen_contact_reasoning_violation": False,
                            "qwen_contact_reasoning_terms": "",
                        }
                    )
                else:
                    raw_text, stronger, confidence, reasoning, status = run_qwen_pair(
                        c1, c2, processor, llm, sampling_params, retry=False
                    )
                    retry_count = 0
                    banned_terms = contact_reasoning_terms(reasoning)
                    if banned_terms:
                        retry_count = 1
                        raw_text, stronger, confidence, reasoning, status = run_qwen_pair(
                            c1, c2, processor, llm, sampling_params, retry=True
                        )
                        banned_terms = contact_reasoning_terms(reasoning)
                    violation = bool(banned_terms)
                    if violation:
                        status = "invalid_contact_reasoning"
                        stronger = "unverifiable"
                    result.update(
                        {
                            "qwen_visual_judgment_status": status,
                            "qwen_visual_stronger_cluster": stronger,
                            "qwen_confidence": confidence,
                            "qwen_reasoning": reasoning,
                            "qwen_raw_output": raw_text,
                            "qwen_retry_count": retry_count,
                            "qwen_contact_reasoning_violation": violation,
                            "qwen_contact_reasoning_terms": ";".join(banned_terms),
                        }
                    )
                expected = clean(result.get("expected_stronger_cluster"))
                stronger = clean(result.get("qwen_visual_stronger_cluster"))
                result["qwen_agrees_with_prompt_prior"] = (
                    bool(stronger == expected)
                    if stronger in {"cluster_1", "cluster_2"} and expected
                    else ""
                )
                for metric in ("max_rms_db", "energy_sum_db", "mean_rms_db", "top2_mean_rms_db"):
                    result.update(compare_audio(result, stronger, metric))

            output_rows.append(result)
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            print(
                json.dumps(
                    {
                        "idx": index,
                        "sample_id": sample_id,
                        "status": result.get("qwen_visual_judgment_status"),
                        "qwen_visual_stronger_cluster": result.get("qwen_visual_stronger_cluster"),
                        "margin_max_rms_db": result.get("qwen_stronger_minus_weaker_max_rms_db"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in output_rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    write_csv(args.output_dir / "qwen_cluster_visual_strength_metrics.csv", output_rows, fieldnames)

    valid = [row for row in output_rows if row.get("qwen_visual_judgment_status") == "valid"]
    statuses = sorted({clean(row.get("qwen_visual_judgment_status")) for row in output_rows})
    score = bool_mean(valid, "qwen_dynamic_pairwise_max_rms_db_no_margin")
    summary = {
        "pairs_jsonl": str(args.pairs_jsonl),
        "checkpoint_path": str(args.checkpoint_path),
        "input_pairs": len(rows),
        "qwen_valid_count": len(valid),
        "qwen_invalid_count": len(output_rows) - len(valid),
        "invalid_contact_reasoning_count": sum(
            row.get("qwen_visual_judgment_status") == "invalid_contact_reasoning"
            for row in output_rows
        ),
        "qwen_retry_count": sum(int(row.get("qwen_retry_count") or 0) for row in output_rows),
        "status_counts": {
            status: sum(row.get("qwen_visual_judgment_status") == status for row in output_rows)
            for status in statuses
        },
        "qwen_prompt_prior_agreement_rate": bool_mean(valid, "qwen_agrees_with_prompt_prior"),
        "qwen_dynamic_pairwise_acc_no_margin_max_rms_db": score,
        "qwen_dynamic_pairwise_acc_margin_1db_max_rms_db": bool_mean(
            valid, "qwen_dynamic_pairwise_max_rms_db_margin_1db"
        ),
        "qwen_dynamic_pairwise_acc_margin_3db_max_rms_db": bool_mean(
            valid, "qwen_dynamic_pairwise_max_rms_db_margin_3db"
        ),
        "qwen_dynamic_pairwise_acc_no_margin_energy_sum_db": bool_mean(
            valid, "qwen_dynamic_pairwise_energy_sum_db_no_margin"
        ),
        "mean_qwen_stronger_minus_weaker_max_rms_db": numeric_mean(
            valid, "qwen_stronger_minus_weaker_max_rms_db"
        ),
        "physical_score": 100.0 * score if score is not None else None,
        "score_formula": (
            "100*mean(qwen_dynamic_pairwise_max_rms_db_no_margin) "
            "over valid Qwen visual-strength judgments"
        ),
        "outputs": {
            "metrics_csv": str(args.output_dir / "qwen_cluster_visual_strength_metrics.csv"),
            "results_jsonl": str(results_jsonl),
            "summary_json": str(args.output_dir / "qwen_cluster_visual_strength_summary.json"),
        },
    }
    (args.output_dir / "qwen_cluster_visual_strength_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
