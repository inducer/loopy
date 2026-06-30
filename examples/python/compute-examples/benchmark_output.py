from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

H200_NVL_ROOFLINE = {
    "device": "NVIDIA H200 NVL",
    "source": "https://www.nvidia.com/en-us/data-center/h200/",
    "memory_bandwidth_gb_s": 4800.0,
    "fp64_peak_gflops": 30000.0,
    "fp32_peak_gflops": 60000.0,
}


def default_json_report_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "artifacts" / filename


def add_json_report_argument(parser, filename: str) -> None:
    parser.add_argument(
        "--json-report",
        nargs="?",
        default=None,
        const=str(default_json_report_path(filename)),
        help=(
            "Write a JSON benchmark report. If PATH is omitted, write to "
            f"{default_json_report_path(filename)}."
        ),
        metavar="PATH",
    )


def variant_result(
        name: str,
        role: str,
        *,
        time_s: float | None = None,
        flop_count: int | float | None = None,
        bytes_moved: int | float | None = None,
        dtype: str | None = None,
        relative_error: float | None = None,
        metadata: dict[str, Any] | None = None,
        error: str | None = None
    ) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "role": role,
    }

    if dtype is not None:
        result["dtype"] = dtype
    if time_s is not None:
        result["time_s"] = float(time_s)
    if flop_count is not None:
        result["flop_count"] = float(flop_count)
    if bytes_moved is not None:
        result["bytes_moved"] = float(bytes_moved)
    if relative_error is not None:
        result["relative_error"] = float(relative_error)
    if metadata:
        result["metadata"] = metadata
    if error is not None:
        result["error"] = error

    if time_s is not None and flop_count is not None and time_s > 0:
        result["throughput_gflops"] = float(flop_count) / float(time_s) / 1e9
    if bytes_moved is not None and flop_count is not None and bytes_moved > 0:
        result["arithmetic_intensity_flop_per_byte"] = (
            float(flop_count) / float(bytes_moved)
        )

    return result


def successful(variant: dict[str, Any]) -> bool:
    return (
        "error" not in variant
        and variant.get("time_s") is not None
        and float(variant["time_s"]) > 0
    )


def build_comparisons(
        variants: list[dict[str, Any]],
        baseline_name: str
    ) -> list[dict[str, Any]]:
    baseline = next(
        (variant for variant in variants if variant["name"] == baseline_name),
        None,
    )
    if baseline is None or not successful(baseline):
        return []

    baseline_time = float(baseline["time_s"])
    comparisons = []
    for variant in variants:
        if variant["name"] == baseline_name or not successful(variant):
            continue

        variant_time = float(variant["time_s"])
        comparisons.append({
            "baseline": baseline_name,
            "variant": variant["name"],
            "speedup": baseline_time / variant_time,
            "time_reduction_pct": (1 - variant_time / baseline_time) * 100,
            "baseline_time_s": baseline_time,
            "variant_time_s": variant_time,
        })

    return comparisons


def benchmark_report(
        *,
        example: str,
        description: str,
        parameters: dict[str, Any],
        baseline_name: str,
        variants: list[dict[str, Any]]
    ) -> dict[str, Any]:
    return to_jsonable({
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "example": example,
        "description": description,
        "parameters": parameters,
        "baseline": baseline_name,
        "variants": variants,
        "comparisons": build_comparisons(variants, baseline_name),
        "roofline": H200_NVL_ROOFLINE,
    })


def write_json_report(path: str | Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return

    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as outf:
        json.dump(to_jsonable(report), outf, indent=2, sort_keys=True)
        outf.write("\n")
    print(f"Wrote JSON report: {report_path}")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return to_jsonable(value.item())
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    return value
