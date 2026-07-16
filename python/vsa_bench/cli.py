from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import apply_overrides, atomic_json, load_config


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VSA-PathMoE TinyStories benchmark")
    p.add_argument("--profile", default="smoke")
    p.add_argument("--device")
    p.add_argument("--dtype")
    p.add_argument("--threads", type=int)
    sub = p.add_subparsers(dest="command", required=True)

    download = sub.add_parser("download")
    download.add_argument(
        "--mode", choices=["auto", "minimal", "deployment", "full"], default="auto"
    )

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--force", action="store_true")

    train = sub.add_parser("train")
    train.add_argument("--resume")

    custom_eval = sub.add_parser("evaluate-custom")
    custom_eval.add_argument("--checkpoint")
    official_eval = sub.add_parser("evaluate-official")
    official_eval.add_argument("--model-index", type=int, default=0)

    custom_bench = sub.add_parser("benchmark-custom")
    custom_bench.add_argument("--checkpoint")
    official_bench = sub.add_parser("benchmark-official")
    official_bench.add_argument("--model-index", type=int, default=0)

    show = sub.add_parser("show-config")
    show.add_argument("--output")
    sub.add_parser("doctor")
    dgx = sub.add_parser("doctor-dgx")
    dgx.add_argument("--strict", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    cfg = load_config(args.profile)
    apply_overrides(cfg, device=args.device, dtype=args.dtype, threads=args.threads)

    if args.command == "download":
        from .assets import download_assets

        _print(download_assets(cfg, mode=args.mode))
    elif args.command == "prepare":
        from .data import prepare_data

        _print(prepare_data(cfg, force=args.force))
    elif args.command == "train":
        from .train import train_model

        _print(train_model(cfg, resume=args.resume))
    elif args.command == "evaluate-custom":
        from .evaluate import evaluate_custom

        _print(evaluate_custom(cfg, checkpoint=args.checkpoint))
    elif args.command == "evaluate-official":
        from .evaluate import evaluate_official

        _print(evaluate_official(cfg, model_index=args.model_index))
    elif args.command == "benchmark-custom":
        from .benchmark import benchmark_custom

        _print(benchmark_custom(cfg, checkpoint=args.checkpoint))
    elif args.command == "benchmark-official":
        from .benchmark import benchmark_official

        _print(benchmark_official(cfg, model_index=args.model_index))
    elif args.command == "doctor":
        from .doctor import doctor

        _print(doctor(cfg))
    elif args.command == "doctor-dgx":
        from .dgx import dgx_spark_doctor

        result = dgx_spark_doctor(cfg)
        _print(result)
        if args.strict and not result["ready"]:
            raise SystemExit("DGX Spark preflight failed")
    elif args.command == "show-config":
        if args.output:
            atomic_json(Path(args.output), cfg)
        _print(cfg)
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
