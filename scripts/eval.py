"""ANDi evaluation 的 CLI 入口。

此 script 負責建立 detector 並執行會寫出 CSV 的完整 volume evaluation。
metric 與 postprocess 邏輯刻意不放在這裡，而是放在 engine/evaluator.py
與 anomaly/postprocess.py。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

try:
    from _bootstrap import bootstrap
except ImportError:
    from andi_rewrite.scripts._bootstrap import bootstrap

bootstrap()

from andi_rewrite.runtime.builders import (
    build_detector_from_config,
    build_evaluator_from_config,
)
from andi_rewrite.utils.reporting import save_inference_report
from andi_rewrite.utils import load_config, print_config, summarize_components


def main() -> None:
    default_config = Path(__file__).resolve().parents[1] / "configs" / "eval.yaml"
    parser = argparse.ArgumentParser(description="Build the rewritten ANDi evaluation pipeline.")
    parser.add_argument("--config", default=str(default_config), help="Path to a YAML eval config.")
    parser.add_argument("--run-eval", action="store_true", help="Run full volume evaluation and write CSV files.")
    args = parser.parse_args()

    config = load_config(args.config)
    detector, accelerator = build_detector_from_config(config)
    print("Loaded eval config:")
    print_config(config)
    print("Built components:")
    print_config(summarize_components(detector=detector, diffusion=detector.diffusion, noise=detector.noise_plan))

    if args.run_eval:
        evaluator, dataloader = build_evaluator_from_config(config, detector, accelerator)
        eval_start = datetime.now().astimezone()
        result = evaluator.evaluate(dataloader)
        eval_end = datetime.now().astimezone()
        if result:
            print("Evaluation result:")
            print_config(result)
            if evaluator.is_main_process:
                save_inference_report(
                    config=config,
                    evaluator=evaluator,
                    result=result,
                    dataloader=dataloader,
                    start_time=eval_start,
                    end_time=eval_end,
                    config_path=args.config,
                    cli_args=vars(args),
                )
    else:
        print("Configuration loaded. Use --run-eval to run full volume evaluation.")


if __name__ == "__main__":
    main()
