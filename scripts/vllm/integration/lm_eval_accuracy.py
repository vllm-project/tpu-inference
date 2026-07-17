# Copyright 2026 Google LLC
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
"""Run lm-eval with explicit ownership of the vLLM engine."""

import argparse
import json
from typing import Any


def _create_vllm_model(model_args: str | dict[str, Any], batch_size: str | int,
                       max_batch_size: int | None, device: str | None):
    from lm_eval.models.vllm_causallms import VLLM

    additional_config = {
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
        "device": device,
    }
    if isinstance(model_args, str):
        return VLLM.create_from_arg_string(model_args, additional_config)
    return VLLM.create_from_arg_obj(model_args, additional_config)


def evaluate_with_vllm(
    *,
    model_args: str | dict[str, Any],
    tasks: str | list[str],
    batch_size: str | int = "auto",
    max_batch_size: int | None = None,
    device: str | None = None,
    **evaluation_args: Any,
):
    """Evaluate with lm-eval and shut down its vLLM EngineCore on every exit."""
    import lm_eval

    model = _create_vllm_model(model_args, batch_size, max_batch_size, device)
    try:
        return lm_eval.simple_evaluate(
            model=model,
            tasks=tasks,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            device=device,
            **evaluation_args,
        )
    finally:
        # vLLM's synchronous LLM client relies on garbage collection to stop
        # EngineCore. Reference cycles can defer collection indefinitely and
        # leave the CI container alive after evaluation has completed.
        model.model.llm_engine.engine_core.shutdown()


def _parse_model_args(value: str) -> str | dict[str, Any]:
    if not value.lstrip().startswith("{"):
        return value
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--model_args JSON must be an object")
    return parsed


def main() -> None:
    from lm_eval.utils import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vllm"], default="vllm")
    parser.add_argument("--model_args", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--max_batch_size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num_fewshot", type=int)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--apply_chat_template", action="store_true")
    args = parser.parse_args()

    results = evaluate_with_vllm(
        model_args=_parse_model_args(args.model_args),
        tasks=args.tasks,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        apply_chat_template=args.apply_chat_template,
    )
    if results is None:
        return

    from lm_eval.utils import make_table

    print(make_table(results))
    if "groups" in results:
        print(make_table(results, "groups"))


if __name__ == "__main__":
    main()
