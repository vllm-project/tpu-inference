# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import os
import time
from concurrent import futures

import vllm.envs as envs
from google.cloud import pubsub_v1, storage
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser

from tpu_commons.core import disagg_utils


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct")
    parser.set_defaults(max_model_len=1024)

    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int, default=128)
    sampling_group.add_argument("--temperature", type=float, default=0.7)
    sampling_group.add_argument("--top-p", type=float, default=0.9)
    sampling_group.add_argument("--top-k", type=int, default=50)

    # Add application-specific arguments
    app_group = parser.add_argument_group("Application parameters")
    app_group.add_argument(
        "--project-id",
        type=str,
        help="Google Cloud project ID (required for pubsub mode).",
    )
    app_group.add_argument(
        "--subscription-id",
        type=str,
        help="Pub/Sub subscription ID (required for pubsub mode).",
    )
    app_group.add_argument(
        "--bucket-name",
        type=str,
        help="GCS bucket name to write results to (required for pubsub mode).",
    )
    app_group.add_argument(
        "--blob-name-prefix",
        type=str,
        default="llm-results/",
        help="Prefix for the GCS blob name.",
    )
    return parser


def run_pubsub_inference(args: dict, llm: LLM, sampling_params: SamplingParams):
    """Listens to Pub/Sub, runs batched LLM inference, and writes to GCS."""
    if not all([args.get("project_id"), args.get("subscription_id"), args.get("bucket_name")]):
        raise ValueError(
            "For 'pubsub' mode, --project-id, --subscription-id, and --bucket-name must be provided."
        )

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(
        args["project_id"], args["subscription_id"]
    )

    storage_client = storage.Client(project=args["project_id"])

    def message_callback(message: pubsub_v1.subscriber.message.Message):
        """Callback function to process Pub/Sub messages."""
        try:
            data = json.loads(message.data)
            prompts = data.get("prompts")
            output_gcs_bucket=data.get("output_gcs_bucket")
            output_gcs_blob=data.get("output_gcs_blob")
            if not isinstance(prompts, list):
                logging.error("'prompts' key not found or not a list in message.")
                message.nack()
                return

            logging.info(f"Received {len(prompts)} prompts. Running inference...")
            outputs = llm.generate(prompts, sampling_params)
            logging.info("Inference complete.")

            results = [
                {"prompt": out.prompt, "generated_text": out.outputs[0].text}
                for out in outputs
            ]

            # Write results to GCS
            bucket = storage_client.bucket(output_gcs_bucket)
            blob = bucket.blob(output_gcs_blob)
            blob.upload_from_string(
                json.dumps(results, indent=2),
                content_type="application/json",
            )
            logging.info(f"Results written to gs://{args['bucket_name']}/{blob_name}")

            message.ack()
        except Exception as e:
            logging.exception(f"Error processing message: {e}")
            message.nack()

    streaming_pull_future = subscriber.subscribe(
        subscription_path, callback=message_callback
    )
    logging.info(f"Listening for messages on {subscription_path}...")

    try:
        streaming_pull_future.result()
    except (futures.TimeoutError, KeyboardInterrupt):
        streaming_pull_future.cancel()
        streaming_pull_future.result()
        subscriber.close()
        logging.info("Shutting down.")


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    # Create an LLM
    llm = LLM(**args)

    # Create a sampling params object
    # Create a sampling params object
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    run_pubsub_inference(args, llm, sampling_params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Skip long warmup for local simple test.
    os.environ["SKIP_JAX_PRECOMPILE"] = "1"

    parser = create_parser()
    parsed_args = vars(parser.parse_args())

    if not disagg_utils.is_disagg_enabled():
        main(parsed_args)
    else:
        from unittest.mock import patch

        from tpu_commons.core.core_tpu import DisaggEngineCoreProc

        with patch("vllm.v1.engine.core.EngineCoreProc", DisaggEngineCoreProc):
            main(parsed_args)
