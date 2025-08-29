# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import os
import time
import sys
import threading
import subprocess
from concurrent import futures

import vllm.envs as envs
from google.cloud import pubsub_v1, storage
import google.auth
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser
from openai import OpenAI
from openai import AsyncOpenAI

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
    app_group.add_argument(
        "--max-pubsub-workers",
        type=int,
        default=1,
        help="Maximum number of workers for the Pub/Sub subscriber.",
    )
    app_group.add_argument(
        "--use-openai-server",
        action="store_true",
        help="Whether to use OpenAI server.",
    )
    return parser

def get_current_service_account_email():
    """Retrieves the email address of the current service account."""
    try:
        credentials, project = google.auth.default()
        logging.error(credentials)
        logging.error(project)
        if hasattr(credentials, 'service_account_email'):
            return credentials.service_account_email
        else:
            return "No service account attached or credential type is not a service account."
    except Exception as e:
        return f"An error occurred: {e}"

def run_pubsub_inference(args: dict, llm: LLM, sampling_params: SamplingParams, server: OpenAIModelServer):
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
            subscriber.modify_ack_deadline(
                request={
                    "subscription": subscription_path,
                    "ack_ids": [message.ack_id],
                    # Must be between 10 and 600.
                    "ack_deadline_seconds": 600,
                }
            )
            prompts = data.get("prompts")
            output_gcs_bucket=data.get("output_gcs_bucket")
            output_gcs_blob=data.get("output_gcs_blob")
            if not isinstance(prompts, list):
                logging.error("'prompts' key not found or not a list in message.")
                message.nack()
                return

            logging.info(f"Received {len(prompts)} prompts. Running inference...")
            if llm:
                outputs = llm.generate(prompts, sampling_params)
            else:
                with getAsyncVLLMClient(server.get_server_port()) as client:
                    try:
                        async_predictions = [
                            client.completions.create(
                                model="model", prompt=prompt, **inference_args)
                            for prompt in prompts
                        ]
                        responses = asyncio.gather(*async_predictions)
                    except Exception as e:
                        model.check_connectivity()
                        raise e
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
            logging.info(f"Results written to gs://{output_gcs_bucket}/{output_gcs_blob}")

            message.ack()
        except json.decoder.JSONDecodeError as e:
            errorMsg = f"Error parsing message: {e}"
            logging.error(errorMsg)
            results = [
                {"error" : errorMsg}
            ]
            bucket = storage_client.bucket(output_gcs_bucket)
            blob = bucket.blob(output_gcs_blob)
            blob.upload_from_string(
                json.dumps(results, indent=2),
                content_type="application/json",
            )
            message.ack()
        except Exception as e:
            logging.exception(f"Error processing message: {e}")
            message.nack()

    # Limit the concurrency to 1. Otherwise vLLM Ray engine crashes.
    executor = futures.ThreadPoolExecutor(max_workers=args["max_pubsub_workers"])
    # A thread pool-based scheduler. It must not be shared across SubscriberClients.
    scheduler = pubsub_v1.subscriber.scheduler.ThreadScheduler(executor)
    streaming_pull_future = subscriber.subscribe(
        subscription_path, callback=message_callback, scheduler=scheduler
    )
    logging.info(f"Listening for messages on {subscription_path}...")

    try:
        streaming_pull_future.result()
    except (futures.TimeoutError, KeyboardInterrupt):
        streaming_pull_future.cancel()
        streaming_pull_future.result()
        subscriber.close()
        logging.info("Shutting down.")

def getVLLMClient(port) -> OpenAI:
  openai_api_key = "EMPTY"
  openai_api_base = f"http://localhost:{port}/v1"
  return OpenAI(
      api_key=openai_api_key,
      base_url=openai_api_base,
  )

def getAsyncVLLMClient(port) -> AsyncOpenAI:
  openai_api_key = "EMPTY"
  openai_api_base = f"http://localhost:{port}/v1"
  return AsyncOpenAI(
      api_key=openai_api_key,
      base_url=openai_api_base,
  )

def start_process(cmd) -> tuple[subprocess.Popen, int]:
  logging.error("Starting service with %s", str(cmd).replace("',", "'"))
  process = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  # Emit the output of this command as info level logging.
  def log_stdout():
    line = process.stdout.readline()
    while line:
      # The log obtained from stdout is bytes, decode it into string.
      # Remove newline via rstrip() to not print an empty line.
      logging.error(line.decode(errors='backslashreplace').rstrip())
      line = process.stdout.readline()

  t = threading.Thread(target=log_stdout)
  t.daemon = True
  t.start()
  return process

class OpenAIModelServer():
  def __init__(self, vllm_server_kwargs: dict[str, str]):
    self._vllm_server_kwargs = vllm_server_kwargs
    self._server_started = False
    self._server_process = None
    self._server_port: int = -1
    self._server_process_lock = threading.RLock()

    self.start_server()

  def start_server(self, retries=3):
    with self._server_process_lock:
      if not self._server_started:
        self._server_port = 1537
        server_cmd = [
            sys.executable,
            '-m',
            'vllm.entrypoints.openai.api_server',
            '--port',
            str(self._server_port),
        ]
        for k, v in self._vllm_server_kwargs.items():
          server_cmd.append(f'--{k}')
          # Only add values for commands with value part.
          if v is not None:
            server_cmd.append(str(v))
        self._server_process = start_process(server_cmd)

      self.check_connectivity(retries)

  def get_server_port(self) -> int:
    if not self._server_started:
      self.start_server()
    return self._server_port

  def check_connectivity(self, retries=3):
    with getVLLMClient(self._server_port) as client:
      while self._server_process.poll() is None:
        try:
          models = client.models.list().data
          logging.error('models: %s' % models)
          if len(models) > 0:
            self._server_started = True
            return
        except:  # pylint: disable=bare-except
          pass
        # Sleep while bringing up the process
        time.sleep(5)

      if retries == 0:
        self._server_started = False
        raise Exception(
            "Failed to start vLLM server, polling process exited with code " +
            "%s.  Next time a request is tried, the server will be restarted" %
            self._server_process.poll())
      else:
        self.start_server(retries - 1)


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    infra_args= {
        'project_id': args.pop("project_id"),
        'subscription_id': args.pop('subscription_id'),
        'bucket_name': args.pop('bucket_name'),
        'blob_name_prefix': args.pop('blob_name_prefix'),
        'max_pubsub_workers': args.pop('max_pubsub_workers'),
        'use_openai_server' : True if args.pop('use_openai_server') else False,
    }
    openai_args = {
       'model' : args['model'],
       'tensor_parallel_size': args['tensor_parallel_size'],
       'max_model_len': args['max_model_len'],
    }

    logging.error(f"Current SA email: {get_current_service_account_email()}")

    # Create an LLM
    if infra_args['use_openai_server']:
        llm = LLM(**args)
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
        run_pubsub_inference(infra_args, llm, sampling_params, None)
    else:
        openai = OpenAIModelServer(openai_args)
        run_pubsub_inference(infra_args, None, None, openai)


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
