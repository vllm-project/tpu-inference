# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class _ProxyLatencyTracker:
    """Small running latency summary for the proxy path."""

    def __init__(self, log_interval_s: float):
        self._log_interval_s = log_interval_s
        self._lock = threading.Lock()
        self._stats: dict[str, list[float]] = {}
        self._next_log = time.monotonic() + log_interval_s

    def record(self, phase: str, ms: float) -> None:
        if self._log_interval_s <= 0:
            return
        with self._lock:
            stat = self._stats.setdefault(phase, [0.0, 0.0, ms, ms])
            stat[0] += 1.0
            stat[1] += ms
            stat[2] = min(stat[2], ms)
            stat[3] = max(stat[3], ms)

            now = time.monotonic()
            if now < self._next_log:
                return
            self._next_log = now + self._log_interval_s
            parts = []
            for key in sorted(self._stats):
                n, total, min_ms, max_ms = self._stats[key]
                parts.append(
                    f"{key}: n={int(n)} avg={total / n:.2f}ms "
                    f"min={min_ms:.2f}ms max={max_ms:.2f}ms")
            msg = f"PERF PROXY latency summary | {' | '.join(parts)}"
            print(msg, flush=True)
            logger.info(msg)


def _proxy_latency_interval() -> float:
    val = os.getenv("PROXY_LATENCY_LOG_INTERVAL", "30")
    try:
        return float(val)
    except ValueError:
        return 30.0


_PROXY_LATENCY = _ProxyLatencyTracker(_proxy_latency_interval())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []

    # Create prefill clients
    for i, (host, port) in enumerate(global_args.prefiller_instances):
        prefiller_base_url = f'http://{host}:{port}'
        app.state.prefill_clients.append({
            'client':
            httpx.AsyncClient(timeout=None, base_url=prefiller_base_url),
            'host':
            host,
            'port':
            port,
            'id':
            i
        })

    # Create decode clients
    for i, (host, port) in enumerate(global_args.decoder_instances):
        decoder_base_url = f'http://{host}:{port}'
        app.state.decode_clients.append({
            'client':
            httpx.AsyncClient(timeout=None, base_url=decoder_base_url),
            'host':
            host,
            'port':
            port,
            'id':
            i
        })

    # Initialize round-robin iterators
    app.state.prefill_iterator = itertools.cycle(
        range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(
        range(len(app.state.decode_clients)))

    print(f"Initialized {len(app.state.prefill_clients)} prefill clients "
          f"and {len(app.state.decode_clients)} decode clients.")

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info['client'].aclose()

    for client_info in app.state.decode_clients:
        await client_info['client'].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")

    # For prefiller instances
    parser.add_argument("--prefiller-hosts",
                        "--prefiller-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--prefiller-ports",
                        "--prefiller-port",
                        type=int,
                        nargs="+",
                        default=[8400])

    # For decoder instances
    parser.add_argument("--decoder-hosts",
                        "--decoder-host",
                        type=str,
                        nargs="+",
                        default=["localhost"])
    parser.add_argument("--decoder-ports",
                        "--decoder-port",
                        type=int,
                        nargs="+",
                        default=[9400])

    args = parser.parse_args()

    # Validate and pair hosts with ports
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports")

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError(
            "Number of decoder hosts must match number of decoder ports")

    # Create tuples of (host, port) for each service type
    args.prefiller_instances = list(
        zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))

    return args


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == 'prefill':
        client_idx = next(app.state.prefill_iterator)
        return app.state.prefill_clients[client_idx]
    elif service_type == 'decode':
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


async def send_request_to_prefill(client_info: dict, endpoint: str,
                                  req_data: dict, request_id: str):
    """
    Send a request to a service using a client from the pool.
    """
    req_data = req_data.copy()
    # Must overwrite these for prefill workers.
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    response = await client_info['client'].post(endpoint,
                                                json=req_data,
                                                headers=headers)
    response.raise_for_status()

    return response


async def stream_from_decode(client_info: dict, endpoint: str, req_data: dict,
                             request_id: str):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id
    }

    async with client_info['client'].stream("POST",
                                            endpoint,
                                            json=req_data,
                                            headers=headers) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api: str, request: Request):
    try:
        t_request_recv_perf = time.perf_counter()
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        print(f"PERF PROXY req_received req_id={request_id} ts={time.time():.6f}",
              flush=True)

        # Get the next prefill client in round-robin fashion
        prefill_client_info = get_next_client(request.app, 'prefill')

        # Send request to prefill service
        t_prefill_send = time.time()
        t_prefill_send_perf = time.perf_counter()
        print(f"PERF PROXY prefill_send req_id={request_id} ts={t_prefill_send:.6f}",
              flush=True)
        response = await send_request_to_prefill(prefill_client_info, api,
                                                 req_data, request_id)
        t_prefill_recv = time.time()
        t_prefill_recv_perf = time.perf_counter()
        prefill_ms = (t_prefill_recv_perf - t_prefill_send_perf) * 1000.0
        _PROXY_LATENCY.record("prefill", prefill_ms)
        _PROXY_LATENCY.record(
            "prefill_from_req",
            (t_prefill_recv_perf - t_request_recv_perf) * 1000.0)

        # Extract the needed fields
        response_json = response.json()
        kv_transfer_params = response_json.get('kv_transfer_params', {})
        kv_uuid = kv_transfer_params.get("uuid") if kv_transfer_params else None
        kv_uuid_log = f" kv_uuid={kv_uuid}" if kv_uuid is not None else ""
        print(f"PERF PROXY prefill_recv req_id={request_id} ts={t_prefill_recv:.6f} "
              f"dur_ms={prefill_ms:.2f}{kv_uuid_log}",
              flush=True)

        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params

        # Get the next decode client in round-robin fashion
        decode_client_info = get_next_client(request.app, 'decode')

        logger.debug("Using %s %s", prefill_client_info, decode_client_info)

        # Stream response from decode service
        async def generate_stream():
            t_decode_send = time.time()
            t_decode_send_perf = time.perf_counter()
            print(f"PERF PROXY decode_send req_id={request_id} "
                  f"ts={t_decode_send:.6f}{kv_uuid_log}", flush=True)
            first = True
            async for chunk in stream_from_decode(decode_client_info,
                                                  api,
                                                  req_data,
                                                  request_id=request_id):
                if first:
                    t_first = time.time()
                    t_first_perf = time.perf_counter()
                    decode_first_ms = (
                        t_first_perf - t_decode_send_perf) * 1000.0
                    ttft_ms = (
                        t_first_perf - t_request_recv_perf) * 1000.0
                    _PROXY_LATENCY.record("decode_first", decode_first_ms)
                    _PROXY_LATENCY.record("ttft", ttft_ms)
                    print(f"PERF PROXY first_chunk req_id={request_id} "
                          f"ts={t_first:.6f} "
                          f"dur_from_decode_send_ms={decode_first_ms:.2f} "
                          f"ttft_ms={ttft_ms:.2f}{kv_uuid_log}",
                          flush=True)
                    first = False
                yield chunk

        return StreamingResponse(generate_stream(),
                                 media_type="application/json")

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
              f" - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    """Simple endpoint to check if the server is running."""
    return {
        "status": "ok",
        "prefill_instances": len(app.state.prefill_clients),
        "decode_instances": len(app.state.decode_clients)
    }


if __name__ == '__main__':
    global global_args
    global_args = parse_args()

    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)
