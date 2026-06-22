#!/usr/bin/env python3
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
"""Minimal round-robin HTTP reverse proxy across a fixed set of backend ports.
Stands in for an external LB (nginx/haproxy) in front of vLLM's
--data-parallel-external-lb child servers, which don't include one.

See b/525257493 for more context.
"""

import itertools
import os
import sys

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

BACKEND_BASE_PORT = int(os.environ.get("BACKEND_BASE_PORT", "8000"))
BACKEND_COUNT = int(os.environ.get("BACKEND_COUNT", "4"))
BACKENDS = [
    f"http://127.0.0.1:{BACKEND_BASE_PORT + i}" for i in range(BACKEND_COUNT)
]
_cycle = itertools.cycle(BACKENDS)
_session: ClientSession | None = None


async def round_robin(request: web.Request) -> web.StreamResponse:
    backend = next(_cycle)
    url = backend + request.path_qs
    body = await request.read()
    assert _session is not None
    async with _session.request(request.method,
                                url,
                                headers=request.headers.copy(),
                                data=body) as resp:
        stream_resp = web.StreamResponse(status=resp.status,
                                         headers=resp.headers.copy())
        await stream_resp.prepare(request)
        async for chunk in resp.content.iter_any():
            await stream_resp.write(chunk)
        await stream_resp.write_eof()
        return stream_resp


async def on_startup(app: web.Application) -> None:
    global _session
    connector = TCPConnector(limit=0, limit_per_host=0)
    _session = ClientSession(timeout=ClientTimeout(total=None),
                             connector=connector)


async def on_cleanup(app: web.Application) -> None:
    if _session is not None:
        await _session.close()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    # Default aiohttp client_max_size is 1MB; multimodal (image) request
    # bodies can exceed that and get reset mid-benchmark. Disable the cap --
    # this proxy only ever forwards trusted local benchmark traffic.
    app = web.Application(client_max_size=0)
    app.router.add_route("*", "/{path:.*}", round_robin)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    # Default backlog (128) is far too small for thousands of concurrent
    # benchmark connections; the kernel clamps this to net.core.somaxconn
    # anyway, so request more than we'd ever expect to need.
    web.run_app(app, host="0.0.0.0", port=port, print=None, backlog=10000)


if __name__ == "__main__":
    main()
