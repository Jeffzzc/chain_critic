import asyncio
from typing import Dict, List

import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

BACKENDS = [f"http://127.0.0.1:{p}/v1" for p in range(8000, 8008)]
TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=600.0, pool=10.0)

app = FastAPI()
client = httpx.AsyncClient(timeout=TIMEOUT)

# 当前每个后端的进行中请求数
inflight: Dict[str, int] = {b: 0 for b in BACKENDS}
healthy: Dict[str, bool] = {b: True for b in BACKENDS}
lock = asyncio.Lock()


async def health_check_loop():
    while True:
        for backend in BACKENDS:
            try:
                r = await client.get(f"{backend}/models")
                healthy[backend] = (r.status_code == 200)
            except Exception:
                healthy[backend] = False
        await asyncio.sleep(5)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(health_check_loop())


async def choose_backend() -> str:
    async with lock:
        candidates: List[str] = [b for b in BACKENDS if healthy[b]]
        if not candidates:
            raise HTTPException(status_code=503, detail="No healthy backends")
        # least-inflight
        backend = min(candidates, key=lambda b: inflight[b])
        inflight[backend] += 1
        return backend


async def release_backend(backend: str):
    async with lock:
        inflight[backend] = max(0, inflight[backend] - 1)


@app.get("/healthz")
async def healthz():
    return {"healthy_backends": [b for b in BACKENDS if healthy[b]], "inflight": inflight}


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def proxy(path: str, request: Request):
    backend = await choose_backend()
    target_url = f"{backend}/{path}"

    try:
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)

        if request.method == "GET":
            resp = await client.get(target_url, headers=headers, params=request.query_params)
        else:
            resp = await client.post(target_url, headers=headers, content=body, params=request.query_params)

        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            return JSONResponse(status_code=resp.status_code, content=resp.json())

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=content_type or None,
        )
    finally:
        await release_backend(backend)