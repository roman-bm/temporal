"""FastAPI app: model roster, run a council, stream it live over SSE."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from hmac import compare_digest
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .orchestrator import Council, CouncilConfig
from .registry import Registry
from .report import to_markdown
from .session import Session, SessionStore

STATIC_DIR = Path(__file__).with_name("static")

registry: Registry
store = SessionStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global registry
    load_dotenv()
    registry = Registry(
        config_path=os.environ.get("ORCHESTRA_MODELS"),
        force_simulation=os.environ.get("ORCHESTRA_SIMULATE") == "1",
        max_concurrency=int(os.environ.get("ORCHESTRA_CONCURRENCY", "8")),
    )
    yield
    await registry.aclose()


app = FastAPI(title="Multi-Model Orchestration Council", lifespan=lifespan)


@app.middleware("http")
async def require_token(request: Request, call_next):
    """Gate the API behind a shared token when one is configured.

    Binding to 0.0.0.0 so a phone can reach the UI also exposes it to everyone
    else on the network — and `POST /api/sessions` spends real money against
    whatever provider keys are loaded. Set ORCHESTRA_TOKEN and the API refuses
    anything without it. The static shell stays open because it can do nothing
    on its own.

    The token is accepted as a `?t=` query parameter as well as a header, since
    typing a URL is the only practical way to hand a phone a credential.
    """
    token = os.environ.get("ORCHESTRA_TOKEN")
    if token and request.url.path.startswith("/api/"):
        supplied = request.headers.get("x-orchestra-token") or request.query_params.get("t")
        if not compare_digest(supplied or "", token):
            return JSONResponse({"detail": "invalid or missing token"}, status_code=401)
    return await call_next(request)


class RunRequest(BaseModel):
    task: str = Field(min_length=1)
    context: str = ""
    orchestrator: str
    panel: list[str] = Field(default_factory=list)
    rounds: int = 2
    convergence_target: float = 0.78
    include_orchestrator_in_panel: bool = False


# ----------------------------------------------------------------------
# API
# ----------------------------------------------------------------------

@app.get("/api/models")
async def list_models() -> dict[str, Any]:
    models = registry.status()
    return {
        "models": models,
        "orchestrators": [m["key"] for m in models if m["can_orchestrate"]],
        "live_count": sum(1 for m in models if m["live"]),
        "simulation_forced": registry.force_simulation,
    }


@app.get("/api/sessions")
async def list_sessions() -> dict[str, Any]:
    return {"sessions": store.recent()}


@app.post("/api/sessions")
async def create_session(req: RunRequest) -> dict[str, Any]:
    panel = req.panel or [
        m.key for m in registry.all() if m.key != req.orchestrator
    ][:8]
    try:
        config = CouncilConfig(
            orchestrator=req.orchestrator,
            panel=panel,
            rounds=req.rounds,
            convergence_target=req.convergence_target,
            include_orchestrator_in_panel=req.include_orchestrator_in_panel,
        )
        council = Council(registry, config)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session = store.create(req.task, req.context, config.__dict__ | {"panel": panel})
    asyncio.create_task(_run(session, council, req))
    return {"session_id": session.id}


async def _run(session: Session, council: Council, req: RunRequest) -> None:
    try:
        report = await council.run(req.task, req.context, emit=session.emit)
        session.report = report
    except Exception as exc:  # noqa: BLE001 - surface, never swallow
        session.error = f"{type(exc).__name__}: {exc}"
        await session.emit({"type": "error", "message": session.error})
    finally:
        session.finished.set()
        await session.emit({"type": "closed"})


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    session = _require(session_id)
    return {
        "id": session.id,
        "task": session.task,
        "done": session.done,
        "error": session.error,
        "report": session.report,
        "event_count": len(session.events),
    }


@app.get("/api/sessions/{session_id}/events")
async def stream_events(session_id: str) -> StreamingResponse:
    session = _require(session_id)

    async def generator():
        queue = session.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20.0)
                except TimeoutError:
                    # Keep proxies from dropping an idle connection while a
                    # slow model is still thinking.
                    yield ": keep-alive\n\n"
                    if session.done and queue.empty():
                        break
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "closed":
                    break
        finally:
            session.unsubscribe(queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sessions/{session_id}/report.md", response_class=PlainTextResponse)
async def markdown_report(session_id: str) -> str:
    session = _require(session_id)
    if session.report is None:
        raise HTTPException(status_code=409, detail="run has not finished")
    return to_markdown(session.report)


def _require(session_id: str) -> Session:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session")
    return session


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------

@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    load_dotenv()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    token = os.environ.get("ORCHESTRA_TOKEN")

    if host not in ("127.0.0.1", "localhost", "::1"):
        if token:
            print(
                f"\n  Reachable on the network at http://<this-machine>:{port}/?t={token}\n"
                f"  Open that exact URL on your phone — the API rejects requests "
                f"without the token.\n",
                flush=True,
            )
        else:
            print(
                f"\n  ⚠  Bound to {host} with NO ORCHESTRA_TOKEN set.\n"
                f"     Anyone who can reach port {port} can start runs that spend "
                f"your provider API credits.\n"
                f"     Set ORCHESTRA_TOKEN=$(openssl rand -hex 16) before exposing "
                f"this beyond localhost.\n",
                flush=True,
            )

    uvicorn.run(
        "orchestra.server:app",
        host=host,
        port=port,
        reload=os.environ.get("ORCHESTRA_RELOAD") == "1",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
