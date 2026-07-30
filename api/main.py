"""The Clutch prediction service.

Loads the trained pipeline once at startup and serves it. The model artifact is
baked into the container image, so a running task needs nothing external to
answer a request.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.schemas import Health, MatchState, Prediction

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "clutch_model.joblib"

#: A probability at or above this is reported as a predicted win. Named rather
#: than inlined because it is a product decision, not a property of the model:
#: the probability is the real output and the boolean is a convenience.
WIN_THRESHOLD = 0.5


def configure_logging() -> None:
    """Emit one JSON object per line.

    Cloud log collectors parse JSON lines natively, so structured output here is
    what makes a field like win_probability queryable later instead of being
    buried in a formatted string.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once per process rather than once per request.

    Deserialising on every call would add latency for no reason, and it would
    also mean a missing artifact surfaced as a 500 mid-traffic instead of as a
    container that refuses to start.
    """
    configure_logging()
    app.state.model = joblib.load(MODEL_PATH)
    app.state.features = list(app.state.model.feature_names_in_)
    log.info("model_loaded", path=str(MODEL_PATH), n_features=len(app.state.features))
    yield
    app.state.model = None


app = FastAPI(
    title="Clutch",
    description="Predicts League of Legends match outcomes from minute-10 state.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def bind_request_id(request: Request, call_next):
    """Tag every log line from one request with the same id.

    Without this, concurrent requests interleave in the log stream and there is
    no way to tell which prediction belongs to which caller. An inbound
    x-request-id is honoured so a trace survives across a load balancer or an
    upstream service rather than restarting at our door.
    """
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
    finally:
        # Workers are reused across requests, so leaving the binding in place
        # would leak this id onto whatever request lands here next.
        structlog.contextvars.clear_contextvars()

    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected failures instead of losing them.

    The default handler returns a 500 with nothing written down, which means the
    one case you most need to investigate is the one leaving no trace. The
    response stays deliberately vague: internal detail in an error body is a
    disclosure risk, and the request id is enough to find the real cause.
    """
    log.error(
        "unhandled_exception",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
    )
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    """Liveness and readiness for the container health check.

    Reports on the model rather than just returning 200, since a process that is
    up but has no model cannot serve traffic and should not pass a health check.
    """
    model = getattr(request.app.state, "model", None)
    return Health(
        status="ok" if model is not None else "degraded",
        model_loaded=model is not None,
        n_features=len(getattr(request.app.state, "features", [])),
    )


@app.post("/predict", response_model=Prediction)
def predict(state: MatchState, request: Request) -> Prediction:
    """Score a single minute-10 match state.

    Validation happens before this runs: FastAPI rejects a malformed body with a
    422 and the model never sees it.
    """
    started = time.perf_counter()

    features = state.to_feature_row(request.app.state.features)
    probability = float(request.app.state.model.predict_proba(features)[0, 1])

    latency_ms = (time.perf_counter() - started) * 1000
    log.info(
        "prediction",
        win_probability=round(probability, 4),
        gold_diff=state.gold_diff,
        latency_ms=round(latency_ms, 2),
    )

    return Prediction(
        will_win=probability >= WIN_THRESHOLD,
        win_probability=probability,
    )
