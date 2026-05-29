import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from transkriptor.config import Settings
from transkriptor.database import Database
from transkriptor.services.pipeline import Pipeline
from transkriptor.services.recorder import Recorder

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    settings.ensure_dirs()

    db = Database(settings.db_path)
    await db.initialize()
    await db.recover_stuck_jobs()
    app.state.db = db

    app.state.pipeline = Pipeline(settings, db)
    app.state.recorder = Recorder(output_dir=settings.data_dir / "recordings")

    if settings.transcription_backend == "remote":
        logger.info(
            "Transkriptor started — transcription=remote (%s), summary=%s (%s)",
            settings.whisperx_url,
            settings.summary_backend,
            settings.openai_model if settings.summary_backend == "openai" else settings.ollama_model,
        )
    else:
        from transkriptor.services.transcriber import detect_engine
        engine = settings.whisper_engine if settings.whisper_engine != "auto" else detect_engine()
        logger.info(
            "Transkriptor started — engine=%s, whisper=%s, diarization=%s (device=%s), summary=%s",
            engine, settings.whisper_model,
            "on" if settings.diarization_enabled else "off",
            settings.diarization_device,
            settings.summary_backend,
        )
    yield

    await db.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    # Initialise OTel tracing before FastAPI so the instrumentor hooks in
    from transkriptor.tracing import setup_tracing
    setup_tracing(settings)

    app = FastAPI(title="Transkriptor", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent.parent

    templates_dir = base_dir / "templates"
    app.state.templates = Jinja2Templates(directory=str(templates_dir))

    static_dir = project_root / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    from transkriptor.routers import chat, jobs, pages, exports
    app.include_router(pages.router)
    app.include_router(jobs.router)
    app.include_router(exports.router)
    app.include_router(chat.router)

    return app
