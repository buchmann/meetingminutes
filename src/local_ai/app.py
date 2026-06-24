import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from local_ai.config import Settings
from local_ai.database import Database
from local_ai.services.pipeline import Pipeline
from local_ai.services.recorder import Recorder

logger = logging.getLogger(__name__)


async def _prune_orphan_dirs(settings: Settings, db: Database) -> None:
    """Delete upload/output dirs that no longer have a matching job row."""
    import shutil

    job_ids = await db.all_job_ids()
    for base in (settings.upload_dir, settings.output_dir):
        if not base.exists():
            continue
        for child in base.iterdir():
            if child.is_dir() and child.name not in job_ids:
                shutil.rmtree(child, ignore_errors=True)
                logger.info("Pruned orphan directory %s", child)


async def _seed_admin(settings: Settings, db: Database) -> None:
    """Create the initial admin from config when no users exist yet."""
    if await db.count_users() > 0:
        return
    if not settings.admin_password:
        logger.warning(
            "No users exist and LOCAL_AI_ADMIN_PASSWORD is unset — "
            "set it in .env to seed the initial admin account."
        )
        return
    from local_ai.auth import hash_password

    await db.create_user(
        username=settings.admin_username,
        password_hash=hash_password(settings.admin_password),
        is_admin=True,
    )
    logger.info("Seeded initial admin user '%s'", settings.admin_username)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    settings.ensure_dirs()

    db = Database(settings.db_path)
    await db.initialize()
    await db.recover_stuck_jobs()
    app.state.db = db

    # Remove orphaned upload/output directories left by deleted jobs
    # (e.g. legacy jobs pruned during the multi-user migration).
    await _prune_orphan_dirs(settings, db)

    # Seed an initial admin account on first run.
    await _seed_admin(settings, db)

    # Apply the persisted active LLM choice (default gpt-oss-120b) to settings,
    # so every LLM feature routes to the chosen model + GPU profile.
    from local_ai.config import DEFAULT_LLM, apply_llm
    active = await db.get_app_config("active_llm", DEFAULT_LLM)
    model = apply_llm(settings, active)
    app.state.active_llm = settings.active_llm
    logger.info("Active LLM: %s (%s) at %s", settings.active_llm, model["model"], settings.openai_base_url)

    app.state.pipeline = Pipeline(settings, db)
    app.state.recorder = Recorder(output_dir=settings.data_dir / "recordings")

    if settings.transcription_backend == "remote":
        logger.info(
            "local-ai started — transcription=remote (%s), summary=%s (%s)",
            settings.whisperx_url,
            settings.summary_backend,
            settings.openai_model if settings.summary_backend == "openai" else settings.ollama_model,
        )
    else:
        from local_ai.services.transcriber import detect_engine
        engine = settings.whisper_engine if settings.whisper_engine != "auto" else detect_engine()
        logger.info(
            "local-ai started — engine=%s, whisper=%s, diarization=%s (device=%s), summary=%s",
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
    from local_ai.tracing import setup_tracing
    setup_tracing(settings)

    app = FastAPI(title="local-ai", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    # ── Auth exception handlers ──────────────────────────────────────
    from fastapi.responses import JSONResponse, RedirectResponse

    from local_ai.auth import NotAuthenticated, NotAuthorized

    @app.exception_handler(NotAuthenticated)
    async def _not_authenticated(request, exc):
        if request.url.path.startswith("/api"):
            return JSONResponse({"detail": "Authentication required"}, status_code=401)
        nxt = request.url.path
        return RedirectResponse(url=f"/login?next={nxt}", status_code=303)

    @app.exception_handler(NotAuthorized)
    async def _not_authorized(request, exc):
        if request.url.path.startswith("/api"):
            return JSONResponse({"detail": "Admin access required"}, status_code=403)
        return RedirectResponse(url="/", status_code=303)

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent.parent

    templates_dir = base_dir / "templates"
    app.state.templates = Jinja2Templates(directory=str(templates_dir))

    static_dir = project_root / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    from local_ai.routers import (
        chat, consolidate, daily, documents, email, immo, jobs, notes, pages, exports,
        projects, translate, websearch,
    )
    app.include_router(pages.router)
    app.include_router(jobs.router)
    app.include_router(exports.router)
    app.include_router(chat.router)
    app.include_router(documents.router)
    app.include_router(consolidate.router)
    app.include_router(translate.router)
    app.include_router(websearch.router)
    app.include_router(email.router)
    app.include_router(notes.router)
    app.include_router(immo.router)
    app.include_router(projects.router)
    app.include_router(daily.router)

    return app
