"""Router for the text improvement chat interface."""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from local_ai.auth import require_user
from local_ai.services.summarizer import clamp_temperature
from local_ai.services.text_improver import improve_text

router = APIRouter()


@router.get("/chat")
async def chat_page(request: Request, user: dict = Depends(require_user)):
    """Render the text improvement chat page."""
    db = request.app.state.db
    profile = await db.get_user_style_profile(user["id"])
    return request.app.state.templates.TemplateResponse(
        request, "chat.html", {"has_style_profile": profile is not None, "user": user}
    )


@router.post("/api/chat/improve")
async def api_improve_text(request: Request, user: dict = Depends(require_user)):
    """Improve the submitted text using the configured LLM and the user's style."""
    settings = request.app.state.settings
    db = request.app.state.db

    body = await request.json()
    text = body.get("text", "").strip()
    temperature = clamp_temperature(body.get("temperature"))

    if not text:
        return JSONResponse({"error": "No text provided."}, status_code=400)

    style_profile = await db.get_user_style_profile(user["id"])

    result = await improve_text(
        text,
        style_profile=style_profile,
        backend=settings.summary_backend,
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        temperature=temperature,
    )

    return JSONResponse(result)
