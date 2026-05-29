"""Router for the text improvement chat interface."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from transkriptor.services.style_analyzer import load_style_profile
from transkriptor.services.text_improver import improve_text

router = APIRouter()


@router.get("/chat")
async def chat_page(request: Request):
    """Render the text improvement chat page."""
    settings = request.app.state.settings
    profile = load_style_profile(settings.style_profile_path)
    return request.app.state.templates.TemplateResponse(
        request, "chat.html", {"has_style_profile": profile is not None}
    )


@router.post("/api/chat/improve")
async def api_improve_text(request: Request):
    """Improve the submitted text using the configured LLM."""
    settings = request.app.state.settings

    body = await request.json()
    text = body.get("text", "").strip()

    if not text:
        return JSONResponse(
            {"error": "No text provided."},
            status_code=400,
        )

    # Load style profile if available
    style_profile = load_style_profile(settings.style_profile_path)

    result = await improve_text(
        text,
        style_profile=style_profile,
        backend=settings.summary_backend,
        openai_base_url=settings.openai_base_url,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
    )

    return JSONResponse(result)
