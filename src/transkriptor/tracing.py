"""OpenTelemetry tracing setup for Transkriptor.

Uses OpenLLMetry instrumentors (from traceloop-sdk) when available to
auto-instrument LLM client libraries (openai, ollama) with the span attributes
Instana's GenAI dashboard expects.  Falls back to raw OpenTelemetry SDK for
local development without traceloop installed.

Custom pipeline spans are created via ``get_tracer()``.
"""

import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from transkriptor.config import Settings

logger = logging.getLogger(__name__)

_initialised = False


def setup_tracing(settings: Settings) -> None:
    """Initialise OTel tracing if enabled in config. Safe to call multiple times."""
    global _initialised
    if _initialised or not settings.otel_enabled:
        return

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    # Build resource attributes — include INSTANA_PLUGIN=genai for GenAI view
    resource_attrs = {
        "service.name": settings.otel_service_name,
        "service.version": "0.1.0",
        "INSTANA_PLUGIN": "genai",
    }
    # Also set it in env for any libraries that read OTEL_RESOURCE_ATTRIBUTES
    existing = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
    if "INSTANA_PLUGIN=genai" not in existing:
        parts = [p for p in [existing, "INSTANA_PLUGIN=genai"] if p]
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(parts)

    resource = Resource.create(resource_attrs)
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=f"{settings.otel_endpoint.rstrip('/')}/v1/traces",
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Instrument FastAPI (incoming HTTP requests)
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor().instrument()
    except Exception as exc:
        logger.warning("FastAPI instrumentation failed: %s", exc)

    # Instrument httpx (outgoing HTTP calls to Spark services)
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        HTTPXClientInstrumentor().instrument()
    except Exception as exc:
        logger.warning("httpx instrumentation failed: %s", exc)

    # Instrument OpenAI SDK (LLM calls to vLLM — creates gen_ai.* spans)
    _llm_instrumented = False
    try:
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument()
        _llm_instrumented = True
        logger.info("OpenAI (traceloop) LLM instrumentation enabled")
    except ImportError:
        logger.debug("opentelemetry-instrumentation-openai not installed — LLM spans disabled")
    except Exception as exc:
        logger.warning("OpenAI LLM instrumentation failed: %s", exc)

    # Instrument Ollama SDK (LLM calls via ollama library)
    try:
        from opentelemetry.instrumentation.ollama import OllamaInstrumentor
        OllamaInstrumentor().instrument()
        _llm_instrumented = True
        logger.info("Ollama (traceloop) LLM instrumentation enabled")
    except ImportError:
        logger.debug("opentelemetry-instrumentation-ollama not installed — Ollama spans disabled")
    except Exception as exc:
        logger.warning("Ollama LLM instrumentation failed: %s", exc)

    _initialised = True
    logger.info(
        "OpenTelemetry tracing enabled → %s (service=%s, llm_instrumented=%s)",
        settings.otel_endpoint, settings.otel_service_name, _llm_instrumented,
    )


def get_tracer(name: str = "transkriptor") -> trace.Tracer:
    """Return a tracer for creating custom spans."""
    return trace.get_tracer(name)
