"""OpenTelemetry tracing + metrics setup for local-ai.

Uses OpenLLMetry instrumentors (from traceloop-sdk) when available to
auto-instrument LLM client libraries (openai, ollama) with the span attributes
Instana's GenAI dashboard expects.  Falls back to raw OpenTelemetry SDK for
local development without traceloop installed.

Additionally sets up an OTEL **MeterProvider** that emits ``llm.usage.*``
and ``llm.request.count`` metrics via OTLP HTTP directly to the Instana
agent.  Instana's GenAI dashboard requires these *metrics* (gauges / sums)
— span attributes alone are not enough.

Custom pipeline spans are created via ``get_tracer()``.
LLM metrics are recorded via ``get_llm_metrics()``.
"""

import logging
import os

from opentelemetry import metrics as otel_metrics
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from local_ai.config import Settings

logger = logging.getLogger(__name__)

_initialised = False

# ── LLM metrics instruments (populated by setup_tracing) ──────────
_llm_input_tokens = None
_llm_output_tokens = None
_llm_total_tokens = None
_llm_response_duration = None
_llm_request_count = None


class LLMMetrics:
    """Thin wrapper around OTEL metric instruments for LLM calls."""

    def __init__(self, input_tokens, output_tokens, total_tokens, duration, count):
        self._input = input_tokens
        self._output = output_tokens
        self._total = total_tokens
        self._duration = duration
        self._count = count

    def record(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        duration_ms: float,
        service_name: str,
        model_id: str,
        ai_system: str = "openai",
    ) -> None:
        """Record one LLM request's metrics with the tags Instana expects.

        Uses both OpenTelemetry GenAI semantic convention names (gen_ai.*)
        AND the short names (model_id, ai_system, service_name) so the
        Instana agent's otel-sensorsdk-genai plugin can find the model ID.
        """
        attrs = {
            # OTEL GenAI semantic convention names
            "gen_ai.request.model": model_id,
            "gen_ai.response.model": model_id,
            "gen_ai.system": ai_system,
            # Short names used by Instana tag filters
            "service_name": service_name,
            "model_id": model_id,
            "ai_system": ai_system,
            # Additional: llm.* convention
            "llm.request.model": model_id,
        }
        if self._input:
            self._input.set(input_tokens, attrs)
        if self._output:
            self._output.set(output_tokens, attrs)
        if self._total:
            self._total.set(total_tokens, attrs)
        if self._duration:
            self._duration.set(duration_ms, attrs)
        if self._count:
            self._count.add(1, attrs)
        logger.debug(
            "LLM metrics recorded: in=%d out=%d total=%d dur=%.0fms model=%s",
            input_tokens, output_tokens, total_tokens, duration_ms, model_id,
        )


_llm_metrics: LLMMetrics | None = None


def get_llm_metrics() -> LLMMetrics | None:
    """Return the LLMMetrics recorder, or None if tracing is disabled."""
    return _llm_metrics


def setup_tracing(settings: Settings) -> None:
    """Initialise OTel tracing + metrics if enabled in config. Safe to call multiple times."""
    global _initialised, _llm_metrics
    if _initialised or not settings.otel_enabled:
        return

    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
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

    # ── Traces ────────────────────────────────────────────────────
    # If Instana init container already set a TracerProvider, use it.
    # Otherwise set our own (e.g. local dev without Instana).
    existing_provider = trace.get_tracer_provider()
    instana_agent_host = os.environ.get("INSTANA_AGENT_HOST")
    if instana_agent_host and not isinstance(existing_provider, TracerProvider):
        # Instana's provider is active — don't override, just use it for traces
        logger.info(
            "Using Instana TracerProvider (agent at %s) — skipping custom TracerProvider",
            instana_agent_host,
        )
    else:
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=f"{settings.otel_endpoint.rstrip('/')}/v1/traces",
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info("Custom TracerProvider set → %s/v1/traces", settings.otel_endpoint)

    # ── Metrics endpoint — prefer local Instana agent if available ──
    if instana_agent_host:
        metrics_endpoint = f"http://{instana_agent_host}:4318/v1/metrics"
        logger.info("Instana agent detected — sending metrics to %s", metrics_endpoint)
    else:
        metrics_endpoint = f"{settings.otel_endpoint.rstrip('/')}/v1/metrics"

    # ── Metrics (llm.usage.* gauges + llm.request.count counter) ──
    metric_exporter = OTLPMetricExporter(
        endpoint=metrics_endpoint,
    )
    metric_reader = PeriodicExportingMetricReader(
        metric_exporter,
        export_interval_millis=15000,   # flush every 15 s for fast feedback
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    otel_metrics.set_meter_provider(meter_provider)

    meter = meter_provider.get_meter("local_ai.llm", version="0.1.0")

    _llm_metrics = LLMMetrics(
        input_tokens=meter.create_gauge(
            "llm.usage.input_tokens",
            unit="tokens",
            description="Number of input (prompt) tokens for the last LLM request",
        ),
        output_tokens=meter.create_gauge(
            "llm.usage.output_tokens",
            unit="tokens",
            description="Number of output (completion) tokens for the last LLM request",
        ),
        total_tokens=meter.create_gauge(
            "llm.usage.total_tokens",
            unit="tokens",
            description="Total tokens (input + output) for the last LLM request",
        ),
        duration=meter.create_gauge(
            "llm.response.duration",
            unit="ms",
            description="LLM response latency in milliseconds",
        ),
        count=meter.create_counter(
            "llm.request.count",
            unit="requests",
            description="Cumulative number of LLM requests",
        ),
    )
    logger.info(
        "OTEL metrics enabled → %s (flush every 15s)",
        metrics_endpoint,
    )

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


def get_tracer(name: str = "local_ai") -> trace.Tracer:
    """Return a tracer for creating custom spans."""
    return trace.get_tracer(name)
