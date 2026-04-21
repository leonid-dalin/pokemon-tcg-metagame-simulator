# src/core/telemetry.py
import os
import logging
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_telemetry(service_name: str):
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        return trace.get_tracer(__name__)
    resource = Resource(attributes={"service.name": service_name})
    provider = TracerProvider(resource=resource)

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    otlp_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)

    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    trace.set_tracer_provider(provider)

    logging.getLogger("opentelemetry.exporter.otlp.proto.grpc.trace_exporter").setLevel(logging.ERROR)

    return trace.get_tracer(__name__)


tracer = setup_telemetry("tcg-simulator")