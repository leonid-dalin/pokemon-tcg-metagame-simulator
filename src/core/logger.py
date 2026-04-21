# src/core/logger.py
import structlog
import logging
import sys

def setup_structured_logging():
    """
    Initialise the global configuration for structured logging.
    This defines the behaviour of the processing pipeline for the entire app.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()  # Emits JSON logs for ELK/Loki
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    # Standardise the base logging level and output stream for third-party modules
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)


logger = structlog.get_logger()