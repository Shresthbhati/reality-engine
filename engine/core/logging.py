"""
Structured JSON logging system (spec sec 82-83, SYSTEMS_DESIGN.md).

All logs are JSON objects with timestamp, level, module, message, and context.
No plaintext output by default—ensures machine-readable diagnostics.

Usage:
    logger = get_logger("engine.physics.rigid")
    logger.info("Body integrated", context={"body_id": "sphere_1", "velocity": "[1, 2, 3]"})
    logger.error("NaN detected", context={"field": "velocity", "body_id": "box_1"})
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Any
from threading import Lock


class LogLevel(str, Enum):
    """Logging severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def numeric_level(self) -> int:
        """Return numeric severity for filtering."""
        levels = {
            LogLevel.DEBUG: 10,
            LogLevel.INFO: 20,
            LogLevel.WARNING: 30,
            LogLevel.ERROR: 40,
            LogLevel.CRITICAL: 50,
        }
        return levels[self]


@dataclass
class LogEntry:
    """A single structured log entry."""
    timestamp: str  # ISO 8601 format
    level: str
    module: str
    message: str
    context: Optional[dict] = None

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = asdict(self)
        return json.dumps(data, separators=(',', ': '))


class Logger:
    """Structured JSON logger for a module."""

    def __init__(self, name: str, min_level: LogLevel = LogLevel.INFO, output_file: Optional[Path] = None):
        """
        Initialize logger.

        Args:
            name: Module name (e.g., "engine.physics.rigid")
            min_level: Minimum level to log
            output_file: Optional file to write logs to (stdout by default)
        """
        self.name = name
        self.min_level = min_level
        self.output_file = output_file
        self._lock = Lock()

    def debug(self, msg: str, context: Optional[dict] = None) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, msg, context)

    def info(self, msg: str, context: Optional[dict] = None) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, msg, context)

    def warning(self, msg: str, context: Optional[dict] = None) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, msg, context)

    def error(self, msg: str, context: Optional[dict] = None) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, msg, context)

    def critical(self, msg: str, context: Optional[dict] = None) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, msg, context)

    def _log(self, level: LogLevel, msg: str, context: Optional[dict] = None) -> None:
        """Internal logging method with level filtering."""
        if level.numeric_level() < self.min_level.numeric_level():
            return  # Skip logs below threshold

        # Create log entry
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat() + "Z",
            level=level.value,
            module=self.name,
            message=msg,
            context=context,
        )

        # Write with lock to prevent interleaving
        with self._lock:
            json_str = entry.to_json()
            if self.output_file:
                with open(self.output_file, "a") as f:
                    f.write(json_str + "\n")
            else:
                print(json_str, file=sys.stdout)


# Global logger instances (lazy-loaded)
_loggers: dict[str, Logger] = {}
_logger_lock = Lock()


def get_logger(name: str, min_level: LogLevel = LogLevel.INFO, output_file: Optional[Path] = None) -> Logger:
    """
    Get or create a logger for a module.

    Args:
        name: Module name (e.g., "engine.physics.rigid")
        min_level: Minimum level to log (defaults to INFO)
        output_file: Optional file to write logs (defaults to stdout)

    Returns:
        Logger instance (cached)
    """
    global _loggers

    with _logger_lock:
        if name not in _loggers:
            _loggers[name] = Logger(name, min_level=min_level, output_file=output_file)
        return _loggers[name]


def set_global_log_level(level: LogLevel) -> None:
    """Set minimum log level for all existing loggers."""
    with _logger_lock:
        for logger in _loggers.values():
            logger.min_level = level


def clear_loggers() -> None:
    """Clear all cached logger instances (useful for testing)."""
    global _loggers
    with _logger_lock:
        _loggers.clear()


@dataclass
class Diagnostics:
    """Physics simulation diagnostics bundle."""
    nan_count: int = 0
    inf_count: int = 0
    solver_iterations: int = 0
    energy_kinetic: float = 0.0
    energy_potential: float = 0.0
    contacts_resolved: int = 0
    warnings: list[str] = None
    timing_us: dict[str, int] = None

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.warnings is None:
            self.warnings = []
        if self.timing_us is None:
            self.timing_us = {}

    def has_issues(self) -> bool:
        """Return True if diagnostics show problems."""
        return self.nan_count > 0 or self.inf_count > 0 or len(self.warnings) > 0

    def log_to(self, logger: Logger, tick: int) -> None:
        """Write diagnostics to logger."""
        if self.nan_count > 0 or self.inf_count > 0:
            logger.error(
                "NaN/Inf detected in physics state",
                context={
                    "tick": tick,
                    "nan_count": self.nan_count,
                    "inf_count": self.inf_count,
                }
            )

        if len(self.warnings) > 0:
            logger.warning(
                "Physics solver warnings",
                context={
                    "tick": tick,
                    "warnings": self.warnings,
                }
            )

        if self.solver_iterations > 0:
            logger.debug(
                "Physics step complete",
                context={
                    "tick": tick,
                    "solver_iterations": self.solver_iterations,
                    "contacts_resolved": self.contacts_resolved,
                    "energy_kinetic_j": self.energy_kinetic,
                    "energy_potential_j": self.energy_potential,
                    "timing_us": self.timing_us,
                }
            )
