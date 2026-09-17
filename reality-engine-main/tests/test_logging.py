"""Tests for structured logging system."""

import json
import pytest
import tempfile
from pathlib import Path
from engine.core.logging import (
    Logger, LogLevel, LogEntry, get_logger, set_global_log_level,
    clear_loggers, Diagnostics
)


def test_log_entry_json_serialization():
    """Test that log entries serialize to valid JSON."""
    entry = LogEntry(
        timestamp="2024-01-01T12:00:00Z",
        level="INFO",
        module="engine.test",
        message="Test message",
        context={"key": "value"}
    )
    json_str = entry.to_json()
    data = json.loads(json_str)
    assert data["level"] == "INFO"
    assert data["module"] == "engine.test"
    assert data["context"]["key"] == "value"


def test_logger_debug():
    """Test debug logging."""
    logger = Logger("test.module", min_level=LogLevel.DEBUG)
    # Logger just needs to not crash
    logger.debug("Debug message", context={"x": 1})


def test_logger_info():
    """Test info logging."""
    logger = Logger("test.module", min_level=LogLevel.INFO)
    logger.info("Info message")


def test_logger_warning():
    """Test warning logging."""
    logger = Logger("test.module", min_level=LogLevel.WARNING)
    logger.warning("Warning message", context={"severity": "medium"})


def test_logger_error():
    """Test error logging."""
    logger = Logger("test.module", min_level=LogLevel.ERROR)
    logger.error("Error message", context={"code": 500})


def test_logger_critical():
    """Test critical logging."""
    logger = Logger("test.module", min_level=LogLevel.CRITICAL)
    logger.critical("Critical message", context={"fatal": True})


def test_logger_level_filtering():
    """Test that log level filtering works."""
    logger = Logger("test.module", min_level=LogLevel.ERROR)
    # These should be no-ops (not logged)
    logger.debug("Debug (skipped)")
    logger.info("Info (skipped)")
    logger.warning("Warning (skipped)")
    # These should be logged
    logger.error("Error (logged)")
    logger.critical("Critical (logged)")


def test_logger_with_file_output():
    """Test logging to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = Logger("test.module", min_level=LogLevel.INFO, output_file=log_file)
        logger.info("Test message", context={"test": "value"})

        # Verify file was written
        assert log_file.exists()
        content = log_file.read_text()
        assert len(content) > 0

        # Verify it's valid JSON
        data = json.loads(content.strip())
        assert data["message"] == "Test message"
        assert data["context"]["test"] == "value"


def test_get_logger_caching():
    """Test that loggers are cached by name."""
    clear_loggers()
    logger1 = get_logger("test.module")
    logger2 = get_logger("test.module")
    assert logger1 is logger2


def test_get_logger_different_names():
    """Test that different names get different loggers."""
    clear_loggers()
    logger1 = get_logger("test.module1")
    logger2 = get_logger("test.module2")
    assert logger1 is not logger2


def test_set_global_log_level():
    """Test setting global log level."""
    clear_loggers()
    logger1 = get_logger("test.module1", min_level=LogLevel.DEBUG)
    logger2 = get_logger("test.module2", min_level=LogLevel.INFO)

    set_global_log_level(LogLevel.ERROR)

    assert logger1.min_level == LogLevel.ERROR
    assert logger2.min_level == LogLevel.ERROR


def test_clear_loggers():
    """Test clearing logger cache."""
    get_logger("test.module1")
    get_logger("test.module2")
    clear_loggers()
    # After clear, new instances should be created
    logger1a = get_logger("test.module1")
    logger1b = get_logger("test.module1")
    assert logger1a is logger1b


def test_diagnostics_has_issues():
    """Test diagnostics issue detection."""
    # No issues
    diag = Diagnostics()
    assert not diag.has_issues()

    # NaN detected
    diag = Diagnostics(nan_count=1)
    assert diag.has_issues()

    # Inf detected
    diag = Diagnostics(inf_count=1)
    assert diag.has_issues()

    # Warnings
    diag = Diagnostics(warnings=["Some warning"])
    assert diag.has_issues()


def test_diagnostics_log_normal():
    """Test logging normal diagnostics (no issues)."""
    clear_loggers()
    logger = get_logger("test.physics")
    diag = Diagnostics(
        nan_count=0,
        inf_count=0,
        solver_iterations=5,
        contacts_resolved=10,
        energy_kinetic=100.0
    )
    # Should log at debug level (should not crash)
    diag.log_to(logger, tick=100)


def test_diagnostics_log_with_nan():
    """Test logging diagnostics with NaN."""
    clear_loggers()
    logger = get_logger("test.physics")
    diag = Diagnostics(
        nan_count=2,
        inf_count=0,
        solver_iterations=5
    )
    # Should log at error level (should not crash)
    diag.log_to(logger, tick=100)


def test_diagnostics_log_with_warnings():
    """Test logging diagnostics with warnings."""
    clear_loggers()
    logger = get_logger("test.physics")
    diag = Diagnostics(
        nan_count=0,
        inf_count=0,
        solver_iterations=5,
        warnings=["Contact unstable", "Penetration unresolved"]
    )
    # Should log at warning level (should not crash)
    diag.log_to(logger, tick=100)


def test_loglevel_numeric_comparison():
    """Test numeric level ordering."""
    assert LogLevel.DEBUG.numeric_level() < LogLevel.INFO.numeric_level()
    assert LogLevel.INFO.numeric_level() < LogLevel.WARNING.numeric_level()
    assert LogLevel.WARNING.numeric_level() < LogLevel.ERROR.numeric_level()
    assert LogLevel.ERROR.numeric_level() < LogLevel.CRITICAL.numeric_level()


def test_logger_thread_safety():
    """Test that logging is thread-safe (basic check)."""
    import threading
    logger = Logger("test.module", min_level=LogLevel.INFO)

    results = []
    def log_messages():
        for i in range(10):
            logger.info(f"Message {i}")
        results.append(True)

    threads = [threading.Thread(target=log_messages) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 3  # All threads completed
