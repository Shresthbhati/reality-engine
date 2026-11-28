from apps.api.jobs import validation_failure_detail
from world_ir.validation import ValidationIssue, ValidationSeverity, WorldValidationReport


def test_failure_detail_lists_errors_not_the_warnings_that_precede_them():
    warns = [ValidationIssue(ValidationSeverity.WARNING, "entity_duplicate_identity", f"boot{i} unobserved") for i in range(8)]
    err = ValidationIssue(ValidationSeverity.ERROR, "real_blocker", "the actual error")
    detail = validation_failure_detail(WorldValidationReport(issues=[*warns, err]))
    assert "the actual error" in detail and "unobserved" not in detail
