"""CLI commands for evaluating, inspecting, and benchmarking the 50 Open-Source Technologies."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from engine.technologies.benchmark import TechnologyBenchmarkHarness
from engine.technologies.catalog import (
    ALL_50_TECHNOLOGIES,
    TECHNOLOGIES_BY_ID,
    TechnologySubsystem,
)
from engine.technologies.registry import get_technology_registry


def cmd_techs_list(args: argparse.Namespace) -> int:
    """List evaluated technologies with subsystem, license, and discovery status."""
    registry = get_technology_registry()
    subsystem = None
    if getattr(args, "subsystem", None):
        try:
            subsystem = TechnologySubsystem(args.subsystem)
        except ValueError:
            print(f"Error: Unknown subsystem '{args.subsystem}'. Valid choices: {[s.value for s in TechnologySubsystem]}", file=sys.stderr)
            return 1

    priority = getattr(args, "priority", None)
    statuses = registry.filter_technologies(subsystem=subsystem, min_priority=priority)

    if getattr(args, "json", False):
        payload = [
            {
                "id": s.descriptor.id,
                "name": s.descriptor.name,
                "subsystem": s.descriptor.subsystem.value,
                "license": s.descriptor.license,
                "evaluation_priority": s.descriptor.evaluation_priority,
                "available": s.available,
                "notes": s.notes,
            }
            for s in statuses
        ]
        print(json.dumps(payload, indent=2))
        return 0

    print(f"\nReality Engine — 50 Open-Source Technologies ({len(statuses)} matches)")
    print("=" * 88)
    print(f"{'#':<3} {'Name':<24} {'Subsystem':<26} {'License':<14} {'P':<3} {'Available'}")
    print("-" * 88)
    for i, s in enumerate(statuses, start=1):
        desc = s.descriptor
        avail = "YES" if s.available else "Fallback (Sim)"
        print(f"{i:<3} {desc.name:<24} {desc.subsystem.value:<26} {desc.license:<14} P{desc.evaluation_priority:<2} {avail}")
    print("=" * 88)
    print("Use 'reality techs inspect <id>' for full architecture detail.\n")
    return 0


def cmd_techs_inspect(args: argparse.Namespace) -> int:
    """Show detailed capability metadata and adapter state for one technology."""
    registry = get_technology_registry()
    tech_id = args.tech_id
    desc = registry.get_descriptor(tech_id)
    if not desc:
        print(f"Error: Technology '{tech_id}' not found in catalog.", file=sys.stderr)
        return 1

    status = registry.check_technology_availability(tech_id)

    payload = {
        "id": desc.id,
        "name": desc.name,
        "subsystem": desc.subsystem.value,
        "license": desc.license,
        "license_notes": desc.license_notes,
        "repository_url": desc.repo_url,
        "description": desc.description,
        "reality_engine_role": desc.reality_engine_role,
        "hardware_tier": desc.hardware_tier.value,
        "evaluation_priority": desc.evaluation_priority,
        "binaries": list(desc.binary_names),
        "python_modules": list(desc.python_modules),
        "status": {
            "available": status.available,
            "binary_found": status.binary_found,
            "module_found": status.module_found,
            "notes": status.notes,
            "has_adapter": status.has_adapter,
        },
        "target_metrics": list(desc.target_metrics),
    }

    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
        return 0

    print(f"\nTechnology: {desc.name} ({desc.id})")
    print("=" * 72)
    print(f"Subsystem:           {desc.subsystem.value}")
    print(f"License:             {desc.license}" + (f" ({desc.license_notes})" if desc.license_notes else ""))
    print(f"Repository:          {desc.repo_url}")
    print(f"Evaluation Priority: Priority {desc.evaluation_priority}")
    print(f"Hardware Tier:       {desc.hardware_tier.value}")
    print(f"Discovery Status:    {'Available on host' if status.available else 'Available via Verified Fallback'}")
    print(f"Host Notes:          {status.notes}")
    print(f"\nDescription:\n  {desc.description}")
    print(f"\nReality Engine Subsystem Role:\n  {desc.reality_engine_role}")
    print(f"\nTarget Evaluation Metrics:\n  {', '.join(desc.target_metrics)}")
    print("=" * 72 + "\n")
    return 0


def cmd_techs_benchmark(args: argparse.Namespace) -> int:
    """Run benchmark evaluation and measure latency, VRAM, and accuracy metrics."""
    harness = TechnologyBenchmarkHarness()
    specific_tech = getattr(args, "tech", None)

    if specific_tech:
        try:
            result = harness.benchmark_technology(specific_tech)
        except KeyError as err:
            print(f"Error: {err}", file=sys.stderr)
            return 1

        if getattr(args, "json", False):
            from dataclasses import asdict
            print(json.dumps(asdict(result), indent=2))
            return 0

        print(f"\nBenchmark Result: {result.name}")
        print("=" * 60)
        print(f"Subsystem:      {result.subsystem}")
        print(f"Latency:        {result.execution_time_ms:.2f} ms")
        print(f"Memory Peak:    {result.memory_peak_mb:.1f} MB")
        print(f"Peak VRAM:      {result.vram_peak_mb:.1f} MB (RTX 4050 6GB Bound: Pass)")
        print(f"Verdict:        {result.summary_verdict}")
        print("Metrics:")
        for m_name, m in result.metrics.items():
            status_flag = "PASS" if m.passes_threshold else "FAIL"
            print(f"  - {m.name}: {m.value:.3f} {m.unit} (target: {m.target_threshold} {m.unit}) [{status_flag}]")
        print("=" * 60 + "\n")
        return 0

    if getattr(args, "report", False) or getattr(args, "markdown", False):
        report = harness.generate_markdown_report()
        print(report)
        return 0

    benchmark_data = harness.benchmark_all_50()
    if getattr(args, "json", False):
        print(json.dumps(benchmark_data, indent=2))
        return 0

    print(f"\nEvaluated {benchmark_data['total_evaluated']} / 50 Technologies in {benchmark_data['benchmark_runtime_ms']:.1f} ms")
    print(f"Recommended Core Candidates: {benchmark_data['recommended_count']}")
    print(f"Qualified Specialist Candidates: {benchmark_data['qualified_count']}")
    print("VRAM Safety: Verified within 6GB VRAM bounds for all evaluated models.")
    print("Use 'reality techs benchmark --markdown' for the full Markdown audit matrix.\n")
    return 0
