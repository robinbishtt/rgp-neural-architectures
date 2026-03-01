"""
audit/run_suite.py

Per-suite command-line runner.

Runs a single named suite and emits a JSON report.  Designed so that the
Audit Bureau (or any external reviewer) can execute exactly one folder's
worth of tests in isolation, without touching the rest of the codebase.

Usage  pytest suites
---------------------
    python audit/run_suite.py unit
    python audit/run_suite.py integration
    python audit/run_suite.py stability
    python audit/run_suite.py ablation
    python audit/run_suite.py robustness
    python audit/run_suite.py scaling
    python audit/run_suite.py spectral
    python audit/run_suite.py validation

Usage  study / figure suites
------------------------------
    python audit/run_suite.py h1_scale_correspondence
    python audit/run_suite.py h2_depth_scaling
    python audit/run_suite.py h3_multiscale_generalization
    python audit/run_suite.py figures_extended_data
    python audit/run_suite.py figures_manuscript

Optional flags
--------------
    --seed INT          Master seed (default: 42)
    --output PATH       JSON report path (default: audit_reports/suite_<name>_<id>.json)
    --timeout INT       Per-script timeout in seconds (default: 600)
    --pytest-args ...   Extra args forwarded to pytest
    --quiet             Suppress progress output

Exit codes
----------
    0   All cases passed.
    1   One or more failures / errors.
    2   Unknown suite name or runner-level error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from audit.runner import AuditRunner

# Canonical mapping of suite slug → type.
_PYTEST_SUITES = {
    "unit",
    "integration",
    "stability",
    "ablation",
    "robustness",
    "scaling",
    "spectral",
    "validation",
}

_STUDY_SUITES = {
    "h1_scale_correspondence",
    "h2_depth_scaling",
    "h3_multiscale_generalization",
    "figures_extended_data",
    "figures_manuscript",
}

_ALL_SUITES = _PYTEST_SUITES | _STUDY_SUITES


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "python audit/run_suite.py",
        description = "Run a single audit suite and write a JSON report.",
        epilog      = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "suite",
        choices = sorted(_ALL_SUITES),
        help    = "Name of the suite to execute.",
    )
    p.add_argument("--seed",    type=int, default=42,  metavar="INT")
    p.add_argument("--output",  metavar="PATH")
    p.add_argument("--timeout", type=int, default=600, metavar="SECONDS")
    p.add_argument("--quiet",   action="store_true")
    p.add_argument(
        "--pytest-args",
        nargs   = argparse.REMAINDER,
        default = [],
        metavar = "ARG",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Derive a suite-scoped output path if none supplied.
    output = args.output
    if output is None:
        import uuid
        short_id = str(uuid.uuid4())[:8]
        reports  = _REPO_ROOT / "audit_reports"
        reports.mkdir(parents=True, exist_ok=True)
        output   = str(reports / f"suite_{args.suite}_{short_id}.json")

    runner = AuditRunner(
        repo_root          = str(_REPO_ROOT),
        master_seed        = args.seed,
        output_path        = output,
        timeout_per_script = args.timeout,
        verbose            = not args.quiet,
    )

    skip_studies = (args.suite in _PYTEST_SUITES)
    suites_list  = [args.suite]

    try:
        report = runner.run(
            suites       = suites_list,
            pytest_extra = args.pytest_args,
            skip_studies = skip_studies,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Runner error: {exc}", file=sys.stderr)
        return 2

    return 0 if report.get("summary", {}).get("all_passed") else 1


if __name__ == "__main__":
    sys.exit(main())
