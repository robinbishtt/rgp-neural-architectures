"""
audit/run_audit.py

Master audit command-line entry point.

One command to orchestrate the complete RGP audit trail:
  - All eight pytest suites (unit, integration, stability, ablation,
    robustness, scaling, spectral, validation)
  - All five study/figure suites (H1, H2, H3 experiments + figures)
  - JSON report written atomically to audit_reports/

Usage
-----
    # Full audit — all tests + all study scripts:
    python audit/run_audit.py

    # Tests only (no study scripts):
    python audit/run_audit.py --tests-only

    # Specific suites:
    python audit/run_audit.py --suites unit stability spectral

    # Custom seed and output path:
    python audit/run_audit.py --seed 123 --output /tmp/my_audit.json

    # Pass extra flags to pytest (e.g. stop on first failure):
    python audit/run_audit.py --pytest-args -x --tb=long

    # Quiet (suppress progress lines):
    python audit/run_audit.py --quiet

Exit codes
----------
    0   All cases passed.
    1   One or more cases failed or errored.
    2   Runner-level error (e.g. wrong working directory).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from audit.runner import AuditRunner


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "python audit/run_audit.py",
        description = "RGP Neural Architectures — master audit orchestrator.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog      = __doc__,
    )

    # ---- Scope ------------------------------------------------------------
    scope = p.add_argument_group("Scope")
    scope.add_argument(
        "--suites",
        nargs  = "+",
        metavar= "SUITE",
        help   = (
            "Whitelist of suite names to run.  Available: "
            "unit, integration, stability, ablation, robustness, "
            "scaling, spectral, validation, "
            "h1_scale_correspondence, h2_depth_scaling, "
            "h3_multiscale_generalization, figures_extended_data, "
            "figures_manuscript."
        ),
    )
    scope.add_argument(
        "--tests-only",
        action  = "store_true",
        help    = "Run pytest suites only; skip all study/figure scripts.",
    )
    scope.add_argument(
        "--studies-only",
        action  = "store_true",
        help    = "Run study/figure scripts only; skip pytest suites.",
    )

    # ---- Reproducibility --------------------------------------------------
    repro = p.add_argument_group("Reproducibility")
    repro.add_argument(
        "--seed",
        type    = int,
        default = 42,
        metavar = "INT",
        help    = "Master seed for SeedRegistry (default: 42).",
    )

    # ---- Output -----------------------------------------------------------
    output = p.add_argument_group("Output")
    output.add_argument(
        "--output",
        metavar = "PATH",
        help    = (
            "Path for the JSON audit report.  Defaults to "
            "audit_reports/audit_<run_id>.json."
        ),
    )
    output.add_argument(
        "--quiet",
        action  = "store_true",
        help    = "Suppress real-time progress output.",
    )

    # ---- Pytest pass-through ----------------------------------------------
    pytest_group = p.add_argument_group("Pytest")
    pytest_group.add_argument(
        "--pytest-args",
        nargs   = argparse.REMAINDER,
        default = [],
        metavar = "ARG",
        help    = "Extra arguments forwarded verbatim to every pytest invocation.",
    )
    pytest_group.add_argument(
        "--timeout",
        type    = int,
        default = 600,
        metavar = "SECONDS",
        help    = "Per-script wall-clock timeout in seconds (default: 600).",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # Build suite whitelist from --studies-only / --tests-only flags.
    suites = args.suites or None
    if args.tests_only and args.studies_only:
        parser.error("--tests-only and --studies-only are mutually exclusive.")

    runner = AuditRunner(
        repo_root           = str(_REPO_ROOT),
        master_seed         = args.seed,
        output_path         = args.output,
        timeout_per_script  = args.timeout,
        verbose             = not args.quiet,
    )

    try:
        report = runner.run(
            suites       = suites,
            pytest_extra = args.pytest_args,
            skip_studies = args.tests_only,
        )
    except KeyboardInterrupt:
        print("\nAudit interrupted by user.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"\nRunner-level error: {exc}", file=sys.stderr)
        return 2

    # If --studies-only, filter suites list to study suites only.
    # (AuditRunner.run() already accepted skip_studies; this path handles
    # the inverse case by re-running with a constrained suites list.)
    if args.studies_only and suites is None:
        study_suites = [
            "h1_scale_correspondence",
            "h2_depth_scaling",
            "h3_multiscale_generalization",
            "figures_extended_data",
            "figures_manuscript",
        ]
        runner2 = AuditRunner(
            repo_root          = str(_REPO_ROOT),
            master_seed        = args.seed,
            output_path        = args.output,
            timeout_per_script = args.timeout,
            verbose            = not args.quiet,
        )
        report = runner2.run(
            suites      = study_suites,
            pytest_extra= args.pytest_args,
        )

    return 0 if report.get("summary", {}).get("all_passed") else 1


if __name__ == "__main__":
    sys.exit(main())
