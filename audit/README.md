# Audit Subsystem

Structured JSON audit trail for the RGP Neural Architectures pipeline.

Every audit run produces one self-contained JSON file in `audit_reports/`
capturing git provenance, hardware specification, seed state, and the
pass/fail outcome of every test case and study script.  The schema is defined
in `audit/schema.py` (JSON Schema Draft 2020-12) and an annotated example is
provided in `audit/example_report.json`.

---

## Quick start

```bash
# Full audit — all pytest suites + all experiment/figure scripts
python audit/run_audit.py

# Tests only
python audit/run_audit.py --tests-only

# Study scripts only
python audit/run_audit.py --studies-only

# Single suite
python audit/run_suite.py unit
python audit/run_suite.py h1_scale_correspondence

# Via Make
make audit
make audit_tests
make audit_unit
make audit_h1
```

Reports are written atomically to `audit_reports/audit_<run-id>.json`.

---

## Suite inventory

### Pytest suites (`--tests-only` scope)

| Suite slug | Directory | Test count (approx.) |
|------------|-----------|----------------------|
| `unit` | `tests/unit/` | 20 |
| `integration` | `tests/integration/` | 7 |
| `stability` | `tests/stability/` | 11 |
| `ablation` | `tests/ablation/` | 14 |
| `robustness` | `tests/robustness/` | 9 |
| `scaling` | `tests/scaling/` | 9 |
| `spectral` | `tests/spectral/` | 7 |
| `validation` | `tests/validation/` | 7 |

### Study suites (`--studies-only` scope)

| Suite slug | Directory | Scripts |
|------------|-----------|---------|
| `h1_scale_correspondence` | `experiments/h1_scale_correspondence/` | 5 |
| `h2_depth_scaling` | `experiments/h2_depth_scaling/` | 5 |
| `h3_multiscale_generalization` | `experiments/h3_multiscale_generalization/` | 5 |
| `figures_extended_data` | `figures/extended_data/` | 11 |
| `figures_manuscript` | `figures/manuscript/` | 5 |

---

## Report structure

```
{
  "run_id":         "...",      ← UUID v4 unique per run
  "timestamp_utc":  "...",      ← ISO 8601 UTC
  "git": {
    "commit_hash":  "...",      ← full SHA-1
    "branch":       "main",
    "dirty":        false       ← true = uncommitted changes present
  },
  "hardware": {
    "device_type":  "cuda",     ← cuda | mps | cpu
    "device_name":  "...",
    "cuda_version": "12.1",
    ...
  },
  "seed_registry": {
    "master_seed":        42,
    "deterministic_cuda": true,
    "benchmark_disabled": true,
    "worker_seed_formula": "..."
  },
  "suites": {
    "unit": {
      "cases": [
        { "name": "...", "status": "passed|failed|error|skipped", ... }
      ]
    },
    ...
  },
  "summary": {
    "total_cases": N,
    "passed": N,  "failed": N,  "errored": N,  "skipped": N,
    "all_passed": true|false,
    "duration_seconds": N
  }
}
```

See `audit/example_report.json` for a fully annotated example.

---

## Robustness guarantee

Each test case is wrapped in a `try-except` inside its own subprocess.
A crash in one case sets `status: "error"` in that case's record and
records the traceback, but never corrupts the JSON file.  The report is
written via atomic rename (`write .tmp` → `os.replace`) so partial writes
cannot occur even if the process is killed mid-write.

---

## Dependency

`pytest-json-report` is required for pytest suites:

```bash
pip install pytest-json-report
```

`jsonschema` is optional (enables `AuditSchema.validate()`):

```bash
pip install jsonschema
```

---

## License

MIT — see `LICENSE` at the repository root.
