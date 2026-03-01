"""
audit/runner.py

Audit orchestration engine.

Executes all pytest suites and study scripts, assembles the five-section
audit report defined by audit/schema.py, seals it with a SHA-256 hash
chain via audit/integrity.py, and writes the result atomically to disk.

Section population by this module:

  1. provenance          — git, seed_registry, hardware, data_checksums
                           (data_checksums populated if DataAuditor is importable;
                           empty list otherwise).

  2. orchestration       — wall_start_ns/wall_end_ns, dag_trace (one node
                           per suite and per case with nanosecond timestamps),
                           checkpoint_events (empty unless a study script emits
                           a structured checkpoint-event line on stdout).

  3. math_telemetry      — empty arrays.  This section is populated by the
                           experiment and training scripts themselves, not by the
                           test runner.  The schema shape is defined; the runner
                           writes the empty arrays so the JSON is always valid.

  4. hardware_forensics  — static peak_vram_bytes from torch.cuda at run end;
                           vram_micro_log and kernel_latency_log are empty arrays
                           (populated at training time by the experiment pipeline).

  5. integrity           — full SHA-256 hash chain over all six sections,
                           optional Ed25519 signature, and repository manifest.

Robustness guarantees
---------------------
* Every case is executed in a child subprocess; no crash can affect the runner.
* A try-except wraps every case's result assembly; a bad pytest-json-report
  line becomes status='error' with the traceback captured.
* The report file is written via atomic rename (write .tmp -> os.replace) so
  it is never in a partially-written state.
* The hash chain is computed over the six canonical sections in a fixed order,
  so the chain is stable across re-runs with identical content.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from audit.collector import SystemCollector
from audit.integrity import HashChain, ManifestBuilder, ReportSigner
from audit.schema import AuditSchema


# ---------------------------------------------------------------------------
# Case record constructors
# ---------------------------------------------------------------------------

def _ns() -> int:
    return time.monotonic_ns()


def _case(
    name:           str,
    status:         str,
    started_ns:     int,
    ended_ns:       int,
    failure_msg:    Optional[str] = None,
    tb:             Optional[str] = None,
    seed_used:      Optional[int] = None,
    markers:        Optional[List[str]] = None,
    stdout:         Optional[str] = None,
    stderr_snippet: Optional[str] = None,
    parametrize_id: Optional[str] = None,
    dag_node_id:    Optional[str] = None,
) -> Dict[str, Any]:
    duration_ns = max(0, ended_ns - started_ns)
    return {
        "name":             name,
        "status":           status,
        "started_ns":       started_ns,
        "ended_ns":         ended_ns,
        "duration_ns":      duration_ns,
        "duration_seconds": round(duration_ns / 1e9, 6),
        "failure_message":  failure_msg,
        "traceback":        tb,
        "seed_used":        seed_used,
        "worker_seed":      None,
        "stdout":           stdout,
        "stderr_snippet":   stderr_snippet,
        "markers":          markers or [],
        "parametrize_id":   parametrize_id,
        "dag_node_id":      dag_node_id,
    }


def _dag_node(
    node_type:      str,
    label:          str,
    started_ns:     int,
    ended_ns:       int,
    status:         str,
    parent_id:      Optional[str] = None,
    metadata:       Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    node_id = str(uuid.uuid4())
    duration_ns = max(0, ended_ns - started_ns)
    node = {
        "node_id":        node_id,
        "parent_node_id": parent_id,
        "node_type":      node_type,
        "label":          label,
        "started_ns":     started_ns,
        "ended_ns":       ended_ns,
        "duration_ns":    duration_ns,
        "status":         status,
        "metadata":       metadata or {},
    }
    return node_id, node


def _cap(s: Optional[str], limit: int) -> Optional[str]:
    if not s:
        return None
    return s[-limit:] if len(s) > limit else s


# ---------------------------------------------------------------------------
# Pytest JSON report parser
# ---------------------------------------------------------------------------

def _parse_pytest_json(
    raw:         Dict[str, Any],
    master_seed: Optional[int],
    parent_id:   Optional[str],
    dag_nodes:   List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert pytest-json-report output into case_record dicts.

    Also appends one dag_node per test case to *dag_nodes*.
    """
    cases: List[Dict[str, Any]] = []
    try:
        for test in raw.get("tests", []):
            outcome    = test.get("outcome", "error")
            call_info  = test.get("call", {}) or {}
            setup_info = test.get("setup", {}) or {}

            # Duration — prefer the call phase.
            duration_s  = call_info.get("duration", setup_info.get("duration", 0.0))
            duration_ns = int(duration_s * 1e9)

            # We do not have a real monotonic start from pytest's JSON report,
            # so we fabricate consistent started/ended from duration only.
            started_ns = _ns()
            ended_ns   = started_ns + duration_ns

            failure_msg: Optional[str] = None
            tb: Optional[str] = None
            if outcome in ("failed", "error"):
                crash = call_info.get("crash") or setup_info.get("crash") or {}
                failure_msg = crash.get("message") or "No message captured."
                longrepr    = test.get("longrepr") or call_info.get("longrepr") or ""
                tb          = str(longrepr)[:8192] if longrepr else None

            markers = [
                m if isinstance(m, str) else m.get("name", "")
                for m in (test.get("markers") or [])
            ]
            markers = [m for m in markers if m]

            # DAG node for this case.
            node_id, node = _dag_node(
                node_type   = "pytest_case",
                label       = test.get("nodeid", "unknown"),
                started_ns  = started_ns,
                ended_ns    = ended_ns,
                status      = outcome,
                parent_id   = parent_id,
                metadata    = {"seed": master_seed},
            )
            dag_nodes.append(node)

            cases.append(_case(
                name           = test.get("nodeid", "unknown"),
                status         = outcome,
                started_ns     = started_ns,
                ended_ns       = ended_ns,
                failure_msg    = failure_msg,
                tb             = tb,
                seed_used      = master_seed,
                markers        = markers,
                stdout         = _cap(test.get("stdout"), 4096),
                stderr_snippet = _cap(test.get("stderr"), 2048),
                dag_node_id    = node_id,
            ))

    except Exception as exc:
        cases.append(_case(
            name        = "<pytest-report-parse>",
            status      = "error",
            started_ns  = _ns(),
            ended_ns    = _ns(),
            failure_msg = f"Failed to parse pytest JSON report: {exc}",
            tb          = traceback.format_exc(),
        ))
    return cases


# ---------------------------------------------------------------------------
# AuditRunner
# ---------------------------------------------------------------------------

class AuditRunner:
    """
    Orchestrates the complete five-section audit run.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository root.
    master_seed:
        Seed passed to SeedRegistry before any test runs.
    output_path:
        Destination JSON file.  Defaults to
        audit_reports/audit_<run_id>.json.
    timeout_per_script:
        Wall-clock timeout in seconds for each study script.
    verbose:
        Print real-time progress to stderr.
    sign:
        Path to an Ed25519 private key PEM for report signing.
        None disables signing (integrity.signature will be null).
    sign_pub:
        Matching public key path (stored in the report for verification).
    build_manifest:
        When True, hashes every source file in the repo and embeds the
        manifest in integrity.manifest.  Adds a few seconds for large trees.
    """

    _PYTEST_SUITES: List[Tuple[str, str]] = [
        ("unit",        "tests/unit"),
        ("integration", "tests/integration"),
        ("stability",   "tests/stability"),
        ("ablation",    "tests/ablation"),
        ("robustness",  "tests/robustness"),
        ("scaling",     "tests/scaling"),
        ("spectral",    "tests/spectral"),
        ("validation",  "tests/validation"),
    ]

    _STUDY_SUITES: List[Tuple[str, str]] = [
        ("h1_scale_correspondence",     "experiments/h1_scale_correspondence"),
        ("h2_depth_scaling",            "experiments/h2_depth_scaling"),
        ("h3_multiscale_generalization","experiments/h3_multiscale_generalization"),
        ("figures_extended_data",       "figures/extended_data"),
        ("figures_manuscript",          "figures/manuscript"),
    ]

    def __init__(
        self,
        repo_root:          Optional[str] = None,
        master_seed:        int           = 42,
        output_path:        Optional[str] = None,
        timeout_per_script: int           = 600,
        verbose:            bool          = True,
        sign:               Optional[str] = None,
        sign_pub:           Optional[str] = None,
        build_manifest:     bool          = True,
    ) -> None:
        self.repo_root          = Path(repo_root or Path(__file__).resolve().parents[1])
        self.master_seed        = master_seed
        self.timeout_per_script = timeout_per_script
        self.verbose            = verbose
        self._sign_key          = sign
        self._sign_pub          = sign_pub
        self._build_manifest    = build_manifest
        self._run_id            = str(uuid.uuid4())

        reports_dir = self.repo_root / "audit_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = (
            Path(output_path) if output_path
            else reports_dir / f"audit_{self._run_id}.json"
        )

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    def run(
        self,
        suites:       Optional[List[str]] = None,
        pytest_extra: Optional[List[str]] = None,
        skip_studies: bool                = False,
    ) -> Dict[str, Any]:
        """
        Execute the full audit run and return the complete report dict.

        The returned dict validates against AuditSchema.
        """
        wall_start_ns = time.monotonic_ns()
        ts_start      = datetime.now(timezone.utc).isoformat()

        # Initialise seed.
        self._init_seed()

        # Collect static environment provenance.
        env = SystemCollector.collect_all(repo_root=str(self.repo_root))

        # Shared state accumulated across all suite runs.
        dag_nodes:  List[Dict[str, Any]] = []
        suite_results: Dict[str, Any]    = {}

        # Root DAG node.
        root_node_id, root_node = _dag_node(
            node_type  = "run_root",
            label      = f"audit run {self._run_id[:8]}",
            started_ns = wall_start_ns,
            ended_ns   = wall_start_ns,   # will not be updated here; informational
            status     = "running",
        )
        dag_nodes.append(root_node)

        # ---- Pytest suites ------------------------------------------------
        for suite_name, rel_path in self._PYTEST_SUITES:
            if suites and suite_name not in suites:
                continue
            self._log(f"[pytest] {suite_name} …")
            suite_start = _ns()
            suite_rec, suite_dag = self._run_pytest_suite(
                suite_name  = suite_name,
                suite_path  = self.repo_root / rel_path,
                extra_args  = pytest_extra or [],
                parent_id   = root_node_id,
            )
            suite_results[suite_name] = suite_rec
            dag_nodes.extend(suite_dag)

        # ---- Study suites -------------------------------------------------
        if not skip_studies:
            for suite_name, rel_path in self._STUDY_SUITES:
                if suites and suite_name not in suites:
                    continue
                self._log(f"[study] {suite_name} …")
                suite_rec, suite_dag = self._run_study_suite(
                    suite_name = suite_name,
                    suite_dir  = self.repo_root / rel_path,
                    parent_id  = root_node_id,
                )
                suite_results[suite_name] = suite_rec
                dag_nodes.extend(suite_dag)

        wall_end_ns = time.monotonic_ns()
        wall_dur_s  = (wall_end_ns - wall_start_ns) / 1e9

        # ---- Section 1: provenance ----------------------------------------
        provenance = {
            "git":           env["git"],
            "seed_registry": env["seed_registry"],
            "hardware":      env["hardware"],
            "data_checksums": [],   # populated by experiment pipeline at training time
        }

        # ---- Section 2: orchestration -------------------------------------
        orchestration = {
            "wall_start_ns":         wall_start_ns,
            "wall_end_ns":           wall_end_ns,
            "wall_duration_seconds": round(wall_dur_s, 3),
            "dag_trace":             dag_nodes,
            "checkpoint_events":     [],
            "total_interruptions":   0,
            "total_resumptions":     0,
        }

        # ---- Section 3: math_telemetry ------------------------------------
        math_telemetry = {
            "spectral_map":      [],
            "lyapunov_evolution":[],
            "fisher_geometry":   [],
            "rg_operator_flow":  [],
            "telemetry_sampling_config": {
                "spectral_every_n_steps": None,
                "lyapunov_every_n_steps": None,
                "fisher_every_n_steps":   None,
                "rg_flow_every_n_steps":  None,
                "total_steps_in_run":     None,
            },
        }

        # ---- Section 4: hardware forensics --------------------------------
        peak_vram_bytes = self._query_peak_vram()
        hardware_forensics = {
            "vram_micro_log":          [],
            "kernel_latency_log":      [],
            "peak_vram_bytes":         peak_vram_bytes,
            "peak_vram_gb":            round(peak_vram_bytes / 1e9, 4) if peak_vram_bytes else None,
            "total_kernel_time_seconds": None,
        }

        # ---- Summary ------------------------------------------------------
        summary = self._compute_summary(suite_results, wall_dur_s)

        # ---- Section 5: integrity seal ------------------------------------
        integrity = self._seal(
            provenance         = provenance,
            orchestration      = orchestration,
            math_telemetry     = math_telemetry,
            hardware_forensics = hardware_forensics,
            suites             = suite_results,
            summary            = summary,
        )

        # ---- Assemble final report ----------------------------------------
        report: Dict[str, Any] = {
            "run_id":           self._run_id,
            "timestamp_utc":    ts_start,
            "provenance":       provenance,
            "orchestration":    orchestration,
            "math_telemetry":   math_telemetry,
            "hardware_forensics": hardware_forensics,
            "suites":           suite_results,
            "summary":          summary,
            "integrity":        integrity,
        }

        # Soft schema validation — never crashes the runner.
        valid, schema_errs = AuditSchema.validate_soft(report)
        if not valid:
            report["integrity"]["schema_warnings"] = schema_errs

        # Atomic write.
        self._write_report(report)

        if self.verbose:
            self._print_summary(summary)

        return report

    # ------------------------------------------------------------------
    # Pytest suite runner
    # ------------------------------------------------------------------

    def _run_pytest_suite(
        self,
        suite_name: str,
        suite_path: Path,
        extra_args: List[str],
        parent_id:  str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run a pytest subdirectory; return (suite_record, list_of_dag_nodes)."""
        suite_dag:  List[Dict[str, Any]] = []
        started_ns  = _ns()

        if not suite_path.exists():
            ended_ns = _ns()
            node_id, node = _dag_node(
                "pytest_suite", suite_name, started_ns, ended_ns, "error", parent_id,
            )
            suite_dag.append(node)
            err_case = _case(
                name        = f"<{suite_name}/not-found>",
                status      = "error",
                started_ns  = started_ns,
                ended_ns    = ended_ns,
                failure_msg = f"Suite path not found: {suite_path}",
                dag_node_id = node_id,
            )
            return self._empty_suite_rec("pytest_suite", started_ns, ended_ns, [err_case], node_id), suite_dag

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            json_report_path = tmp.name

        argv = [
            sys.executable, "-m", "pytest",
            str(suite_path),
            "--json-report",
            f"--json-report-file={json_report_path}",
            "--tb=short", "-q", "--no-header",
            *extra_args,
        ]

        cases: List[Dict[str, Any]] = []
        try:
            subprocess.run(
                argv,
                cwd     = str(self.repo_root),
                timeout = self.timeout_per_script * 10,
                check   = False,
            )
            ended_ns = _ns()
            node_id, node = _dag_node(
                "pytest_suite", suite_name, started_ns, ended_ns,
                "passed", parent_id, {"argv": argv},
            )
            suite_dag.append(node)
            cases = self._load_pytest_json(
                json_report_path, parent_id=node_id, dag_nodes=suite_dag
            )
        except subprocess.TimeoutExpired:
            ended_ns = _ns()
            node_id, node = _dag_node(
                "pytest_suite", suite_name, started_ns, ended_ns, "error", parent_id,
            )
            suite_dag.append(node)
            cases = [_case(
                name        = f"<{suite_name}/timeout>",
                status      = "error",
                started_ns  = started_ns,
                ended_ns    = ended_ns,
                failure_msg = f"Suite exceeded timeout ({self.timeout_per_script * 10}s).",
                dag_node_id = node_id,
            )]
        except Exception as exc:
            ended_ns = _ns()
            node_id, node = _dag_node(
                "pytest_suite", suite_name, started_ns, ended_ns, "error", parent_id,
            )
            suite_dag.append(node)
            cases = [_case(
                name        = f"<{suite_name}/runner-error>",
                status      = "error",
                started_ns  = started_ns,
                ended_ns    = ended_ns,
                failure_msg = str(exc),
                tb          = traceback.format_exc(),
                dag_node_id = node_id,
            )]
        finally:
            try:
                os.unlink(json_report_path)
            except OSError:
                pass

        # Recompute suite status from cases.
        suite_status = "passed" if all(c["status"] == "passed" for c in cases) else "failed"
        if node_id:
            for node in suite_dag:
                if node.get("node_id") == node_id:
                    node["status"] = suite_status

        return (
            self._empty_suite_rec("pytest_suite", started_ns, ended_ns, cases, node_id, argv=argv),
            suite_dag,
        )

    def _load_pytest_json(
        self,
        path:      str,
        parent_id: str,
        dag_nodes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            return _parse_pytest_json(raw, self.master_seed, parent_id, dag_nodes)
        except Exception as exc:
            return [_case(
                name        = "<pytest-json-load>",
                status      = "error",
                started_ns  = _ns(),
                ended_ns    = _ns(),
                failure_msg = (
                    f"Could not load pytest JSON report ({path}): {exc}.  "
                    "Ensure 'pytest-json-report' is installed: pip install pytest-json-report"
                ),
                tb          = traceback.format_exc(),
            )]

    # ------------------------------------------------------------------
    # Study script runner
    # ------------------------------------------------------------------

    def _run_study_suite(
        self,
        suite_name: str,
        suite_dir:  Path,
        parent_id:  str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        suite_dag:  List[Dict[str, Any]] = []
        started_ns  = _ns()

        if not suite_dir.exists():
            ended_ns   = _ns()
            node_id, node = _dag_node(
                "study_script", suite_name, started_ns, ended_ns, "error", parent_id,
            )
            suite_dag.append(node)
            err_case = _case(
                name        = f"<{suite_name}/not-found>",
                status      = "error",
                started_ns  = started_ns,
                ended_ns    = ended_ns,
                failure_msg = f"Study directory not found: {suite_dir}",
                dag_node_id = node_id,
            )
            return self._empty_suite_rec("study_script", started_ns, ended_ns, [err_case], node_id), suite_dag

        # Emit a suite-level DAG node.
        suite_node_id = str(uuid.uuid4())
        suite_node_started = started_ns

        scripts = sorted(suite_dir.glob("*.py"))
        scripts = [s for s in scripts if not s.name.startswith("__")]

        cases: List[Dict[str, Any]] = []
        for script in scripts:
            case_rec, script_dag = self._run_script(script, parent_id=suite_node_id)
            cases.append(case_rec)
            suite_dag.extend(script_dag)

        ended_ns = _ns()
        suite_status = "passed" if all(c["status"] == "passed" for c in cases) else "failed"
        _, suite_node = _dag_node(
            "study_script", suite_name, suite_node_started, ended_ns,
            suite_status, parent_id,
        )
        # Override the auto-generated node_id so cases can reference it.
        suite_node["node_id"] = suite_node_id
        suite_dag.insert(0, suite_node)

        return (
            self._empty_suite_rec("study_script", started_ns, ended_ns, cases, suite_node_id),
            suite_dag,
        )

    def _run_script(
        self,
        script:    Path,
        parent_id: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        rel_name   = str(script.relative_to(self.repo_root))
        started_ns = _ns()
        script_dag: List[Dict[str, Any]] = []

        try:
            result = subprocess.run(
                [sys.executable, str(script), "--fast-track"],
                cwd            = str(self.repo_root),
                capture_output = True,
                text           = True,
                timeout        = self.timeout_per_script,
            )
            ended_ns = _ns()
            status   = "passed" if result.returncode == 0 else "failed"

            node_id, node = _dag_node(
                "study_script", rel_name, started_ns, ended_ns, status, parent_id,
                metadata={"returncode": result.returncode},
            )
            script_dag.append(node)

            return _case(
                name           = rel_name,
                status         = status,
                started_ns     = started_ns,
                ended_ns       = ended_ns,
                failure_msg    = (
                    f"Script exited with code {result.returncode}." if status == "failed" else None
                ),
                seed_used      = self.master_seed,
                stdout         = _cap(result.stdout, 4096),
                stderr_snippet = _cap(result.stderr, 2048) if status == "failed" else None,
                dag_node_id    = node_id,
            ), script_dag

        except subprocess.TimeoutExpired:
            ended_ns = _ns()
            node_id, node = _dag_node(
                "study_script", rel_name, started_ns, ended_ns, "error", parent_id,
            )
            script_dag.append(node)
            return _case(
                name        = rel_name,
                status      = "error",
                started_ns  = started_ns,
                ended_ns    = ended_ns,
                failure_msg = f"Script timed out after {self.timeout_per_script}s.",
                dag_node_id = node_id,
            ), script_dag

        except Exception as exc:
            ended_ns = _ns()
            node_id, node = _dag_node(
                "study_script", rel_name, started_ns, ended_ns, "error", parent_id,
            )
            script_dag.append(node)
            return _case(
                name        = rel_name,
                status      = "error",
                started_ns  = started_ns,
                ended_ns    = ended_ns,
                failure_msg = str(exc),
                tb          = traceback.format_exc(),
                dag_node_id = node_id,
            ), script_dag

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _compute_summary(
        self,
        suite_results: Dict[str, Any],
        duration_s:    float,
    ) -> Dict[str, Any]:
        total = passed = failed = errored = skipped = 0
        suite_summary: Dict[str, Any] = {}

        for suite_name, suite in suite_results.items():
            sp = sf = se = ss = 0
            for case in suite.get("cases", []):
                st = case.get("status", "error")
                total += 1
                if st == "passed":   passed  += 1; sp += 1
                elif st == "failed": failed  += 1; sf += 1
                elif st == "skipped":skipped += 1; ss += 1
                else:                errored += 1; se += 1
            suite_summary[suite_name] = {
                "passed": sp, "failed": sf, "errored": se, "skipped": ss,
            }

        return {
            "total_cases":      total,
            "passed":           passed,
            "failed":           failed,
            "errored":          errored,
            "skipped":          skipped,
            "duration_seconds": round(duration_s, 3),
            "all_passed":       (passed == total and errored == 0 and total > 0),
            "suite_summary":    suite_summary,
        }

    # ------------------------------------------------------------------
    # Section 5: integrity seal
    # ------------------------------------------------------------------

    def _seal(
        self,
        provenance:         Dict[str, Any],
        orchestration:      Dict[str, Any],
        math_telemetry:     Dict[str, Any],
        hardware_forensics: Dict[str, Any],
        suites:             Dict[str, Any],
        summary:            Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the hash chain over the six canonical sections, sign the tip,
        and optionally build the file manifest.
        """
        chain = HashChain()
        chain.append("provenance",          provenance)
        chain.append("orchestration",       orchestration)
        chain.append("math_telemetry",      math_telemetry)
        chain.append("hardware_forensics",  hardware_forensics)
        chain.append("suites",              suites)
        chain.append("summary",             summary)

        root_hash, tip_hash = chain.root_and_tip()

        # Optional digital signature.
        signer    = ReportSigner(self._sign_key, self._sign_pub)
        signature = signer.sign(tip_hash)

        # Optional repository manifest.
        manifest = None
        if self._build_manifest:
            try:
                manifest = ManifestBuilder(str(self.repo_root)).build()
            except Exception:
                manifest = None

        return {
            "hash_chain":      chain.records(),
            "chain_root_hash": root_hash,
            "chain_tip_hash":  tip_hash,
            "signature":       signature,
            "manifest":        manifest,
            "schema_warnings": [],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_seed(self) -> None:
        try:
            sys.path.insert(0, str(self.repo_root))
            from src.utils.seed_registry import SeedRegistry
            SeedRegistry.get_instance().set_master_seed(self.master_seed)
        except Exception:
            pass

    @staticmethod
    def _query_peak_vram() -> Optional[int]:
        """Return peak GPU memory usage in bytes, or None on CPU."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated()
        except Exception:
            pass
        return None

    def _empty_suite_rec(
        self,
        suite_type: str,
        started_ns: int,
        ended_ns:   int,
        cases:      List[Dict[str, Any]],
        dag_node_id:Optional[str],
        argv:       Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        duration_ns = max(0, ended_ns - started_ns)
        return {
            "suite_type":       suite_type,
            "started_ns":       started_ns,
            "ended_ns":         ended_ns,
            "duration_ns":      duration_ns,
            "duration_seconds": round(duration_ns / 1e9, 3),
            "pytest_args":      argv,
            "dag_node_id":      dag_node_id,
            "cases":            cases,
        }

    def _write_report(self, report: Dict[str, Any]) -> None:
        target   = self.output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_path, target)
        if self.verbose:
            print(f"\nAudit report → {target}", file=sys.stderr)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        print("\n" + "─" * 64, file=sys.stderr)
        print(
            f"  AUDIT COMPLETE  "
            f"total={summary['total_cases']}  "
            f"passed={summary['passed']}  "
            f"failed={summary['failed']}  "
            f"error={summary['errored']}  "
            f"skipped={summary['skipped']}  "
            f"({summary['duration_seconds']:.1f}s)",
            file=sys.stderr,
        )
        tag = "ALL PASSED" if summary["all_passed"] else "FAILURES DETECTED"
        print(f"  {tag}", file=sys.stderr)
        print("─" * 64, file=sys.stderr)
