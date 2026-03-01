"""
audit/collector.py

Environment provenance collector.

Gathers three categories of audit evidence at the start of every run:

  1. Git state    commit hash, branch, dirty-flag, tags
  2. Hardware     device type, GPU specs or CPU specs, Python/Torch versions
  3. Seed state   SeedRegistry snapshot confirming deterministic configuration

All methods return plain dicts that slot directly into the JSON report
produced by AuditRunner.  No external package beyond the standard library
and the project's existing dependencies is required; 'git' is called via
subprocess and the result is gracefully degraded to 'UNKNOWN' strings when
the repository metadata is unavailable.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def _run_git(*args: str, cwd: Optional[str] = None) -> str:
    """Run a git command and return stdout, or empty string on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


class SystemCollector:
    """
    Collects environment provenance for a single audit run.

    All public methods are stateless class-methods that can be called
    independently; the top-level ``collect_all`` method bundles every
    category into a single dict ready for JSON serialisation.
    """

    @classmethod
    def collect_git(cls, repo_root: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect git repository state.

        Parameters
        ----------
        repo_root:
            Absolute path to the repository root.  When None the current
            working directory is used.

        Returns
        -------
        dict with keys: commit_hash, branch, dirty, tags, remote_url.
        """
        commit_hash = _run_git("rev-parse", "HEAD", cwd=repo_root) or "UNKNOWN"
        branch      = _run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo_root) or "UNKNOWN"

        # Dirty-flag: non-empty output means uncommitted changes exist.
        dirty_out   = _run_git("status", "--porcelain", cwd=repo_root)
        dirty       = bool(dirty_out)

        # Tags pointing at HEAD.
        tags_out    = _run_git("tag", "--points-at", "HEAD", cwd=repo_root)
        tags: List[str] = [t for t in tags_out.splitlines() if t] if tags_out else []

        remote_url  = _run_git(
            "remote", "get-url", "origin", cwd=repo_root
        ) or None

        return {
            "commit_hash": commit_hash,
            "branch":      branch,
            "dirty":       dirty,
            "tags":        tags,
            "remote_url":  remote_url,
        }

    @classmethod
    def collect_hardware(cls) -> Dict[str, Any]:
        """
        Collect hardware and software version information.

        Returns
        -------
        dict with device_type, device_name, CUDA specs (when available),
        CPU specs, and Python/Torch/NumPy version strings.
        """
        import torch

        info: Dict[str, Any] = {
            "python_version": platform.python_version(),
            "torch_version":  torch.__version__,
            "platform":       platform.platform(),
        }

        # NumPy version (optional but typically present).
        try:
            import numpy as np
            info["numpy_version"] = np.__version__
        except ImportError:
            info["numpy_version"] = "unavailable"

        # CPU baseline (always populated).
        import os
        info["cpu_count"] = os.cpu_count() or 0
        info["cpu_model"] = platform.processor() or _cpu_model_linux()

        # Device detection follows DeviceManager priority order.
        if torch.cuda.is_available():
            idx   = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info.update({
                "device_type":           "cuda",
                "device_name":           props.name,
                "total_memory_gb":       round(props.total_memory / 1_073_741_824, 2),
                "multi_processor_count": props.multi_processor_count,
                "compute_capability":    f"{props.major}.{props.minor}",
                "cuda_version":          torch.version.cuda or "UNKNOWN",
                "cudnn_version":         (
                    str(torch.backends.cudnn.version())
                    if torch.backends.cudnn.is_available()
                    else None
                ),
            })
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info.update({
                "device_type":           "mps",
                "device_name":           "Apple Silicon MPS",
                "total_memory_gb":       None,
                "multi_processor_count": None,
                "compute_capability":    None,
                "cuda_version":          None,
                "cudnn_version":         None,
            })
        else:
            info.update({
                "device_type":           "cpu",
                "device_name":           info["cpu_model"],
                "total_memory_gb":       None,
                "multi_processor_count": None,
                "compute_capability":    None,
                "cuda_version":          None,
                "cudnn_version":         None,
            })

        return info

    @classmethod
    def collect_seed(cls) -> Dict[str, Any]:
        """
        Snapshot the SeedRegistry singleton state.

        The registry is expected to have had ``set_master_seed`` called
        before this method is invoked (as is done in AuditRunner.run()).

        Returns
        -------
        dict with master_seed, step, deterministic_cuda, benchmark_disabled,
        and the worker_seed_formula string for documentation purposes.
        """
        try:
            import torch
            from src.utils.seed_registry import SeedRegistry

            reg = SeedRegistry.get_instance()
            return {
                "master_seed":        reg.master_seed,
                "step":               reg.step,
                "deterministic_cuda": bool(torch.backends.cudnn.deterministic),
                "benchmark_disabled": not bool(torch.backends.cudnn.benchmark),
                "worker_seed_formula": (
                    "(master_seed * 2654435761 + worker_id * 1013904223) & 0xFFFFFFFF"
                ),
            }
        except Exception as exc:
            # SeedRegistry unavailable (e.g., running outside project root).
            return {
                "master_seed":        None,
                "step":               0,
                "deterministic_cuda": False,
                "benchmark_disabled": False,
                "worker_seed_formula": "SeedRegistry unavailable",
                "_error":             str(exc),
            }

    @classmethod
    def collect_all(cls, repo_root: Optional[str] = None) -> Dict[str, Any]:
        """
        Return a single dict containing git, hardware, and seed sections.

        This is the convenience entry-point used by AuditRunner.
        """
        return {
            "git":           cls.collect_git(repo_root=repo_root),
            "hardware":      cls.collect_hardware(),
            "seed_registry": cls.collect_seed(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cpu_model_linux() -> str:
    """
    Extract 'model name' from /proc/cpuinfo on Linux.
    Returns empty string on all other platforms.
    """
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""
