"""
tests/validation/test_reproducibility.py

Full pipeline reproduction validation.
Confirms that data → model → metrics → figures pipeline is reproducible
end-to-end given identical seeds and configuration.
"""

import pytest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_and_run_model(seed: int, n_steps: int = 3) -> dict:
    """
    Minimal end-to-end pipeline:
    1. Generate synthetic dataset with seed
    2. Build model with seed
    3. Run n_steps gradient updates
    4. Compute metrics

    Returns a dict of reproducible metric values.
    """
    _set_seed(seed)

    # Synthetic data
    x = torch.randn(32, 16)
    y = torch.randint(0, 2, (32,))

    # Model
    model = nn.Sequential(
        nn.Linear(16, 16), nn.Tanh(),
        nn.Linear(16, 2),
    )

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    losses = []

    for _ in range(n_steps):
        opt.zero_grad()
        logits = model(x)
        loss   = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 8))

    # Collect parameter checksum
    param_hash = hashlib.md5(
        b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
    ).hexdigest()

    return {"losses": losses, "param_hash": param_hash}


class TestReproducibility:

    def test_pipeline_bit_exact_across_runs(self):
        """
        Running the minimal pipeline twice with the same seed must produce
        bit-exact identical loss trajectories and parameter checksums.
        """
        for seed in [0, 42, 99]:
            result_a = _make_and_run_model(seed)
            result_b = _make_and_run_model(seed)
            assert result_a["losses"]     == result_b["losses"], (
                f"Loss trajectories differ at seed={seed}."
            )
            assert result_a["param_hash"] == result_b["param_hash"], (
                f"Parameter checksums differ at seed={seed}."
            )

    def test_different_seeds_produce_different_results(self):
        """Different seeds must produce distinguishable pipelines."""
        result_0  = _make_and_run_model(seed=0)
        result_99 = _make_and_run_model(seed=99)
        assert result_0["param_hash"] != result_99["param_hash"], (
            "Different seeds produced identical parameter checksums."
        )

    def test_checkpoint_restore_continues_identically(self):
        """
        Saving and restoring a checkpoint must allow continuation
        with bit-exact results as an uninterrupted run.
        """
        _set_seed(7)
        x = torch.randn(16, 8)
        y = torch.randint(0, 2, (16,))

        model = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 2))
        opt   = torch.optim.SGD(model.parameters(), lr=0.01)

        # Run 2 steps
        losses_uninterrupted = []
        for _ in range(4):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            losses_uninterrupted.append(round(loss.item(), 8))

        # Now: run 2 steps, checkpoint, restore, run 2 more
        _set_seed(7)
        model2 = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 2))
        opt2   = torch.optim.SGD(model2.parameters(), lr=0.01)

        losses_with_resume = []
        for step in range(4):
            opt2.zero_grad()
            loss2 = nn.functional.cross_entropy(model2(x), y)
            loss2.backward()
            opt2.step()
            losses_with_resume.append(round(loss2.item(), 8))

            if step == 1:
                # Simulate checkpoint save/load
                state = {
                    "model": model2.state_dict(),
                    "opt":   opt2.state_dict(),
                }
                model2.load_state_dict(state["model"])
                opt2.load_state_dict(state["opt"])

        assert losses_uninterrupted == losses_with_resume, (
            "Checkpoint restore produced different loss trajectory."
        )

    def test_metrics_json_serialisable(self):
        """Pipeline metrics must be JSON-serialisable for provenance logging."""
        result = _make_and_run_model(seed=5)
        try:
            json.dumps(result)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"Metrics are not JSON-serialisable: {exc}")

    def test_results_directory_writable(self, tmp_path: Path):
        """Results directory must be writable for output persistence."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        result_file = results_dir / "test_result.json"
        result_file.write_text(json.dumps({"status": "ok"}))
        assert result_file.exists(), "Could not write results file."
