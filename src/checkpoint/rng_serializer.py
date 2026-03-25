from __future__ import annotations
import pickle
from pathlib import Path
class RNGStateSerializer:
    def save(self, path: Path) -> None:
        from src.utils.seed_registry import SeedRegistry
        state = SeedRegistry.get_instance().snapshot_state()
        with open(Path(path) / "rng_state.pkl", "wb") as f:
            pickle.dump(state, f)
    def load(self, path: Path) -> None:
        from src.utils.seed_registry import SeedRegistry
        rng_file = Path(path) / "rng_state.pkl"
        if rng_file.exists():
            with open(rng_file, "rb") as f:
                state = pickle.load(f)
            SeedRegistry.get_instance().restore_state(state)