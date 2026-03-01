"""
audit/integrity.py

Cryptographic integrity primitives for audit report sealing.

Provides three capabilities:

  1. HashChain    SHA-256 block-hash chaining in the style described
                   in the audit schema (block_sha256 + previous_hash
                   hashed together to produce chain_hash).  Modifying
                   any block's content breaks all subsequent chain_hash
                   values and invalidates the tip.

  2. ReportSigner  Optional Ed25519 digital signature over the chain
                   tip hash.  Signing requires cryptography>=41.0 to be
                   installed; the signer degrades gracefully when absent,
                   leaving integrity.signature as null.

  3. ManifestBuilder  Walks a directory tree and records SHA-256 and
                   size for every file, producing the manifest block
                   required by the schema's integrity section.

All functions are pure with respect to the file system: they accept
dicts and return dicts; they do not write JSON themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_of_str(s: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _canonical_json(obj: Any) -> str:
    """
    Deterministic JSON serialisation for hashing.

    Uses sort_keys=True and separators without trailing spaces so that
    the same Python dict always produces the same byte sequence regardless
    of insertion order.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# 1. Hash chain
# ---------------------------------------------------------------------------

_GENESIS_PREVIOUS = "0" * 64


class HashChain:
    """
    Append-only SHA-256 hash chain.

    Each block seals one named section of the audit report.  The chain
    is initialised with a 64-zero genesis hash so that the first block's
    previous_hash is well-defined.

    Usage
    -----
        chain = HashChain()
        chain.append("provenance",         provenance_dict)
        chain.append("orchestration",      orchestration_dict)
        chain.append("math_telemetry",     math_dict)
        chain.append("hardware_forensics", hw_dict)
        chain.append("suites",             suites_dict)
        chain.append("summary",            summary_dict)

        records        = chain.records()       # list[dict] for integrity.hash_chain
        root, tip      = chain.root_and_tip()  # (chain_root_hash, chain_tip_hash)
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._previous_hash: str = _GENESIS_PREVIOUS

    def append(self, block_name: str, block_content: Any) -> str:
        """
        Hash *block_content* and extend the chain.

        Parameters
        ----------
        block_name:
            Logical name of the content block (e.g. 'provenance').
        block_content:
            Any JSON-serialisable object.

        Returns
        -------
        The chain_hash of the new block (to be stored as previous_hash
        by the next call).
        """
        block_sha256 = _sha256_of_str(_canonical_json(block_content))
        chain_input  = self._previous_hash + block_sha256
        chain_hash   = _sha256_of_str(chain_input)

        record: Dict[str, Any] = {
            "block_index":   len(self._records),
            "block_name":    block_name,
            "block_sha256":  block_sha256,
            "previous_hash": self._previous_hash,
            "chain_hash":    chain_hash,
        }
        self._records.append(record)
        self._previous_hash = chain_hash
        return chain_hash

    def records(self) -> List[Dict[str, Any]]:
        """Return the list of block_hash_record dicts for the JSON report."""
        return list(self._records)

    def root_and_tip(self) -> Tuple[str, str]:
        """
        Return (chain_root_hash, chain_tip_hash).

        chain_root_hash is the chain_hash of block 0.
        chain_tip_hash  is the chain_hash of the last block appended.
        """
        if not self._records:
            return _GENESIS_PREVIOUS, _GENESIS_PREVIOUS
        return self._records[0]["chain_hash"], self._records[-1]["chain_hash"]

    def verify(self) -> Tuple[bool, List[str]]:
        """
        Re-compute all chain hashes and compare against stored values.

        Returns (True, []) if the chain is intact; (False, [error ...]) otherwise.
        This method is intended for post-write verification, not for sealing.
        """
        errors: List[str] = []
        prev   = _GENESIS_PREVIOUS

        for rec in self._records:
            expected_chain = _sha256_of_str(prev + rec["block_sha256"])

            if rec["previous_hash"] != prev:
                errors.append(
                    f"Block {rec['block_index']} ({rec['block_name']}): "
                    f"previous_hash mismatch.  "
                    f"Expected {prev[:8]}..., got {rec['previous_hash'][:8]}..."
                )
            if rec["chain_hash"] != expected_chain:
                errors.append(
                    f"Block {rec['block_index']} ({rec['block_name']}): "
                    f"chain_hash mismatch.  "
                    f"Expected {expected_chain[:8]}..., got {rec['chain_hash'][:8]}..."
                )
            prev = rec["chain_hash"]

        return (len(errors) == 0), errors


# ---------------------------------------------------------------------------
# 2. Digital signer (Ed25519 via cryptography package)
# ---------------------------------------------------------------------------

class ReportSigner:
    """
    Signs the chain tip hash with an Ed25519 private key.

    Requires: pip install cryptography>=41.0

    If the cryptography package is not installed, sign() returns None and
    logs a warning rather than raising, so unsigned development runs are
    never blocked.

    Key management
    --------------
    For a development key pair, run once:

        python -c "
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import (
            Encoding, PublicFormat, PrivateFormat, NoEncryption
        )
        priv = Ed25519PrivateKey.generate()
        open('audit/signing.key', 'wb').write(
            priv.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        )
        open('audit/signing.pub', 'wb').write(
            priv.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
        )
        "

    Add audit/signing.key to .gitignore.  Commit audit/signing.pub.
    """

    def __init__(
        self,
        private_key_path: Optional[str] = None,
        public_key_path:  Optional[str] = None,
    ) -> None:
        self._priv_path = private_key_path
        self._pub_path  = public_key_path

    def sign(self, chain_tip_hash: str) -> Optional[Dict[str, Any]]:
        """
        Sign *chain_tip_hash* and return a digital_signature dict.

        Returns None if the cryptography package is absent or no key path
        was configured.
        """
        if not self._priv_path:
            return None
        priv_path = Path(self._priv_path)
        if not priv_path.exists():
            return None

        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            from cryptography.hazmat.primitives.serialization import (
                Encoding, PublicFormat,
            )

            with open(priv_path, "rb") as fh:
                priv_key = load_pem_private_key(fh.read(), password=None)

            payload_bytes   = chain_tip_hash.encode("utf-8")
            signature_bytes = priv_key.sign(payload_bytes)
            signed_at_ns    = time.monotonic_ns()

            # Fingerprint of the public key.
            pub_bytes   = priv_key.public_key().public_bytes(
                Encoding.DER, PublicFormat.SubjectPublicKeyInfo
            )
            fingerprint_raw = hashlib.sha256(pub_bytes).digest()
            fingerprint_hex = ":".join(
                f"{b:02X}" for b in fingerprint_raw
            )

            pub_pem_path = (
                str(Path(self._pub_path).relative_to(Path.cwd()))
                if self._pub_path else None
            )

            return {
                "algorithm":              "Ed25519",
                "public_key_fingerprint": fingerprint_hex,
                "public_key_pem_path":    pub_pem_path,
                "signed_payload_sha256":  _sha256_of_bytes(payload_bytes),
                "signature_hex":          signature_bytes.hex(),
                "signed_at_ns":           signed_at_ns,
            }

        except ImportError:
            return None
        except Exception:
            return None

    @staticmethod
    def verify(
        chain_tip_hash:  str,
        signature_dict:  Dict[str, Any],
        public_key_path: str,
    ) -> Tuple[bool, str]:
        """
        Verify a signature dict against the given public key.

        Returns (True, "OK") on success; (False, reason) on failure.
        """
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            from cryptography.hazmat.primitives.serialization import load_pem_public_key
            from cryptography.exceptions import InvalidSignature

            with open(public_key_path, "rb") as fh:
                pub_key = load_pem_public_key(fh.read())

            payload_bytes   = chain_tip_hash.encode("utf-8")
            signature_bytes = bytes.fromhex(signature_dict["signature_hex"])
            pub_key.verify(signature_bytes, payload_bytes)
            return True, "OK"

        except ImportError:
            return False, "cryptography package not installed"
        except InvalidSignature:
            return False, "signature verification failed  report may have been tampered with"
        except Exception as exc:
            return False, f"verification error: {exc}"


# ---------------------------------------------------------------------------
# 3. Manifest builder
# ---------------------------------------------------------------------------

class ManifestBuilder:
    """
    Builds the integrity manifest by hashing every source file in a
    directory tree.

    Only files whose extensions are in *include_extensions* are hashed.
    Default: Python source, YAML configs, Markdown docs, and style files.
    The audit_reports/ directory is always excluded.
    """

    DEFAULT_EXTENSIONS = {
        ".py", ".yaml", ".yml", ".md", ".txt",
        ".toml", ".cfg", ".cff", ".mplstyle", ".json",
        ".sh", ".def", ".dockerignore",
    }

    ALWAYS_EXCLUDE = {
        "audit_reports", "__pycache__", ".git",
        ".mypy_cache", ".pytest_cache", "*.egg-info",
    }

    def __init__(
        self,
        root:               str,
        include_extensions: Optional[set] = None,
    ) -> None:
        self._root       = Path(root)
        self._extensions = include_extensions or self.DEFAULT_EXTENSIONS

    def build(self) -> Dict[str, Any]:
        """
        Walk the repository tree and return the manifest block dict.

        Returns a dict matching the schema's integrity.manifest shape.
        """
        entries: List[Dict[str, Any]] = []
        t0 = time.monotonic_ns()

        for path in sorted(self._root.rglob("*")):
            if not path.is_file():
                continue

            # Skip excluded directories.
            parts = set(path.relative_to(self._root).parts)
            if parts & self.ALWAYS_EXCLUDE:
                continue

            # Extension filter.
            if path.suffix not in self._extensions:
                continue

            try:
                sha256     = self._hash_file(path)
                size_bytes = path.stat().st_size
                mtime_ns   = int(path.stat().st_mtime_ns)
                entries.append({
                    "relative_path": str(path.relative_to(self._root)),
                    "sha256":        sha256,
                    "size_bytes":    size_bytes,
                    "verified":      True,
                    "mtime_ns":      mtime_ns,
                })
            except OSError:
                pass

        manifest_hash = _sha256_of_str(_canonical_json(entries))

        return {
            "entries":        entries,
            "manifest_hash":  manifest_hash,
            "verified_at_ns": time.monotonic_ns(),
            "total_files":    len(entries),
            "total_bytes":    sum(e["size_bytes"] for e in entries),
        }

    @staticmethod
    def _hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def verify_manifest(
        manifest: Dict[str, Any],
        root:     str,
    ) -> Tuple[bool, List[str]]:
        """
        Re-hash every file in the manifest and compare against stored SHA-256 values.

        Returns (True, []) if all files match; (False, [error ...]) listing mismatches.
        """
        errors: List[str] = []
        root_path = Path(root)

        for entry in manifest.get("entries", []):
            file_path = root_path / entry["relative_path"]
            if not file_path.exists():
                errors.append(f"Missing: {entry['relative_path']}")
                continue
            actual = ManifestBuilder._hash_file(file_path)
            if actual != entry["sha256"]:
                errors.append(
                    f"Hash mismatch: {entry['relative_path']}  "
                    f"expected={entry['sha256'][:16]}...  actual={actual[:16]}..."
                )

        return (len(errors) == 0), errors
