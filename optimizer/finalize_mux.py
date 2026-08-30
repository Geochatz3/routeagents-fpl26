# Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
# Copyright (C) 2026, Georgios Chatzitsompanis.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache-2.0

"""Provides stateless helpers for final-candidate selection and artifact
integrity.

Atomic writes expose either the previous artifact or the complete replacement.
Shipping manifests record size and MD5 as the authoritative on-disk identity.
Trusted candidate digests may skip reopening only after stronger registration
checks, with ZIP sanity, size, and MD5 verification. Path guards ensure
published artifacts remain within the output tree. Run-state-dependent
registration and selection remain on ``DCPOptimizer``.
"""

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional


def _path_is_in_submission(path) -> bool:
    """Return True when ``path`` looks like a write target inside the
    submission tree.  Used by the QoR capture path to refuse JSON
    targets under ``submission/``.

    Lightweight string check (does not resolve symlinks).
    """
    s = str(path).replace("\\", "/").lower()
    return "/submission/" in s or s.endswith("/submission") or s.startswith("submission/")


def _atomic_copy(src, dst) -> None:
    """Copy ``src`` → ``dst`` atomically from a reader's point of view.

    Writes to a temp file in ``dst``'s own directory, fsyncs it, then
    ``os.replace``s it onto ``dst`` (an atomic rename on the same
    filesystem).  Guarantee: a concurrent reader/validator — or the state
    left on disk after a SIGKILL / budget-kill / power loss mid-copy —
    observes ``dst`` as EITHER its prior contents OR the complete new
    contents, never a truncated/partial DCP.  This is the never-fail
    discipline applied to the one artifact the contest actually scores:
    a half-written output DCP is an invalid submission (score 0); the
    prior direct ``shutil.copy2`` left exactly that window open.

    Design-property-agnostic: no design-name or feature branching — pure
    filesystem infrastructure that hardens every finalize/mirror path
    identically.

    Falls back to a direct ``shutil.copy2`` if the atomic path raises for
    any reason (e.g. ``os.replace`` across an unexpected filesystem
    boundary), so it is NEVER worse than the prior direct-copy behaviour.
    Raises the same way ``shutil.copy2`` would when ``src`` is unreadable,
    so existing callers' try/except contracts are preserved.
    """
    src = str(src)
    dst_path = Path(dst)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(
            prefix=dst_path.name + ".", suffix=".tmp",
            dir=str(dst_path.parent),
        )
        os.close(fd)
        shutil.copy2(src, tmp)  # raises if src is missing/unreadable
        # Best-effort durability: flush the temp file to stable storage
        # before the rename so a power loss can't expose an empty dst.
        try:
            with open(tmp, "rb") as _f:
                os.fsync(_f.fileno())
        except OSError:
            pass
        os.replace(tmp, str(dst_path))  # atomic on the same filesystem
        tmp = None
    except Exception:
        # Clean up the temp file (if any) and fall back to the prior
        # direct-copy behaviour — never regress below the status quo.
        if tmp is not None and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        shutil.copy2(src, str(dst_path))


# File existence alone does not prove that publication succeeded.
# Record size and MD5 after a successful finalize or emergency copy, then
# require both to match when verifying the published file. Corruption or
# unreadable identity data fails closed.

def _stream_md5(path, chunk_size: int = 1024 * 1024) -> str:
    """md5 of a file, streamed (DCPs are 100-300MB; ~1-2s, signal-safe)."""
    h = hashlib.md5()
    with open(str(path), "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _artifact_identity(path) -> Optional[dict]:
    """Return {"path", "size", "md5"} for an on-disk artifact, or None.

    None (not an exception) when the file is missing/empty/unreadable —
    callers in signal context must never crash on a hostile filesystem.
    """
    try:
        p = Path(path)
        if not p.exists():
            return None
        size = p.stat().st_size
        if size <= 0:
            return None
        return {"path": str(p), "size": size, "md5": _stream_md5(p)}
    except Exception:
        return None


def _write_shipped_manifest(output_dcp, identity: dict) -> None:
    """Atomically writes an identity manifest beside the published artifact.

    The manifest records the artifact identity used to verify an in-flight file
    before emergency publication, preventing unrelated bytes at the same path
    from being accepted. This operation is best-effort and suppresses all
    exceptions.
    """
    try:
        mf = Path(str(output_dcp) + ".shipped.json")
        payload = dict(identity)
        payload["timestamp_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = mf.with_name(mf.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(str(tmp), str(mf))
    except Exception:
        pass


def _verify_shipped_identity(path, identity) -> tuple:
    """Verify the on-disk artifact still matches its recorded identity.

    Returns (ok: bool, reason: str).  Checks, cheapest first:
      1. file exists and size matches identity["size"];
      2. first bytes are the ZIP magic ("PK") — a Vivado DCP is a zip, so
         this independently rejects random-garbage injection even in the
         (tiny) window where garbage landed BEFORE the identity was
         recorded;
      3. streamed md5 matches identity["md5"].

    ANY exception (unreadable file, hostile bytes) returns (False, ...) —
    in the emergency handler a verification failure must fall through to
    the restore path, never crash the handler.
    """
    try:
        if not isinstance(identity, dict) or not identity.get("md5"):
            return False, "no_recorded_identity"
        p = Path(path)
        if not p.exists():
            return False, "output_missing"
        size = p.stat().st_size
        if size != identity.get("size"):
            return False, f"size_mismatch({size}!={identity.get('size')})"
        with open(str(p), "rb") as f:
            if f.read(2) != b"PK":
                return False, "not_a_zip_dcp"
        md5 = _stream_md5(p)
        if md5 != identity.get("md5"):
            return False, f"md5_mismatch({md5}!={identity.get('md5')})"
        return True, "ok"
    except Exception as e:  # pragma: no cover — hostile-fs defensive
        return False, f"verify_exception:{type(e).__name__}"


# Fully verified registrations store the candidate file's size and MD5.
# When enabled, finalization skips structural reopening only if the current
# file matches that identity. Missing, mismatched, or unreadable identity
# data falls back to structural validation.
# This avoids a redundant reopen under bounded tool time without relaxing
# registration gates. The feature defaults off, and FPL26_NO_MUX_MD5_TRUST
# takes precedence over FPL26_MUX_MD5_TRUST.


def mux_md5_trust_enabled() -> bool:
    """FPL26_MUX_MD5_TRUST master flag (DEFAULT OFF).  Kill switch
    FPL26_NO_MUX_MD5_TRUST wins, per house convention."""
    if (os.environ.get("FPL26_NO_MUX_MD5_TRUST", "")
            .strip().lower() in ("1", "true", "on", "yes")):
        return False
    return (os.environ.get("FPL26_MUX_MD5_TRUST", "")
            .strip().lower() in ("1", "true", "on", "yes"))


def dcp_zip_sane(path) -> bool:
    """Cheap container-integrity check on a DCP (a DCP is a zip): local
    file header magic PK\\x03\\x04 at byte 0 AND an End-Of-Central-Directory
    signature in the final 66 KiB.  Rationale: the md5-trust
    skip must never ship bytes no code path ever opened — a truncated or
    corrupt store has a valid digest of its broken self, and this check is
    the ~free proxy for "would open".  False -> caller falls back to the
    full structural-validate path (fail closed).  NOT a zip parser: crc /
    entry walks stay the full validator's job."""
    try:
        p = Path(path)
        size = p.stat().st_size
        if size < 100:
            return False
        with open(p, "rb") as f:
            if f.read(4) != b"PK\x03\x04":
                return False
            tail_len = min(66 * 1024, size)
            f.seek(size - tail_len)
            tail = f.read(tail_len)
        return b"PK\x05\x06" in tail
    except Exception:
        return False


def mux_md5_digest(path):
    """(md5_hexdigest, size_bytes) of a candidate file, or None on ANY
    failure (missing file, IO error) — the caller falls back to the full
    structural-validate path, never trusts a candidate it cannot hash.
    Streams via _stream_md5 (1 MiB chunks; a DCP-sized hash costs
    seconds, vs the minutes-class open_checkpoint it replaces)."""
    try:
        p = Path(path)
        size = p.stat().st_size
        if size <= 0:
            return None
        return (_stream_md5(p), int(size))
    except Exception:
        return None
