"""Provenance helper: stamp every result JSON with git commit, full config, and library versions.

Result JSONs are the source of truth for this project (skip-if-done keys on them, not on
checkpoints), so each must record exactly how it was produced. Merge the returned dict into
a result under the '_provenance' key -- it is purely additive and never collides with the
scientific metric keys.
"""
import os
import platform
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=_REPO_ROOT)
        return out.stdout.strip() if out.returncode == 0 else "uncommitted"
    except Exception:
        return "unknown"


def _versions():
    vers = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "numpy", "sklearn", "matplotlib"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            vers[mod] = "absent"
    return vers


def provenance(config=None, **extra):
    """Return a provenance block to merge into a result JSON under '_provenance'.

    config: the full config dict the run was produced from.
    extra:  per-run scalars (p_repeat, seed, model_type, ...).
    """
    block = {"git_commit": _git_commit(), "versions": _versions(), "config": config}
    block.update(extra)
    return block


if __name__ == "__main__":  # runnable check for the subprocess/version logic
    b = provenance(config={"smoke": True}, seed=0)
    assert set(b) >= {"git_commit", "versions", "config", "seed"}, b
    assert "python" in b["versions"] and "torch" in b["versions"], b
    print("provenance OK:", b["git_commit"], b["versions"]["python"], b["versions"]["torch"])
