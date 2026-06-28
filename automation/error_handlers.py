#!/usr/bin/env python
from pathlib import Path


def _read_tail(path: Path, n=200000):
    if not path.exists():
        return ""
    txt = path.read_text(errors="ignore")
    return txt[-n:]


def _parse_incar(path: Path):
    d = {}
    if not path.exists():
        return d
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("!"):
            continue
        if "=" in s:
            k, v = s.split("=", 1)
            d[k.strip().upper()] = v.strip()
    return d


def _write_incar(path: Path, kv):
    existing = path.read_text(errors="ignore").splitlines() if path.exists() else []
    out = []
    seen = set()

    for line in existing:
        s = line.strip()
        if "=" in s and not s.startswith("#") and not s.startswith("!"):
            k = s.split("=", 1)[0].strip().upper()
            if k in kv:
                out.append(f"{k} = {kv[k]}")
                seen.add(k)
            else:
                out.append(line)
        else:
            out.append(line)

    for k, v in kv.items():
        if k not in seen:
            out.append(f"{k} = {v}")

    path.write_text("\n".join(out) + "\n")


def detect_and_patch(job_dir: Path, cfg):
    outcar_tail = _read_tail(job_dir / "OUTCAR")
    vaspout_tail = _read_tail(job_dir / "vasp.out")
    text = (outcar_tail + "\n" + vaspout_tail).upper()

    if not text.strip():
        return False

    rules = cfg.get("failure_rules", [])
    for rule in rules:
        pats = [p.upper() for p in rule.get("match", [])]
        if not pats:
            continue
        if all(p in text for p in pats):
            patch = rule.get("patch", {})
            incar_patch = {k.upper(): str(v) for k, v in patch.get("INCAR", {}).items()}
            if incar_patch:
                _write_incar(job_dir / "INCAR", incar_patch)
                return True

    return False
