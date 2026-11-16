"""Small file helpers shared by the CLIs: scenario lists and CSV/JSON output."""

from __future__ import annotations

import csv
import glob
import json
import os
from typing import Iterable


def read_list_file(path: str) -> list[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def expand_scenario_inputs(inputs: Iterable[str]) -> list[str]:
    """Scenario YAML paths from files, .txt lists, directories or globs; absolute, de-duplicated, sorted."""
    out: list[str] = []
    for raw in inputs:
        p = os.path.expanduser(raw)

        if os.path.isfile(p) and p.endswith(".txt"):
            out.extend(read_list_file(p))
            continue

        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "*.yaml"))))
            out.extend(sorted(glob.glob(os.path.join(p, "*.yml"))))
            continue

        matches = sorted(glob.glob(p))
        if matches:
            out.extend(matches)
        else:
            out.append(p)

    out = [os.path.abspath(os.path.expanduser(p)) for p in out if p.endswith((".yaml", ".yml"))]
    out = sorted(list(dict.fromkeys(out)))
    return out


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
