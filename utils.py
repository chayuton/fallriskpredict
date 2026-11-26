"""Small helpers shared by the scripts (tables, JSON, filenames)."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def safe_name(s) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def _to_builtin(o):
    if isinstance(o, dict):
        return {str(k): _to_builtin(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_builtin(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _to_builtin(o.tolist())
    return o


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(obj), f, indent=2, ensure_ascii=False)


def pct(x, digits: int = 2) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}"


def mean_sd_pct(mean, sd, digits: int = 2) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "–"
    if sd is None or (isinstance(sd, float) and math.isnan(sd)):
        return f"{100 * mean:.{digits}f}"
    return f"{100 * mean:.{digits}f} ± {100 * sd:.{digits}f}"


def num(x, digits: int = 0) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    """Plain Markdown table without the optional 'tabulate' dependency."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)
