from __future__ import annotations

import csv
import json
import math
import os
import re
import string
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_json(path: str | Path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=json_default)


def json_default(obj: Any):
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


def read_csv_df(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv_df(path: str | Path, df: pd.DataFrame) -> None:
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)


def safe_float(value, default=float("nan")):
    if value is None:
        return default
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return default


def file_exists(value: str | Path | None) -> bool:
    if not value:
        return False
    return Path(value).exists()


def normalize_label(label: str | None) -> str:
    if label is None:
        return "unknown"
    text = str(label).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(rf"[{re.escape(string.punctuation.replace('_', '').replace('-', ''))}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "unknown"


def load_label_mapping(path: str | Path) -> dict:
    raw = load_yaml(path)
    label_to_category = {}
    category_map = {}
    for category, payload in raw.items():
        vis = [normalize_label(x) for x in payload.get("visual_labels", [])]
        aud = [normalize_label(x) for x in payload.get("audio_labels", [])]
        category_map[category] = {"visual_labels": vis, "audio_labels": aud}
        for label in vis + aud:
            label_to_category[label] = category
    return {"raw": raw, "category_map": category_map, "label_to_category": label_to_category}


def label_category(label: str | None, mapping: dict) -> str:
    norm = normalize_label(label)
    return mapping["label_to_category"].get(norm, "unknown")


def label_compatibility(a: str | None, b: str | None, mapping: dict) -> Tuple[float, str]:
    na = normalize_label(a)
    nb = normalize_label(b)
    if na == "unknown" and nb == "unknown":
        return 0.15, "both_unknown"
    if na == nb:
        return 1.0, "exact_label"
    ca = label_category(a, mapping)
    cb = label_category(b, mapping)
    if ca != "unknown" and ca == cb:
        return 1.0, f"same_category:{ca}"
    if ca == "unknown" and cb == "unknown":
        if na and nb and (na in nb or nb in na):
            return 0.55, "substring_unknown"
        return 0.25, "both_unknown_time_only"
    if ca == "unknown" or cb == "unknown":
        if na and nb and (na in nb or nb in na):
            return 0.45, "partial_overlap_unknown"
        return 0.35, "one_unknown"
    return 0.10, f"category_mismatch:{ca}__{cb}"


def merge_label_score(a: str | None, b: str | None, mapping: dict) -> float:
    score, _ = label_compatibility(a, b, mapping)
    return score


def read_csv_safe(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def list_existing_candidates(root: str | Path, rel_patterns: Sequence[str]) -> List[Path]:
    base = Path(root)
    out = []
    for pattern in rel_patterns:
        out.extend(sorted(base.glob(pattern)))
    # Stable order, de-duplicate
    uniq = []
    seen = set()
    for p in out:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            uniq.append(p)
    return uniq


def first_existing(paths: Sequence[str | Path]) -> Optional[Path]:
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    return None


def groupby_rows(df: pd.DataFrame, key: str) -> Dict[str, pd.DataFrame]:
    if df.empty or key not in df.columns:
        return {}
    return {str(k): g.copy() for k, g in df.groupby(key, dropna=False)}


def maybe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def mean_median_std(values: Sequence[float]) -> dict:
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return {"mean": math.nan, "median": math.nan, "std": math.nan, "p10": math.nan, "p90": math.nan}
    s = pd.Series(clean)
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=0)),
        "p10": float(s.quantile(0.10)),
        "p90": float(s.quantile(0.90)),
    }


def sequence_edit_distance(seq_a: Sequence[str], seq_b: Sequence[str]) -> float:
    if not seq_a and not seq_b:
        return math.nan
    if not seq_a or not seq_b:
        return 1.0
    ratio = SequenceMatcher(a=list(seq_a), b=list(seq_b)).ratio()
    return float(max(0.0, min(1.0, 1.0 - ratio)))


def kendall_spearman_from_orders(visual_ranks: Sequence[int], audio_ranks: Sequence[int]) -> Tuple[float, float]:
    if len(visual_ranks) < 2 or len(audio_ranks) < 2:
        return math.nan, math.nan
    try:
        from scipy.stats import kendalltau, spearmanr

        kt = kendalltau(visual_ranks, audio_ranks, nan_policy="omit")
        sr = spearmanr(visual_ranks, audio_ranks, nan_policy="omit")
        return float(kt.correlation) if kt.correlation is not None else math.nan, float(sr.correlation) if sr.correlation is not None else math.nan
    except Exception:
        return math.nan, math.nan


def draw_text_block(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], lines: Sequence[str], fill=(255, 255, 255), font=None, line_gap=4):
    x, y = xy
    if font is None:
        font = ImageFont.load_default()
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += font.size + line_gap
    return y


def save_histogram_png(values: Sequence[float], path: str | Path, title: str, xlabel: str, bins: int = 24, width: int = 960, height: int = 540):
    ensure_dir(Path(path).parent)
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    img = Image.new("RGB", (width, height), (20, 20, 24))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle((40, 40, width - 30, height - 60), outline=(90, 90, 100), width=1)
    draw.text((48, 12), title, fill=(255, 255, 255), font=font)
    draw.text((48, height - 40), xlabel, fill=(220, 220, 220), font=font)
    if not values:
        draw.text((48, 80), "No valid values", fill=(220, 120, 120), font=font)
        img.save(path)
        return
    lo = min(values)
    hi = max(values)
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    hist = [0] * bins
    for v in values:
        idx = int((v - lo) / (hi - lo) * bins)
        if idx == bins:
            idx -= 1
        idx = max(0, min(bins - 1, idx))
        hist[idx] += 1
    max_count = max(hist) if hist else 1
    plot_left, plot_top, plot_right, plot_bottom = 60, 60, width - 40, height - 80
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top
    bar_w = plot_w / bins
    for i, count in enumerate(hist):
        x0 = plot_left + i * bar_w + 1
        x1 = plot_left + (i + 1) * bar_w - 2
        h = 0 if max_count == 0 else (count / max_count) * plot_h
        y0 = plot_bottom - h
        draw.rectangle((x0, y0, x1, plot_bottom), fill=(80, 170, 220))
    for i, frac in enumerate([0.0, 0.5, 1.0]):
        x = plot_left + frac * plot_w
        value = lo + frac * (hi - lo)
        draw.line((x, plot_bottom, x, plot_bottom + 5), fill=(180, 180, 180))
        label = f"{value:.2f}"
        draw.text((x - 10, plot_bottom + 8), label, fill=(200, 200, 200), font=font)
    draw.text((width - 160, 12), f"n={len(values)}", fill=(220, 220, 220), font=font)
    img.save(path)


def save_scatter_png(x_vals: Sequence[float], y_vals: Sequence[float], path: str | Path, title: str, xlabel: str, ylabel: str, width: int = 960, height: int = 540):
    ensure_dir(Path(path).parent)
    pairs = [(float(x), float(y)) for x, y in zip(x_vals, y_vals) if not math.isnan(float(x)) and not math.isnan(float(y))]
    img = Image.new("RGB", (width, height), (18, 18, 22))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle((60, 50, width - 40, height - 80), outline=(90, 90, 100), width=1)
    draw.text((48, 12), title, fill=(255, 255, 255), font=font)
    draw.text((width // 2 - 60, height - 40), xlabel, fill=(220, 220, 220), font=font)
    draw.text((8, height // 2), ylabel, fill=(220, 220, 220), font=font)
    if not pairs:
        draw.text((80, 100), "No valid points", fill=(220, 120, 120), font=font)
        img.save(path)
        return
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmin -= 1.0
        xmax += 1.0
    if ymin == ymax:
        ymin -= 1.0
        ymax += 1.0
    plot_left, plot_top, plot_right, plot_bottom = 60, 50, width - 40, height - 80
    for x, y in pairs:
        px = plot_left + (x - xmin) / (xmax - xmin) * (plot_right - plot_left)
        py = plot_bottom - (y - ymin) / (ymax - ymin) * (plot_bottom - plot_top)
        r = 3
        draw.ellipse((px - r, py - r, px + r, py + r), fill=(245, 170, 70))
    draw.text((width - 160, 12), f"n={len(pairs)}", fill=(220, 220, 220), font=font)
    img.save(path)


def collect_numeric(series: pd.Series) -> List[float]:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float).tolist()
    return [float(v) for v in vals]


def dataframe_to_records(df: pd.DataFrame) -> List[dict]:
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def sort_df(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    if not present:
        return df
    return df.sort_values(by=present, kind="stable").reset_index(drop=True)

