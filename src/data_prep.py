"""
Step 2 of the brief: data collection and preprocessing.

Source data
-----------
PKLot (Almeida et al., 2015) - 12,417 photographs of three Brazilian parking
lots (PUCPR, UFPR04, UFPR05) taken every five minutes across sunny, overcast
and rainy days. Every photograph ships with a hand-drawn quadrilateral and an
occupied/empty label for each bay. We consume the Voxel51 mirror on the
Hugging Face Hub, which stores those annotations as one JSON document.

What this module produces
-------------------------
1. ``data/raw/scene_summary.csv``  - one row per photograph (lot, weather,
   timestamp, bay counts). Used for the exploratory statistics in the report.
2. ``data/raw/images/``            - the whole-lot photographs we actually use.
3. ``data/processed/{train,val,test}/{occupied,empty}/*.jpg`` - 224x224 slot
   crops, in the folder layout the brief asks for.
4. ``data/demo/``                  - whole-lot scenes reserved for the web app.

Leakage control
---------------
PKLot frames are five minutes apart, so two frames from the same lot on the
same day are almost identical. Splitting crops at random would leak an image's
near-twin into the test set and inflate accuracy. We therefore assign whole
(lot, date) groups to a split, so no calendar day is ever seen in more than one
split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from . import config as C
from .slots import crop_slot, polygon_area, polygon_to_pixels

USER_AGENT = "ParkVision-AI/1.0 (student project; PKLot via Voxel51 mirror)"

# Keep at most this many photographs per (lot, date, weather) bucket in memory.
# One or two frames per lot-day is plenty and avoids near-duplicate frames.
SCENES_PER_BUCKET = 2


# ==========================================================================
# 1. Streaming reader for the 267 MB annotation document
# ==========================================================================
def iter_samples(samples_path: Path) -> Iterator[dict]:
    """
    Yield each sample object from the annotation document without loading all
    267 MB of it into memory. The file is a single JSON object whose "samples"
    key holds one entry per photograph.
    """
    decoder = json.JSONDecoder()
    chunk_size = 1 << 20

    with open(samples_path, "r", encoding="utf-8") as handle:
        buffer = handle.read(chunk_size)
        start = buffer.index("[", buffer.index('"samples"')) + 1
        buffer = buffer[start:]
        pos = 0

        while True:
            # Skip separators, refilling the buffer when it runs dry.
            while True:
                if pos >= len(buffer):
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        return
                    buffer = chunk
                    pos = 0
                    continue
                char = buffer[pos]
                if char in " \t\r\n,":
                    pos += 1
                elif char == "]":
                    return
                else:
                    break

            # Grow the buffer until one complete object can be decoded.
            while True:
                try:
                    obj, end = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        return
                    buffer += chunk

            yield obj
            pos = end

            if pos > (4 << 20):          # keep the working buffer small
                buffer = buffer[pos:]
                pos = 0


def _weather_of(sample: dict) -> str:
    weather = sample.get("weather")
    if isinstance(weather, dict):
        return str(weather.get("label", "unknown")).lower()
    return "unknown"


def _date_of(sample: dict) -> str:
    date = sample.get("date")
    if isinstance(date, dict):
        return str(date.get("$date", ""))[:10]
    return ""


def _time_of(sample: dict) -> str:
    stamp = sample.get("parking_timestamp")
    if isinstance(stamp, dict):
        return str(stamp.get("$date", ""))[11:19]
    return ""


def _spaces_of(sample: dict) -> List[dict]:
    """Extract labelled bays, dropping any without usable geometry or a label."""
    container = sample.get("parking_spaces") or {}
    out: List[dict] = []
    for poly in container.get("polylines", []) or []:
        status = str(poly.get("occupancy_status", "unknown")).lower()
        if status == "occupied":
            label = "occupied"
        elif status in ("not occupied", "empty", "free"):
            label = "empty"
        else:
            continue                      # bay carries no ground truth
        points = poly.get("points") or []
        if not points or len(points[0]) < 4:
            continue
        out.append(
            {
                "space_id": int(poly.get("space_id") or poly.get("index") or 0),
                "points": points[0],
                "label": label,
            }
        )
    return out


# ==========================================================================
# 2. Build the scene catalogue
# ==========================================================================
def build_catalogue(samples_path: Path, verbose: bool = True) -> List[dict]:
    """
    Single pass over the annotation document.

    Writes ``scene_summary.csv`` for every photograph and returns a
    memory-bounded pool of candidate scenes (at most ``SCENES_PER_BUCKET`` per
    lot/date/weather bucket) together with their full bay geometry.
    """
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = C.RAW_DIR / "scene_summary.csv"

    pool: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    total = 0

    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["filepath", "lot", "weather", "date", "time",
             "n_spaces", "n_occupied", "occupancy_pct"]
        )

        for sample in iter_samples(samples_path):
            filepath = sample.get("filepath", "")
            if not filepath:
                continue
            lot = str(sample.get("source", "unknown")).lower()
            weather = _weather_of(sample)
            date = _date_of(sample)
            time_str = _time_of(sample)
            spaces = _spaces_of(sample)
            if not spaces:
                continue

            n_spaces = len(spaces)
            n_occupied = sum(1 for s in spaces if s["label"] == "occupied")
            writer.writerow(
                [filepath, lot, weather, date, time_str, n_spaces, n_occupied,
                 round(100.0 * n_occupied / n_spaces, 2)]
            )
            total += 1

            bucket = (lot, date, weather)
            kept = pool[bucket]
            if len(kept) < SCENES_PER_BUCKET:
                hours = {s["time"][:2] for s in kept}
                if not kept or time_str[:2] not in hours:
                    kept.append(
                        {
                            "filepath": filepath,
                            "lot": lot,
                            "weather": weather,
                            "date": date,
                            "time": time_str,
                            "spaces": spaces,
                        }
                    )

            if verbose and total % 2000 == 0:
                print(f"    parsed {total:,} photographs", flush=True)

    scenes = [scene for group in pool.values() for scene in group]
    scenes.sort(key=lambda s: (s["lot"], s["date"], s["time"]))
    if verbose:
        print(f"    {total:,} photographs indexed -> {len(scenes):,} candidate scenes kept")
    return scenes


# ==========================================================================
# 3. Split whole (lot, date) groups across train / val / test
# ==========================================================================
def assign_groups(scenes: List[dict], seed: int = C.RANDOM_SEED) -> Dict[Tuple[str, str], str]:
    """
    Deterministically map each ``(lot, date)`` group to a split, applying the
    70/15/15 ratio *within every lot* so all three lots appear in all splits.
    """
    by_lot: Dict[str, set] = defaultdict(set)
    for scene in scenes:
        by_lot[scene["lot"]].add(scene["date"])

    assignment: Dict[Tuple[str, str], str] = {}
    rng = random.Random(seed)

    for lot in sorted(by_lot):
        dates = sorted(by_lot[lot])
        rng.shuffle(dates)
        n = len(dates)
        n_train = max(1, round(n * C.TRAIN_RATIO))
        n_val = max(1, round(n * C.VAL_RATIO))
        # Guarantee a non-empty test group even for lots with very few days.
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1 if n - n_train > 1 else 0
        for i, date in enumerate(dates):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            assignment[(lot, date)] = split
    return assignment


def reserve_demo_scenes(
    scenes: List[dict],
    assignment: Dict[Tuple[str, str], str],
    n_demo: int = C.DEMO_SCENES,
    seed: int = C.RANDOM_SEED,
) -> List[dict]:
    """
    Pick whole-lot scenes for the Streamlit demo from the *test* groups only,
    spreading them over lots, weather and occupancy band so the dashboard can
    demonstrate low, moderate and high congestion.
    """
    candidates = [s for s in scenes if assignment.get((s["lot"], s["date"])) == "test"]
    rng = random.Random(seed + 7)

    buckets: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for scene in candidates:
        occupied = sum(1 for x in scene["spaces"] if x["label"] == "occupied")
        ratio = occupied / max(1, len(scene["spaces"]))
        band = "low" if ratio < 0.40 else ("moderate" if ratio <= 0.75 else "high")
        buckets[(scene["lot"], scene["weather"], band)].append(scene)

    keys = sorted(buckets)
    rng.shuffle(keys)
    chosen: List[dict] = []
    seen: set = set()

    while keys and len(chosen) < n_demo:
        progressed = False
        for key in list(keys):
            if len(chosen) >= n_demo:
                break
            group = buckets[key]
            if not group:
                keys.remove(key)
                continue
            scene = group.pop(rng.randrange(len(group)))
            progressed = True
            if scene["filepath"] in seen:
                continue
            seen.add(scene["filepath"])
            chosen.append(scene)
        if not progressed:
            break
    return chosen


# ==========================================================================
# 4. Download the photographs we actually need
# ==========================================================================
def _local_raw_path(filepath: str) -> Path:
    return C.RAW_DIR / "images" / Path(filepath).name


def download_scene(scene: dict, retries: int = 3) -> Optional[Path]:
    """Fetch one photograph from the Hugging Face mirror, skipping cached files."""
    target = _local_raw_path(scene["filepath"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1024:
        return target

    url = C.HF_BASE_URL + scene["filepath"]
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if len(payload) < 1024:
                raise IOError("suspiciously small response")
            target.write_bytes(payload)
            return target
        except Exception:
            if attempt == retries - 1:
                return None
    return None


def download_scenes(scenes: List[dict], workers: int = 12, verbose: bool = True) -> List[dict]:
    """Download every scene in parallel and return those that arrived intact."""
    ok: List[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (scene, path) in enumerate(zip(scenes, pool.map(download_scene, scenes)), 1):
            if path is not None:
                enriched = dict(scene)
                enriched["local_path"] = str(path)
                ok.append(enriched)
            if verbose and i % 50 == 0:
                print(f"    downloaded {i}/{len(scenes)} scenes", flush=True)
    if verbose:
        print(f"    {len(ok)}/{len(scenes)} scenes downloaded")
    return ok


# ==========================================================================
# 5. Crop bays into the classification folder tree
# ==========================================================================
def _crop_name(scene: dict, space_id: int) -> str:
    stem = Path(scene["filepath"]).stem
    digest = hashlib.md5(f"{scene['lot']}/{stem}/{space_id}".encode()).hexdigest()[:6]
    return f"{scene['lot']}_{scene['weather']}_{stem}_s{space_id:03d}_{digest}.jpg"


def build_crops(
    scenes: List[dict],
    assignment: Dict[Tuple[str, str], str],
    per_class: int = C.CROPS_PER_CLASS,
    verbose: bool = True,
) -> Dict[str, Dict[str, int]]:
    """
    Warp every annotated bay to 224x224 and write it into
    ``data/processed/<split>/<class>/``.

    Per-split quotas follow the 70/15/15 ratio and are filled class-balanced.
    Scenes are visited in a shuffled order and each scene contributes only a
    capped slice of its bays, so quotas fill with an even spread over lots,
    weather and time of day instead of being exhausted by a few crowded frames.

    The cap is raised over successive rounds. A single tight cap gives the best
    spread but can leave a quota short when a split has few scenes, or when its
    lots are nearly full and simply do not contain enough empty bays; relaxing
    the cap only for the classes that are still short tops the quota up without
    letting any one scene dominate the easy classes.
    """
    quotas = {
        "train": int(round(per_class * C.TRAIN_RATIO)),
        "val": int(round(per_class * C.VAL_RATIO)),
        "test": int(round(per_class * C.TEST_RATIO)),
    }
    written = {split: {cls: 0 for cls in C.CLASS_NAMES} for split in C.SPLITS}
    manifest: List[dict] = []
    emitted: set = set()
    rejected = {"flat": 0, "tiny": 0, "clipped": 0}

    rng = random.Random(C.RANDOM_SEED)
    ordered = [s for s in scenes if assignment.get((s["lot"], s["date"])) is not None]
    ordered.sort(key=lambda s: (s["weather"], s["lot"], s["date"], s["time"]))
    rng.shuffle(ordered)

    scenes_per_split: Dict[str, int] = defaultdict(int)
    for scene in ordered:
        scenes_per_split[assignment[(scene["lot"], scene["date"])]] += 1

    def quota_met() -> bool:
        return all(written[s][c] >= quotas[s] for s in C.SPLITS for c in C.CLASS_NAMES)

    for multiplier in (1.0, 2.5, 8.0, 1000.0):
        if quota_met():
            break
        for index, scene in enumerate(ordered, 1):
            split = assignment[(scene["lot"], scene["date"])]
            if all(written[split][cls] >= quotas[split] for cls in C.CLASS_NAMES):
                continue

            image = cv2.imread(scene["local_path"], cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]

            spaces = list(scene["spaces"])
            rng.shuffle(spaces)

            fair_share = quotas[split] / max(1, scenes_per_split[split])
            per_scene_cap = max(4, int(fair_share * multiplier))
            taken = {cls: 0 for cls in C.CLASS_NAMES}

            for space in spaces:
                label = space["label"]
                if written[split][label] >= quotas[split] or taken[label] >= per_scene_cap:
                    continue
                key = (scene["filepath"], space["space_id"])
                if key in emitted:
                    continue
                polygon = polygon_to_pixels(space["points"], width, height)
                if polygon_area(polygon) < C.MIN_SLOT_AREA:
                    rejected["tiny"] += 1
                    continue
                if (polygon[:, 0].min() < -2 or polygon[:, 1].min() < -2
                        or polygon[:, 0].max() > width + 2 or polygon[:, 1].max() > height + 2):
                    rejected["clipped"] += 1       # bay clipped by the frame edge
                    continue

                patch = crop_slot(image, polygon, C.IMG_SIZE)
                if float(patch.std()) < C.MIN_CROP_STD:
                    rejected["flat"] += 1     # featureless: carries no usable signal
                    continue

                out_path = C.split_dir(split, label) / _crop_name(scene, space["space_id"])
                cv2.imwrite(str(out_path), patch, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                emitted.add(key)
                written[split][label] += 1
                taken[label] += 1
                manifest.append(
                    {
                        "file": str(out_path.relative_to(C.PROJECT_ROOT)).replace("\\", "/"),
                        "split": split,
                        "label": label,
                        "lot": scene["lot"],
                        "weather": scene["weather"],
                        "date": scene["date"],
                        "time": scene["time"],
                        "source_image": scene["filepath"],
                        "space_id": space["space_id"],
                    }
                )

            if verbose and index % 150 == 0:
                print(f"    pass x{multiplier:g}  scene {index}/{len(ordered)} -> "
                      f"{ {s: dict(written[s]) for s in C.SPLITS} }", flush=True)

            if quota_met():
                break

    (C.PROCESSED_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if verbose:
        print(f"    rejected bays: {rejected}")
    return written


def write_demo_index(demo_scenes: List[dict]) -> None:
    """
    Copy demo photographs into ``data/demo/`` and record their bay geometry and
    ground truth so the app can show a like-for-like comparison.
    """
    C.DEMO_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for scene in demo_scenes:
        source = Path(scene["local_path"])
        target = C.DEMO_DIR / source.name
        if not target.exists():
            target.write_bytes(source.read_bytes())
        occupied = sum(1 for s in scene["spaces"] if s["label"] == "occupied")
        entries.append(
            {
                "image": target.name,
                "lot": scene["lot"],
                "weather": scene["weather"],
                "date": scene["date"],
                "time": scene["time"],
                "n_spaces": len(scene["spaces"]),
                "n_occupied_truth": occupied,
                "occupancy_pct_truth": round(100.0 * occupied / max(1, len(scene["spaces"])), 2),
                "spaces": [
                    {"space_id": s["space_id"], "points": s["points"], "label": s["label"]}
                    for s in scene["spaces"]
                ],
            }
        )
    entries.sort(key=lambda e: (e["lot"], e["date"], e["time"]))
    C.DEMO_INDEX_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")
