"""
Build the ParkVision AI dataset from PKLot.

Usage
-----
    python scripts/01_build_dataset.py

Produces
--------
    data/raw/scene_summary.csv                       exploratory statistics
    data/raw/images/                                 whole-lot photographs
    data/processed/{train,val,test}/{occupied,empty} 224x224 slot crops
    data/demo/                                       scenes for the web app
    results/dataset_report.json                      what was built, in numbers
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src import data_prep as D


def fetch_annotations() -> Path:
    """Download the PKLot annotation document unless it is already cached."""
    target = C.RAW_DIR / "samples.json"
    if target.exists() and target.stat().st_size > 200_000_000:
        print(f"[1/6] annotations already cached ({target.stat().st_size / 1e6:.0f} MB)")
        return target

    print(f"[1/6] downloading annotations from {C.HF_SAMPLES_URL}")
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(C.HF_SAMPLES_URL, headers={"User-Agent": D.USER_AGENT})
    started = time.time()
    with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as fh:
        total = int(response.headers.get("Content-Length", 0))
        got = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if got % (32 << 20) < (1 << 20):
                print(f"      {got / 1e6:.0f}/{total / 1e6:.0f} MB", flush=True)
    print(f"      done in {time.time() - started:.0f}s")
    return target


def stratified_take(scenes, assignment, limit: int):
    """
    Round-robin over (lot, weather, split) buckets so that capping the working
    set keeps every lot, every weather condition and every split represented
    instead of whichever bucket happens to sort first.
    """
    import random
    from collections import defaultdict

    buckets = defaultdict(list)
    for scene in scenes:
        split = assignment.get((scene["lot"], scene["date"]))
        if split is None:
            continue
        buckets[(scene["lot"], scene["weather"], split)].append(scene)

    rng = random.Random(C.RANDOM_SEED + 3)
    for group in buckets.values():
        rng.shuffle(group)

    keys = sorted(buckets)
    picked, cursor = [], 0
    while len(picked) < limit and any(buckets[k] for k in keys):
        key = keys[cursor % len(keys)]
        cursor += 1
        if buckets[key]:
            picked.append(buckets[key].pop())
    return picked


def main() -> None:
    started = time.time()
    C.ensure_dirs()

    samples_path = fetch_annotations()

    print("[2/6] indexing photographs and bay geometry")
    scenes = D.build_catalogue(samples_path)

    lots = Counter(s["lot"] for s in scenes)
    weather = Counter(s["weather"] for s in scenes)
    print(f"      lots:    {dict(lots)}")
    print(f"      weather: {dict(weather)}")

    print("[3/6] assigning whole (lot, date) groups to train / val / test")
    assignment = D.assign_groups(scenes)
    split_groups = Counter(assignment.values())
    print(f"      day-groups per split: {dict(split_groups)}")

    print(f"[4/6] reserving {C.DEMO_SCENES} whole-lot scenes for the dashboard")
    demo_scenes = D.reserve_demo_scenes(scenes, assignment)
    demo_files = {s["filepath"] for s in demo_scenes}
    print(f"      reserved {len(demo_scenes)} scenes from the test day-groups")

    print("[5/6] downloading the photographs we need")
    # Cap the working set: quotas are met long before every candidate is used.
    # Interleave (lot, weather, split) buckets so the cap keeps all three lots
    # and all three weather conditions rather than the alphabetically first one.
    working = stratified_take(
        [s for s in scenes if s["filepath"] not in demo_files], assignment, limit=1400
    )
    print(f"      working set: {len(working)} scenes, "
          f"lots={dict(Counter(s['lot'] for s in working))}, "
          f"weather={dict(Counter(s['weather'] for s in working))}")
    downloaded = D.download_scenes(working + demo_scenes)

    by_path = {s["filepath"]: s for s in downloaded}
    demo_ready = [by_path[s["filepath"]] for s in demo_scenes if s["filepath"] in by_path]
    train_pool = [s for s in downloaded if s["filepath"] not in demo_files]

    print("[6/6] cropping bays into the classification folder tree")
    counts = D.build_crops(train_pool, assignment)
    D.write_demo_index(demo_ready)

    total_crops = sum(v for split in counts.values() for v in split.values())
    print("\n=== dataset built ===")
    for split in C.SPLITS:
        row = counts[split]
        print(f"  {split:<5} empty={row['empty']:>5}  occupied={row['occupied']:>5}")
    print(f"  total slot crops: {total_crops:,}")
    print(f"  demo scenes:      {len(demo_ready)}")

    report = {
        "source_dataset": "PKLot (Almeida et al., 2015), Voxel51 mirror on Hugging Face",
        "photographs_indexed": sum(1 for _ in open(C.RAW_DIR / "scene_summary.csv", encoding="utf-8")) - 1,
        "candidate_scenes": len(scenes),
        "scenes_downloaded": len(downloaded),
        "lots": dict(lots),
        "weather": dict(weather),
        "day_groups_per_split": dict(split_groups),
        "crops_per_split": counts,
        "total_crops": total_crops,
        "demo_scenes": len(demo_ready),
        "image_size": list(C.IMG_SIZE),
        "split_ratios": {"train": C.TRAIN_RATIO, "val": C.VAL_RATIO, "test": C.TEST_RATIO},
        "split_strategy": "grouped by (lot, date) so no calendar day appears in two splits",
        "build_seconds": round(time.time() - started, 1),
    }
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {C.RESULTS_DIR / 'dataset_report.json'}  ({report['build_seconds']}s total)")


if __name__ == "__main__":
    main()
