"""
Pre-submission check.

Runs through the assignment's checklist of evidence and the repository
checklist, and reports anything missing — so the submission is verified rather
than assumed.

    python scripts/07_verify_submission.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C

OK, WARN, FAIL = "PASS", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "", warn_only: bool = False) -> bool:
    status = OK if condition else (WARN if warn_only else FAIL)
    results.append((status, name, detail))
    return condition


def file_check(name: str, path: Path, min_bytes: int = 1, warn_only: bool = False) -> bool:
    exists = path.exists() and path.stat().st_size >= min_bytes
    size = f"{path.stat().st_size / 1024:.0f} KB" if path.exists() else "missing"
    return check(name, exists, f"{path.relative_to(C.PROJECT_ROOT)} — {size}", warn_only)


# ==========================================================================
def check_code():
    print("\n— Code and scripts —")
    for rel in ["app.py", "requirements.txt", "packages.txt", ".gitignore",
                ".streamlit/config.toml"]:
        file_check(rel, C.PROJECT_ROOT / rel)

    for module in ["config", "slots", "data_prep", "dataset", "model",
                   "inference", "insights", "overlay", "layouts"]:
        file_check(f"src/{module}.py", C.PROJECT_ROOT / "src" / f"{module}.py")

    scripts = sorted((C.PROJECT_ROOT / "scripts").glob("*.py"))
    check("scripts present", len(scripts) >= 6, f"{len(scripts)} scripts")

    notebooks = list((C.PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    check("Jupyter notebook (.ipynb)", bool(notebooks),
          notebooks[0].name if notebooks else "none found")

    if notebooks:
        nb = json.loads(notebooks[0].read_text(encoding="utf-8"))
        executed = sum(1 for c in nb.get("cells", [])
                       if c.get("cell_type") == "code" and c.get("outputs"))
        total = sum(1 for c in nb.get("cells", []) if c.get("cell_type") == "code")
        check("notebook cells executed", executed > 0,
              f"{executed}/{total} code cells have stored output", warn_only=True)

    file_check("test suite", C.PROJECT_ROOT / "tests" / "test_parkvision.py")


def check_dataset():
    print("\n— Dataset —")
    total = 0
    balanced = True
    for split in C.SPLITS:
        counts = {cls: len(list(C.split_dir(split, cls).glob("*.jpg")))
                  for cls in C.CLASS_NAMES}
        total += sum(counts.values())
        balanced &= counts["empty"] == counts["occupied"]
        check(f"data/processed/{split}", all(v > 0 for v in counts.values()),
              ", ".join(f"{k}={v:,}" for k, v in counts.items()))

    per_class = total // len(C.CLASS_NAMES)
    check("at least 100 images per class (brief)", per_class >= 100,
          f"{per_class:,} per class")
    check("splits are class balanced", balanced)

    manifest = C.PROCESSED_DIR / "manifest.json"
    if file_check("crop manifest", manifest):
        records = json.loads(manifest.read_text(encoding="utf-8"))
        groups: dict = {}
        for record in records:
            groups.setdefault((record["lot"], record["date"]), set()).add(record["split"])
        leaks = [k for k, v in groups.items() if len(v) > 1]
        check("no calendar day leaks across splits", not leaks,
              f"{len(groups)} (lot, date) groups, {len(leaks)} leaking")

    demo = list(C.DEMO_DIR.glob("*.jpg"))
    check("demo scenes for the app", len(demo) >= 5, f"{len(demo)} scenes")
    file_check("demo ground truth", C.DEMO_INDEX_PATH)


def check_model():
    print("\n— Model —")
    file_check("trained model", C.MODEL_PATH, min_bytes=100_000)
    file_check("class index", C.CLASS_INDEX_PATH)
    file_check("camera layouts", C.MODELS_DIR / "layouts.json")

    if C.MODEL_PATH.exists():
        mb = C.MODEL_PATH.stat().st_size / 1e6
        check("model under GitHub's 100 MB file limit", mb < 100, f"{mb:.1f} MB")


def check_results():
    print("\n— Results and evidence —")
    for name, rel in [
        ("metrics", "metrics.json"),
        ("classification report", "classification_report.txt"),
        ("confusion matrix", "confusion_matrix.png"),
        ("training curves", "training_curves.png"),
        ("ROC / PR curves", "roc_pr_curves.png"),
        ("threshold sweep", "threshold_sweep.png"),
        ("accuracy by slice", "accuracy_by_slice.png"),
        ("misclassified examples", "misclassified_examples.png"),
        ("augmentation examples", "augmentation_examples.png"),
        ("robustness report", "robustness_report.md"),
        ("robustness chart", "robustness_chart.png"),
        ("scene-level results", "scene_level_results.csv"),
        ("dataset report", "dataset_report.json"),
        ("training summary", "training_summary.json"),
    ]:
        file_check(name, C.RESULTS_DIR / rel)

    shots = list(C.SCREENSHOT_DIR.glob("*.png"))
    check("annotated demo screenshots", len(shots) >= 5, f"{len(shots)} images")

    if C.METRICS_PATH.exists():
        metrics = json.loads(C.METRICS_PATH.read_text(encoding="utf-8"))
        accuracy = metrics.get("test", {}).get("accuracy", 0)
        check("test accuracy reported", accuracy > 0, f"{accuracy * 100:.2f}%")
        check("per-weather breakdown", bool(metrics.get("by_weather")),
              ", ".join(metrics.get("by_weather", {})))
        check("per-lot breakdown", bool(metrics.get("by_lot")),
              ", ".join(metrics.get("by_lot", {})))


def check_docs():
    print("\n— Documentation —")
    readme = C.PROJECT_ROOT / "README.md"
    if file_check("README.md", readme, min_bytes=3000):
        text = readme.read_text(encoding="utf-8").lower()
        for topic, needles in [
            ("research findings", ["research", "almeida"]),
            ("academic references", ["references", "cvpr"]),
            ("data preparation", ["preprocessing", "augmentation"]),
            ("models and training parameters", ["mobilenetv2", "epochs"]),
            ("accuracy metrics", ["accuracy", "confusion matrix"]),
            ("screenshots", ["![", ".png"]),
            ("deployed link", ["streamlit.app"]),
        ]:
            check(f"README covers {topic}", all(n in text for n in needles))

    for doc in ["01_problem_definition", "02_data_preprocessing", "03_model_development",
                "04_insight_logic", "05_web_app", "06_deployment", "references"]:
        file_check(f"docs/{doc}.md", C.DOCS_DIR / f"{doc}.md")

    # Relative links must resolve, or the README is broken on GitHub.
    pattern = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
    broken = []
    for md in list(C.DOCS_DIR.glob("*.md")) + [C.PROJECT_ROOT / "README.md"]:
        if not md.exists():
            continue
        for _, target in pattern.findall(md.read_text(encoding="utf-8")):
            if target.startswith(("http", "#", "mailto:")):
                continue
            path_part = target.split("#")[0]
            if path_part and not (md.parent / path_part).exists():
                broken.append(f"{md.name} -> {target}")
    check("all relative links resolve", not broken,
          "; ".join(broken[:4]) + (f" (+{len(broken) - 4} more)" if len(broken) > 4 else ""))


def check_submission():
    print("\n— Submission document —")
    details_path = C.PROJECT_ROOT / "submission" / "student_details.json"
    pdf = C.PROJECT_ROOT / "submission" / "SA1_Submission_ParkVision_AI.pdf"

    file_check("submission PDF", pdf, min_bytes=2000)

    if details_path.exists():
        details = json.loads(details_path.read_text(encoding="utf-8"))
        outstanding = [k for k, v in details.items()
                       if isinstance(v, str) and v.startswith("FILL_IN")
                       and not k.startswith("_")]
        check("student details completed", not outstanding,
              ", ".join(outstanding) if outstanding else "all fields set",
              warn_only=True)
        check("CRS name is Artificial Intelligence",
              details.get("crs_name") == "Artificial Intelligence")
        check("assessor access recorded",
              "ai.assignments@wacpinternational.org" in str(details.values()))


def main() -> int:
    print("=" * 74)
    print("  ParkVision AI — pre-submission verification")
    print("=" * 74)

    check_code()
    check_dataset()
    check_model()
    check_results()
    check_docs()
    check_submission()

    print("\n" + "=" * 74)
    width = max(len(name) for _, name, _ in results)
    for status, name, detail in results:
        mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
        print(f"[{mark}] {name:<{width}}  {detail}")

    passed = sum(1 for s, _, _ in results if s == OK)
    warned = sum(1 for s, _, _ in results if s == WARN)
    failed = sum(1 for s, _, _ in results if s == FAIL)

    print("=" * 74)
    print(f"  {passed} passed · {warned} warnings · {failed} failed")
    if failed:
        print("\n  Not ready to submit — resolve the FAIL items above.")
    elif warned:
        print("\n  Ready to submit once the warnings above are addressed.")
    else:
        print("\n  Everything checks out.")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
