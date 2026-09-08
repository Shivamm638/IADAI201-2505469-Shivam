"""
Fill in the candidate and deployment details, then rebuild the submission PDF.

The submission document needs six facts that only the student has. Hand-editing
`submission/student_details.json` works, but a stray comma silently breaks the
JSON and the failure surfaces two scripts later. This sets the fields, validates
them, and regenerates the PDF in one step.

    python scripts/08_set_student_details.py \
        --name "Your Name" \
        --registration "1000000" \
        --school "Your School" \
        --github-user "yourhandle" \
        --repo "IADAI201-1000000-YourName" \
        --app-url "https://your-app.streamlit.app"

Any flag left out is prompted for. Pass --skip-rebuild to only write the file.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C

DETAILS = C.PROJECT_ROOT / "submission" / "student_details.json"

FIELDS = [
    ("student_full_name", "name", "Your full name"),
    ("candidate_registration_number", "registration", "Candidate registration number"),
    ("school_name", "school", "School name"),
    ("github_username", "github_user", "GitHub username"),
    ("github_repository", "repo", "Repository name (IADAI201<student-id>-<name>)"),
    ("streamlit_app_url", "app_url", "Live Streamlit app URL"),
]


def warn(message: str) -> None:
    print(f"  ! {message}")


def validate(key: str, value: str) -> None:
    """Warn about values that will not work, without blocking the write."""
    if key == "github_username" and not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", value):
        warn(f"'{value}' is not a valid GitHub username — links to it will 404.")

    if key == "github_repository":
        if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
            warn(f"'{value}' contains characters GitHub does not allow in a repository name.")
        elif not value.upper().startswith("IADAI201"):
            warn(f"The brief asks for 'IADAI201(Student_id)-studentname'; '{value}' does not match.")

    if key == "streamlit_app_url":
        if not value.startswith(("http://", "https://")):
            warn(f"'{value}' is missing the https:// prefix.")
        elif "streamlit.app" not in value:
            warn(f"'{value}' is not a streamlit.app address — check it is the deployed URL.")

    if key == "candidate_registration_number" and not any(ch.isdigit() for ch in value):
        warn(f"'{value}' contains no digits — is that the registration number?")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set candidate details and rebuild the submission PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for _, flag, help_text in FIELDS:
        parser.add_argument(f"--{flag.replace('_', '-')}", help=help_text)
    parser.add_argument("--skip-rebuild", action="store_true",
                        help="write the details file but do not regenerate the PDF")
    args = parser.parse_args()

    if not DETAILS.exists():
        print(f"error: {DETAILS} is missing.", file=sys.stderr)
        return 1

    details = json.loads(DETAILS.read_text(encoding="utf-8"))

    print(f"updating {DETAILS.relative_to(C.PROJECT_ROOT)}\n")
    changed = []
    for key, flag, prompt in FIELDS:
        value = getattr(args, flag, None)

        if value is None:
            current = str(details.get(key, ""))
            settled = current and not current.startswith("FILL_IN")
            shown = f" [{current}]" if settled else ""
            try:
                value = input(f"  {prompt}{shown}: ").strip()
            except EOFError:
                value = ""
            if not value:
                if settled:
                    continue                       # keep what is already there
                warn(f"{key} left unset.")
                continue

        value = value.strip()
        validate(key, value)
        if details.get(key) != value:
            details[key] = value
            changed.append(key)

    details.pop("_instructions", None)
    DETAILS.write_text(json.dumps(details, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")

    outstanding = [k for k, _, _ in FIELDS
                   if not str(details.get(k, "")).strip()
                   or str(details.get(k, "")).startswith("FILL_IN")]

    print()
    print(f"  updated {len(changed)} field(s)" if changed else "  no changes")
    if outstanding:
        print(f"  still unset: {', '.join(outstanding)}")

    if args.skip_rebuild:
        return 0

    print("\nrebuilding the submission document…\n")
    for script in ("05_make_submission.py", "06_make_readme.py"):
        result = subprocess.run([sys.executable, str(C.PROJECT_ROOT / "scripts" / script)])
        if result.returncode != 0:
            print(f"error: {script} failed.", file=sys.stderr)
            return result.returncode

    print("\nverifying…\n")
    subprocess.run([sys.executable, str(C.PROJECT_ROOT / "scripts" / "07_verify_submission.py")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
