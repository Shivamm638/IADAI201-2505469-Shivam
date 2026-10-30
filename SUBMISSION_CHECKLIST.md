# ParkVision AI — Final Submission Checklist

This checklist maps the current repository to the CRS Machine Learning and Deep Learning summative-assessment requirements for Scenario 1.

## Repository contents already present

- [x] Streamlit application: `app.py`
- [x] Model and supporting files: `models/parkvision_mobilenetv2.keras`, `models/layouts.json`
- [x] Jupyter notebook: `notebooks/ParkVision_AI_end_to_end.ipynb`
- [x] Data preparation, training, evaluation and robustness scripts: `scripts/`
- [x] Processed dataset with train, validation and test splits: `data/processed/`
- [x] Dataset provenance and attribution: `data/README.md`
- [x] Metrics, classification report, confusion matrix and training plots: `results/`
- [x] App/demo screenshots: `results/screenshots/`
- [x] Model card and technical documentation: `models/model_card.md`, `docs/`
- [x] Python dependencies: `requirements.txt`

## Final items that still need the student's details or an external action

- [ ] Add the student's full name to `submission/student_details.json`.
- [ ] Add the candidate registration number.
- [ ] Confirm the school name.
- [ ] Deploy the app on Streamlit Community Cloud and add its real public URL. The README currently has a placeholder live-app URL; do not submit that placeholder as a working link.
- [ ] Regenerate `submission/SA1_Submission_ParkVision_AI.pdf` after completing the student details and deployment URL.
- [ ] Give `ai.assignments@wacpinternational.org` access to the GitHub repository (GitHub Settings → Collaborators → Add people), unless public access is accepted by the assessor.
- [ ] Open the deployed app in a logged-out/incognito browser and test an image upload and the demo examples.
- [ ] Open the final PDF and confirm that all personal details and both project links are correct before uploading it to the assignment portal.

## Rebuild the submission PDF

After filling in the remaining values in `submission/student_details.json`, run:

```bash
python scripts/05_make_submission.py
python scripts/07_verify_submission.py
```

The helper `scripts/08_set_student_details.py` can also collect the missing values and rebuild the PDF. See its help output with:

```bash
python scripts/08_set_student_details.py --help
```

## Important notes

- The repository name is currently `ParkVision-AI`. The assignment gives `IADAI201(Student_id)-studentname` as an example naming format; changing the repository name is optional only if your teacher requires that exact convention.
- The README and model card report measured evaluation results from the committed result files. Do not replace them with estimated or invented metrics.
- The full original PKLot source photographs and annotation document are not committed; the repository's data documentation explains the processed dataset and how the source data can be rebuilt.
