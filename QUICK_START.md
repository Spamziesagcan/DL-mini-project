# Resume Screening CLI

This project is now centered on a clean CLI workflow with:

- `CNN` and `LSTM` resume-vs-job scoring
- explainable skill overlap output
- generated graphs for scores and skill alignment
- JSON export for later analysis

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## One-Time Warmup

Warmup creates reusable local artifacts in `artifacts/` so later runs are faster.

```bash
python main.py warmup
```

## Score One Resume

```bash
python main.py match ^
  --resume BansariNaik-thirdyr.pdf ^
  --jd-text "Full stack developer with Python, React, SQL, Docker, Git, and REST APIs" ^
  --output-dir outputs
```

Generated output:

- terminal score summary
- matched and missing skills
- recommendation text
- `outputs/*_score_summary.png`
- `outputs/*_skill_alignment.png`

## Compare Multiple Resumes

```bash
python main.py compare ^
  --resumes BansariNaik-thirdyr.pdf "ChaitanyaShah_Resume (1).pdf" ^
  --jd-text "Full stack developer with Python, React, Node.js, SQL, Docker, Git, and REST APIs" ^
  --output-dir outputs ^
  --json-out outputs\comparison.json
```

Generated output:

- ranked candidate list in the terminal
- per-candidate graphs
- `outputs/comparison_scores.png`
- structured JSON export

## Use a Job Description File

```bash
python main.py match ^
  --resume BansariNaik-thirdyr.pdf ^
  --jd-file job_description.txt ^
  --output-dir outputs
```

## Run Tests

```bash
python -m unittest discover -s tests -v
```

## CLI Commands

```bash
python main.py --help
python main.py match --help
python main.py compare --help
python main.py warmup --help
```
