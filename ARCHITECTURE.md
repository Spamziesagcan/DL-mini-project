# Resume Screening System - Architecture

**Project**: Resume Screening System using Deep Learning  
**Date**: April 22, 2026  
**Interface**: CLI only  
**Models**: LSTM and CNN for resume-to-job-description similarity

---

## Overview

This project is a command-line resume screening system.

It does one thing:

1. read a resume from PDF,
2. compare it against a job description,
3. run two deep learning models on the same input pair,
4. report the raw `LSTM` score and raw `CNN` score,
5. add explainable skill overlap output and graphs.

There is no frontend in the current architecture.

---

## Current Goal

The system is designed to test **only two models** on resumes:

- `LSTM`
- `CNN`

The numeric scores shown in the CLI come only from those two models.

The project also prints:

- matched skills
- missing skills
- resume-only skills
- recommendation text
- graphs

These are **explainability and reporting layers**. They do **not** change the raw LSTM or CNN score.

---

## Runtime Flow

```text
Resume PDF
  ->
PDF text extraction
  ->
text cleaning
  ->
tokenization + padding
  ->
same resume/JD pair sent to:
  - LSTM similarity model
  - CNN similarity model
  ->
raw sigmoid outputs
  ->
percent conversion
  ->
CLI report + graphs + JSON export
```

---

## CLI Architecture

The current entrypoint is:

- `main.py`

It forwards to:

- `cli.py`

Main commands:

- `python main.py warmup`
- `python main.py match --resume <pdf> --jd-text "..."`
- `python main.py compare --resumes <pdf1> <pdf2> --jd-text "..."`

### Command Roles

- `warmup`: builds reusable local artifacts for faster later runs
- `match`: tests one resume against one job description
- `compare`: tests multiple resumes against one job description and ranks them

---

## Code Structure

### Core Files

- `main.py`: CLI entrypoint
- `cli.py`: argument parsing and terminal output
- `pipeline.py`: end-to-end screening pipeline

### Preprocessing

- `preprocessing/pdf_text.py`: extract text from PDF using `pdfplumber`
- `preprocessing/text_cleaning.py`: lowercase, punctuation removal, tokenization, stopword filtering
- `preprocessing/tokenizer_pipeline.py`: custom tokenizer + padding utilities

### Models

- `models/lstm_similarity.py`: LSTM similarity model definition
- `models/cnn_text_classification.py`: CNN similarity model definition
- `models/similarity_runtime.py`: artifact loading, fallback bootstrap training, and prediction

### Explainability and Reporting

- `utils/skill_extraction.py`: rule-based skill extraction
- `utils/reporting.py`: explanation objects and recommendation logic
- `utils/visualization.py`: graph generation

### Tests

- `tests/test_cli.py`
- `tests/test_pipeline.py`
- `tests/test_reporting.py`

---

## Preprocessing Pipeline

### Step 1: PDF Extraction

Resume text is extracted from PDF pages with `pdfplumber`.

### Step 2: Cleaning

The text is normalized by:

- converting to lowercase
- removing punctuation
- tokenizing words
- removing stopwords

### Step 3: Tokenization

The project uses a lightweight custom tokenizer for runtime text preparation.

### Step 4: Padding

Sequences are padded or truncated to a fixed length.

Current runtime setting:

- `max_sequence_length = 60`

---

## LSTM Model

The LSTM model is a dual-input similarity model.

### Structure

```text
resume tokens -> shared embedding -> LSTM -> resume vector
jd tokens     -> shared embedding -> LSTM -> jd vector

resume vector + jd vector
  ->
concatenate
  ->
dense
  ->
sigmoid
  ->
similarity score
```

### Current Runtime Configuration

- embedding dimension: `16`
- LSTM units: `32`
- hidden dense units: `32`
- output: `1 sigmoid unit`

### Purpose

The LSTM is intended to capture sequential and contextual text patterns.

---

## CNN Model

The CNN model is also a dual-input similarity model.

### Structure

```text
resume tokens -> shared embedding -> Conv1D -> GlobalMaxPool -> Dense -> resume vector
jd tokens     -> shared embedding -> Conv1D -> GlobalMaxPool -> Dense -> jd vector

resume vector + jd vector
  ->
concatenate
  ->
dense
  ->
sigmoid
  ->
similarity score
```

### Current Runtime Configuration

- embedding dimension: `16`
- Conv1D filters: `32`
- kernel size: `5`
- encoder dense units: `32`
- merge dense units: `32`
- output: `1 sigmoid unit`

### Purpose

The CNN is intended to capture local lexical and phrase-level patterns.

---

## Scoring Logic

This is the most important point in the current project.

### Raw Model Scores

The CLI reports:

- `LSTM Score`
- `CNN Score`

Each one is computed as:

```text
sigmoid_output * 100
```

So if a model outputs `0.4912`, the displayed score is `49.12%`.

### Ensemble Score

The CLI also shows:

```text
Ensemble Score = (LSTM Score + CNN Score) / 2
```

This is only a reporting average. It is **not** a third model.

### Explainability Is Separate

These are not part of the raw neural score:

- matched skills
- missing skills
- overlap ratio
- recommendation text

They are computed after model prediction and are used for interpretation only.

---

## Artifact Strategy

The runtime first tries to load reusable local artifacts from:

- `artifacts/tokenizer.pkl`
- `artifacts/cnn_similarity_model.h5`
- `artifacts/lstm_similarity_model.h5`

If artifacts are missing, the system falls back to a lightweight bootstrap training process using synthetic text templates.

The purpose of `warmup` is to create those artifacts ahead of time so later CLI runs are faster.

---

## Synthetic Bootstrap Training

The fallback training path is intentionally small and fast.

### Current Bootstrap Settings

- sequence length: `60`
- embedding dimension: `16`
- epochs: `1`
- batch size: `16`
- negative ratio: `1.0`
- limited bootstrap role profiles: `6`

### Why This Exists

The project needs to stay runnable even if pretrained artifacts are not already available locally.

This fallback is for convenience, not for final-quality model calibration.

---

## Explainability Layer

After the raw model scores are produced, the system performs rule-based skill extraction on:

- the resume text
- the job description text

It then computes:

- matched skills
- missing skills
- resume-only skills
- overlap ratio

This layer helps interpret results, especially because the current raw model outputs are conservative.

---

## Graph Generation

For each run, the CLI can generate PNG charts:

### Per Resume

- score summary chart
  - LSTM
  - CNN
  - ensemble

- skill alignment chart
  - matched skills
  - missing skills
  - resume-only skills

### Comparison Mode

- candidate comparison chart based on ensemble scores

Output is written to `outputs/` by default.

---

## Current Observed Behavior

Based on testing with the real resumes in this repository:

- the skill extraction layer is informative,
- the CLI and graph generation work correctly,
- but the raw LSTM and CNN scores are still tightly clustered around `49%`.

### What This Means

The current models are being tested correctly on the resumes, but they are **not yet well calibrated**.

For example:

- frontend-aligned job descriptions should score higher for frontend-heavy resumes
- data science job descriptions should score lower when the resume lacks data science skills

At the moment, the explainability layer reflects those differences more clearly than the raw neural outputs.

---

## Known Limitations

1. Raw model scores are too compressed.
2. The bootstrap training data is synthetic and limited.
3. The explainability layer is rule-based, not learned.
4. The ensemble score is just an average, not a validated fusion model.
5. TensorFlow prints a Windows GPU warning during runs.

---

## Recommended Next Technical Step

If the goal is stronger model-only evaluation on your resumes, the next improvement should be:

1. collect or construct better labeled resume/JD training examples,
2. retrain the CNN and LSTM with clearer positive/negative separation,
3. evaluate them across frontend, backend, and data science JDs,
4. verify that raw LSTM/CNN scores move in the expected direction without relying on skill heuristics.

---

## Summary

This project is now a **clean CLI architecture** for resume-to-job-description matching using:

- `LSTM` for sequential similarity
- `CNN` for local-pattern similarity

The numeric scores shown to the user come directly from those two models.

The surrounding reporting layer adds:

- explanations
- rankings
- graphs
- JSON export

The system is operational and tested, but the current raw neural scores still need better calibration before they can be treated as strong standalone decision signals.
