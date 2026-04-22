# Resume Screening System Using Deep Learning

## Abstract

This project presents a deep learning based resume screening system that compares a candidate resume with a target job description and estimates job-fit similarity. The implemented system uses two neural architectures, namely a Long Short-Term Memory (LSTM) model and a Convolutional Neural Network (CNN), along with a rule-based skill extraction layer for interpretability. The workflow begins with PDF resume parsing, followed by text cleaning, tokenization, sequence padding, similarity scoring, and explainable reporting. In addition to raw model outputs, the system reports matched skills, missing skills, overlap ratio, recommendation text, JSON exports, and graphs. Experimental artifacts in the repository show that the CNN model performs more reliably on real resume-job evaluation cases, while the LSTM model remains useful as a secondary semantic similarity component. The overall system is implemented as a command-line application and is intended as a practical mini-project in automated resume screening.

## Introduction

Recruitment teams often receive a large number of resumes for a limited set of job roles. Manual screening is time-consuming and may become inconsistent when the volume of applicants increases. Automated resume screening can reduce initial filtering effort by quickly identifying candidates whose profiles align with a given job description.

Traditional keyword matching methods are easy to implement but they often fail to capture contextual relationships between resume content and job requirements. Deep learning methods offer a stronger alternative because they can model semantic patterns in text rather than relying only on exact string overlap. However, model predictions in hiring-related tasks also need interpretability so that the screening output is understandable and defensible.

This project addresses that need by combining deep learning similarity scoring with an explainability layer. The system reads resume text from PDF files, preprocesses both resume and job description text, sends the pair to CNN and LSTM similarity models, and then supplements those scores with skill overlap analysis. The result is a practical screening workflow that provides both numerical scores and human-readable reasoning.

## Proposed Methodology

The proposed methodology is based on a dual-input text similarity pipeline.

1. Resume text is extracted from PDF documents using `pdfplumber`.
2. Resume and job description text are normalized through lowercasing, punctuation removal, token handling, and stopword filtering.
3. The cleaned text is converted into token sequences using a tokenizer pipeline.
4. Sequences are padded or truncated to a fixed length of 60 tokens.
5. The same resume-job pair is evaluated by two deep learning models:
   - LSTM similarity model for sequential and contextual text understanding
   - CNN similarity model for local lexical and phrase-level feature extraction
6. Each model produces a sigmoid output between 0 and 1, which is converted to percentage form.
7. An ensemble score is calculated as the simple average of the LSTM and CNN scores.
8. A rule-based skill extraction module identifies matched skills, missing skills, and resume-only skills to improve interpretability.
9. The final output is returned through the CLI along with optional PNG charts and JSON exports.

This hybrid methodology was chosen because the neural models provide semantic scoring while the skill layer provides transparency for manual review.

## Overview of System

The current system is implemented as a CLI-based application. The main execution flow is:

```text
Resume PDF
  ->
PDF text extraction
  ->
Text cleaning
  ->
Tokenization and padding
  ->
LSTM model + CNN model
  ->
Score conversion to percentage
  ->
Skill overlap analysis
  ->
Recommendation + graphs + JSON report
```

### Major Modules

- `main.py` and `cli.py`: command-line entry and user interaction
- `pipeline.py`: end-to-end resume screening logic
- `preprocessing/pdf_text.py`: resume text extraction from PDF
- `preprocessing/text_cleaning.py`: cleaning and normalization
- `preprocessing/tokenizer_pipeline.py`: tokenization and padding utilities
- `models/lstm_similarity.py`: LSTM similarity architecture
- `models/cnn_text_classification.py`: CNN similarity architecture
- `models/similarity_runtime.py`: artifact loading, prediction, and fallback bootstrap training
- `utils/skill_extraction.py`: rule-based skill extraction
- `utils/reporting.py`: explanation and recommendation generation
- `utils/visualization.py`: score and skill alignment graph generation
- `baseline/tfidf_similarity.py`: TF-IDF cosine similarity baseline

### Model Details

#### LSTM Model

The LSTM model uses a shared embedding layer for both input texts, followed by LSTM encoding for the resume and job description independently. Their encoded representations are concatenated and passed through dense layers to produce a final sigmoid similarity score.

Current runtime configuration:

- Embedding dimension: 16
- LSTM units: 32
- Dense units: 32
- Output layer: 1 sigmoid neuron

#### CNN Model

The CNN model also uses dual input branches. Each branch applies shared embeddings, `Conv1D`, global max pooling, and dense encoding before merging both representations into a final sigmoid output.

Current runtime configuration:

- Embedding dimension: 16
- Conv1D filters: 32
- Kernel size: 5
- Encoder dense units: 32
- Merge dense units: 32
- Output layer: 1 sigmoid neuron

## Dataset Description

The project uses two forms of data.

### 1. Synthetic Bootstrap Training Data

The runtime can generate synthetic resume-job pairs when trained artifacts are not already present in the `artifacts/` folder. This synthetic data is created from predefined role templates and skill lists. The code currently limits bootstrap role profiles to 6 and uses a negative ratio of 1.0 to generate both positive and negative pairs.

Important characteristics:

- Generated from role and skill templates
- Used for lightweight fallback training
- Intended to keep the project runnable
- Not sufficient as a final benchmark dataset

### 2. Real Resume Evaluation Data

The repository contains real PDF resumes and saved evaluation outputs. The major evaluation artifacts indicate:

- 2 real candidate resumes were tested
- CNN evaluation was reported on 16 real cases formed from 2 resumes across 8 job descriptions
- Example roles include Full Stack Developer, Frontend Developer, Backend Developer, MERN Stack Developer, Machine Learning Engineer, Data Scientist, Java Developer, and Full Stack Web Engineer

The two main real resumes used in the repository are:

- `BansariNaik-thirdyr.pdf`
- `ChaitanyaShah_Resume (1).pdf`

This means the project is best viewed as a functional prototype with limited real-world evaluation rather than a large-scale industrial benchmark.

## Accuracy Metrics Used

The repository and result artifacts use the following evaluation metrics.

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

These metrics are explicitly reported for the CNN model in `cnn_test_results.json`.

### Similarity and Reporting Metrics

- LSTM score in percentage
- CNN score in percentage
- Ensemble score, computed as the average of LSTM and CNN scores
- Overlap ratio, computed as matched skills divided by total required job-description skills

### Why These Metrics Matter

- Accuracy measures overall correctness of match or non-match decisions.
- Precision measures how many predicted matches are actually relevant.
- Recall measures how many relevant candidates are successfully identified.
- F1-score balances precision and recall.
- ROC-AUC evaluates ranking quality across thresholds.
- Overlap ratio adds interpretability by directly showing skill alignment.

## Results and Discussion

The most important stored quantitative result in the repository is the CNN evaluation on real resume-job test cases.

### CNN Evaluation Results

From `cnn_test_results.json`, the CNN model achieved:

- Accuracy: 87.50%
- Precision: 85.71%
- Recall: 100.00%
- F1-score: 0.9231
- ROC-AUC: 0.7083

Confusion matrix values:

- True Positives: 12
- True Negatives: 2
- False Positives: 2
- False Negatives: 0

These values indicate that the CNN model is strong at identifying actual matches and does not miss relevant candidates in the tested set, which is especially useful for shortlisting tasks.

### Resume Comparison Results

The comparison artifact in `outputs/comparison.json` shows the system behavior for a full stack oriented job description.

For `BansariNaik-thirdyr.pdf`:

- LSTM score: 49.47%
- CNN score: 48.81%
- Ensemble score: 49.14%
- Overlap ratio: 0.8333
- Matched skills include `backend`, `frontend`, `git`, `html`, `javascript`, `node.js`, `python`, `react`, and `rest apis`

For `ChaitanyaShah_Resume (1).pdf`:

- LSTM score: 49.33%
- CNN score: 48.72%
- Ensemble score: 49.02%
- Overlap ratio: 0.7500
- Matched skills include `backend`, `git`, `html`, `javascript`, `node.js`, `python`, `react`, and `sql`

These results show that the explainability layer is more clearly differentiated than the raw neural scores in the current setup. While both candidates have similar neural scores, the skill overlap gives clearer evidence of relative job alignment.

### Discussion

The repository documentation repeatedly indicates that current neural scores are conservative and compressed. In practice, that means the models often produce close values for different candidate-job combinations, especially when using lightweight synthetic bootstrap training or limited calibration data. As a result:

- the system workflow is operational,
- the skill extraction output is informative,
- the CNN model shows promising real-case classification behavior,
- but the raw LSTM and CNN percentage outputs still need better calibration for stronger standalone decision-making.

This is a common limitation when small or synthetic training data is used for text similarity tasks.

## Comparisons

### CNN vs LSTM

The project documentation and stored results indicate that CNN is the stronger practical model in the current system.

#### CNN

- Better validated on real resume-job cases
- 87.50% accuracy on the reported real-case test set
- 100% recall on that test set
- More suitable as the primary screening model in the present implementation

#### LSTM

- Useful for contextual and sequential text modeling
- Produces stable but tightly clustered scores in current testing
- Appears less reliable as a primary production signal under the current data regime
- Better suited as a secondary validation component

### Deep Learning Models vs TF-IDF Baseline

The repository also includes a TF-IDF cosine similarity baseline. Compared with TF-IDF:

- TF-IDF is simple, fast, and easy to interpret
- TF-IDF mainly captures lexical overlap
- CNN and LSTM are better suited to learning semantic similarity patterns
- The explainability layer partly compensates for deep model opacity by surfacing matched and missing skills

Therefore, the current comparison suggests:

- TF-IDF is useful as a classical baseline
- CNN is the best current model for practical matching
- LSTM remains a supplementary semantic model
- skill overlap acts as an important explanatory layer alongside model scores

## Conclusion

This project successfully implements a complete resume screening pipeline using deep learning and explainable reporting. The system can read resumes from PDF files, preprocess text, generate LSTM and CNN similarity scores, analyze skill overlap, export results, and create visual summaries.

Among the implemented models, CNN currently provides the most reliable performance based on the stored real-case evaluation artifacts. The LSTM model contributes semantic similarity analysis but requires better calibration and stronger real-world training data to become equally dependable. The explainability layer is a major strength of the project because it makes the output more understandable for manual review.

In conclusion, the project demonstrates a practical and extensible mini-project in automated resume-job matching. Future improvement should focus on collecting a larger labeled dataset of real resume-job pairs, retraining the neural models on richer examples, and validating the system on broader job categories so that the raw scores become more discriminative and reliable.
