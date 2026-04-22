# Model Performance Evaluation Guide

## Overview

This guide explains how to evaluate and compare the performance of LSTM and CNN models used in the resume screening system.

## Quick Start Command

Run the evaluation script with:

```bash
cd "c:\Users\HP\OneDrive\Desktop\bansari projects\college\DL-mini-project"
c:/python313/python.exe model_evaluation.py
```

Or with filtered output (removes TensorFlow warnings):

```bash
c:/python313/python.exe model_evaluation.py 2>&1 | findstr /V "WARNING\|oneDNN\|TensorFlow\|optimized\|cpu_feature\|I0000\|E0000"
```

---

## What the Script Does

The `model_evaluation.py` script performs 5 comprehensive steps:

### 1. **Data Preparation**

- Generates synthetic training data (resume-JD pairs)
- Creates balanced positive and negative samples
- Splits into 80% training and 20% testing data
- Shows class distribution

### 2. **Tokenization**

- Builds vocabulary from training data
- Converts text to padded sequences
- Prepares input for both models

### 3. **Model Training**

- Builds LSTM and CNN models with identical architecture structure
- Trains LSTM on 20 epochs with batch size 8
- Trains CNN on 20 epochs with batch size 8
- Uses 20% validation split during training

### 4. **Model Evaluation**

- Tests both models on unseen test data
- Generates predictions for both binary classification and probability scores
- Calculates comprehensive metrics

### 5. **Analysis & Recommendations**

- Compares performance across metrics
- Identifies overfitting
- Provides recommendations

---

## Performance Metrics Explained

### Accuracy

- **Definition**: Percentage of correct predictions
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Good for**: Balanced datasets
- **From Results**: LSTM 66.67%, CNN 50%

### Precision

- **Definition**: Of positive predictions, how many were correct?
- **Formula**: TP / (TP + FP)
- **Good for**: When false positives are costly
- **From Results**: LSTM 57.14%, CNN 42.86%

### Recall

- **Definition**: Of actual positives, how many did we find?
- **Formula**: TP / (TP + FN)
- **Good for**: When false negatives are costly
- **From Results**: LSTM 80%, CNN 60%

### F1-Score

- **Definition**: Harmonic mean of precision and recall
- **Formula**: 2 _ (Precision _ Recall) / (Precision + Recall)
- **Good for**: Imbalanced datasets
- **From Results**: LSTM 66.67%, CNN 50%

### ROC-AUC

- **Definition**: Area under the ROC curve
- **Range**: 0 to 1 (1 is perfect)
- **Good for**: Evaluating probability predictions
- **From Results**: LSTM 82.86%, CNN 31.43%

---

## Sample Output Breakdown

### Current Results:

```
LSTM Performance:
  Train Accuracy: 89.58% (Good - model learned the training data well)
  Test Accuracy:  66.67% (Decent - generalizes to unseen data)
  F1-Score:       66.67% (Balanced precision and recall)
  ROC-AUC:        82.86% (Good probability predictions)

CNN Performance:
  Train Accuracy: 75.00% (Lower than LSTM)
  Test Accuracy:  50.00% (Poor - barely better than random)
  F1-Score:       50.00% (Weak performance)
  ROC-AUC:        31.43% (Poor probability predictions)
```

### Confusion Matrix Interpretation:

For LSTM (Test Data):

```
        Predicted
       Neg  Pos
Actual Neg    4    3    <- Correctly predicted 4 negatives, 3 false positives
       Pos    1    4    <- 1 false negative, correctly predicted 4 positives
```

This means:

- Out of 7 actual negative samples: 4 correct, 3 wrong
- Out of 5 actual positive samples: 4 correct, 1 wrong

---

## Key Findings

### 1. **LSTM is Better Than CNN**

- LSTM test accuracy: 66.67%
- CNN test accuracy: 50%
- **Difference: 16.67% advantage for LSTM**

### 2. **Overfitting Detected**

Both models show overfitting:

- LSTM: 22.92% gap between train (89.58%) and test (66.67%)
- CNN: 25% gap between train (75%) and test (50%)

This suggests:

- Models memorize training data but don't generalize well
- Need more training data
- Consider adding L1/L2 regularization

### 3. **Model Characteristics**

```
LSTM:
- Better sequential pattern understanding
- Catches most positive cases (high recall: 80%)
- Some false positives (precision: 57%)
- Better overall performance

CNN:
- Weaker pattern detection
- Misses many positive cases (low recall: 60%)
- More false positives (precision: 43%)
- Random-like performance on test data (ROC-AUC: 31%)
```

---

## How to Improve Model Performance

### 1. **Increase Training Data**

Currently: 60 synthetic samples (48 train, 12 test)
Recommendation: 500-1000 real resume-JD pairs

```python
# In model_evaluation.py, increase:
NEGATIVE_RATIO = 2.0  # More negative samples
# And generate more role templates
```

### 2. **Add Regularization**

Reduce overfitting:

```python
# In models/lstm_similarity.py and cnn_text_classification.py:
from keras.layers import Dropout
from keras.regularizers import L1L2

# Add dropout layers
x = Dropout(0.3)(x)

# Add regularization to Dense layers
Dense(64, activation='relu', kernel_regularizer=L1L2(l1=0.001, l2=0.001))
```

### 3. **Tune Hyperparameters**

Experiment with:

```python
# Increase training
TRAINING_EPOCHS = 50  # From 20
TRAINING_BATCH_SIZE = 4  # Smaller batches

# Adjust architecture
EMBEDDING_DIM = 128  # From 64 (larger embeddings)
```

### 4. **Early Stopping**

Prevent overfitting:

```python
from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(..., callbacks=[early_stop])
```

### 5. **Class Weighting**

Handle imbalanced data:

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_targets),
    y=train_targets
)

model.fit(
    x, y,
    class_weight=dict(enumerate(class_weights))
)
```

---

## Interpreting Your Results

### If Test Accuracy > 75%

✅ Model is ready for production

- Good generalization
- Low overfitting risk

### If Test Accuracy 50-75%

⚠️ Model needs improvement

- Apply regularization
- Increase training data
- Adjust hyperparameters

### If Test Accuracy < 50%

❌ Model needs significant work

- Not better than random
- Substantially more data/training needed
- May need architecture changes

---

## Running Custom Evaluations

### Modify Training Parameters

Edit `model_evaluation.py`:

```python
TRAINING_EPOCHS = 50        # Increase training
TRAINING_BATCH_SIZE = 4     # Smaller batches
EMBEDDING_DIM = 128         # Larger embeddings
TEST_SIZE = 0.3             # 30% test data
NEGATIVE_RATIO = 2.0        # More negative samples
```

### Test on Different Data Splits

```python
# Change these values:
TEST_SIZE = 0.1   # 10% test (90% train)
TEST_SIZE = 0.5   # 50% test (50% train)
```

### Compare with Different Architecture

Create copies of the models with modified layers and compare.

---

## Quick Reference: Metrics Table

| Metric    | LSTM   | CNN    | Better? | Interpretation              |
| --------- | ------ | ------ | ------- | --------------------------- |
| Train Acc | 89.58% | 75%    | LSTM    | LSTM learned better         |
| Test Acc  | 66.67% | 50%    | LSTM    | LSTM generalizes better     |
| Precision | 57.14% | 42.86% | LSTM    | LSTM has fewer false alarms |
| Recall    | 80%    | 60%    | LSTM    | LSTM catches more matches   |
| F1-Score  | 66.67% | 50%    | LSTM    | LSTM is more balanced       |
| ROC-AUC   | 82.86% | 31.43% | LSTM    | LSTM has better ranking     |

---

## Troubleshooting

### Script takes too long

- Reduce `TRAINING_EPOCHS` to 5-10
- Reduce `TEST_SIZE` to 0.1 (less test data)

### Out of memory

- Reduce `TRAINING_BATCH_SIZE` to 4 or 2
- Reduce `MAX_SEQUENCE_LENGTH` to 100

### Metrics show all zeros

- Check if data has correct labels
- Verify tokenizer is fitting properly
- Check model predictions are in [0, 1] range

---

## Next Steps

1. **Run the evaluation**: Execute the command above
2. **Analyze results**: Compare your metrics with these benchmarks
3. **Improve models**: Apply the recommendations above
4. **Re-evaluate**: Run the script again after changes
5. **Deploy best model**: Use LSTM for production based on current results

---

## Files Reference

- **Script**: `model_evaluation.py`
- **Models**: `models/lstm_similarity.py`, `models/cnn_text_classification.py`
- **Data**: Generated synthetically in the script

---

## Command Cheat Sheet

```bash
# Full evaluation
c:/python313/python.exe model_evaluation.py

# Clean output (no warnings)
c:/python313/python.exe model_evaluation.py 2>&1 | findstr /V "WARNING\|oneDNN\|TensorFlow"

# Save to file
c:/python313/python.exe model_evaluation.py > evaluation_results.txt 2>&1

# Quick test (5 epochs instead of 20 - edit script first)
c:/python313/python.exe model_evaluation.py
```

---

**Last Evaluated**: Current Results
**Models Tested**: LSTM vs CNN
**Recommendation**: LSTM wins - use for production (but fix overfitting first)
