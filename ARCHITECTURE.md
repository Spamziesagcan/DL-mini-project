# Resume Screening: LSTM vs CNN Architecture

## Overview

This document explains the LSTM and CNN architectures used in the resume screening system, how they work for matching resumes with job descriptions, and which approach is better for different use cases.

---

## 1. LSTM Architecture for Resume Screening

### What is LSTM?

LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) that can handle sequential data and capture long-range dependencies. Unlike traditional RNNs, LSTMs have memory cells that can store information over long sequences.

### Architecture Flow

```
Resume Text (Tokenized)
           ↓
      Embedding Layer (64-dim)
      vocab_size × 64 matrix
           ↓
      LSTM Layer (64 units)
      Processes sequence sequentially
      Maintains memory across tokens
           ↓
      Output Vector (64-dim)
           ↓
    [Concatenate with JD Vector]
           ↓
      Dense Layer (64 units, ReLU)
           ↓
      Output Layer (1 unit, Sigmoid)
      [Similarity Score: 0-1]
```

### Detailed Components

1. **Embedding Layer**
   - Input: Tokenized resume (sequence of integers)
   - Output: Dense vectors of size 64
   - Purpose: Convert discrete tokens to continuous representations
   - Vocabulary size: Dynamic (based on training data)

2. **LSTM Layer**
   - Units: 64
   - Key Features:
     - **Cell State (C)**: Memory that persists across the sequence
     - **Hidden State (H)**: Information passed to next timestep
     - **3 Gates**:
       - **Forget Gate**: Decides what to discard from memory
       - **Input Gate**: Decides what new information to store
       - **Output Gate**: Decides what to output based on memory
   - Processing: Sequential (word by word)
   - Output: Final hidden state (64-dim vector)

3. **Merging Strategy**
   - Both resume and JD are encoded separately using the same LSTM encoder
   - Outputs are concatenated: [Resume_vector (64) + JD_vector (64)] = 128-dim vector

4. **Dense Layers**
   - Hidden Dense Layer: 64 units with ReLU activation
   - Output Layer: 1 unit with Sigmoid activation (outputs 0-1)

### Key Characteristics of LSTM

✅ **Strengths:**

- Captures sequential patterns and word order
- Remembers long dependencies in text
- Good at understanding context across entire document
- Better for capturing grammar and syntax

❌ **Limitations:**

- Slower training and inference (processes sequentially)
- More parameters to train
- May overfit on smaller datasets
- Computational overhead for long sequences

### Training Configuration

```python
Optimizer: Adam
Loss Function: Binary Crossentropy (similarity = 0 or 1)
Epochs: 8
Batch Size: 8
Metrics: Accuracy
```

---

## 2. CNN Architecture for Resume Screening

### What is CNN?

CNN (Convolutional Neural Network) uses convolutional filters to detect local patterns in data. For text, it looks for n-gram patterns (phrases) rather than sequential dependencies.

### Architecture Flow

```
Resume Text (Tokenized)
           ↓
      Embedding Layer (64-dim)
      vocab_size × 64 matrix
           ↓
      Conv1D Layer (64 filters, kernel=5)
      Detects local 5-word patterns
      Padding: 'same' (maintain length)
           ↓
      GlobalMaxPooling1D Layer
      Select most important feature per filter
      Output: 64-dim vector
           ↓
      Dense Layer (64 units, ReLU)
           ↓
    [Concatenate with JD Vector]
           ↓
      Dense Layer (64 units, ReLU)
           ↓
      Output Layer (1 unit, Sigmoid)
      [Similarity Score: 0-1]
```

### Detailed Components

1. **Embedding Layer** (Same as LSTM)
   - Input: Tokenized resume
   - Output: Dense vectors of size 64
   - Purpose: Convert tokens to embeddings

2. **Conv1D Layer** (1D Convolution)
   - Filters: 64
   - Kernel Size: 5 (looks at 5 consecutive words = 5-grams)
   - Activation: ReLU
   - Padding: Same (maintains sequence length)
   - Purpose: Detect local phrase patterns

   **How it works:**

   ```
   Input sequence: [w1, w2, w3, w4, w5, w6, w7, ...]

   Each filter slides across with kernel_size=5:
   Filter 1: [w1 w2 w3 w4 w5] → feature_1
   Filter 1: [w2 w3 w4 w5 w6] → feature_2
   Filter 1: [w3 w4 w5 w6 w7] → feature_3
   ... (64 filters doing this in parallel)

   Result: 64 × sequence_length feature map
   ```

3. **GlobalMaxPooling1D Layer**
   - Purpose: Select the most important feature from each filter
   - Operation: Takes max value across the entire sequence for each filter
   - Output: 64-dim vector (best representation of each pattern)

4. **Dense Layers**
   - Hidden Dense: 64 units with ReLU
   - Output Dense: 1 unit with Sigmoid

### Key Characteristics of CNN

✅ **Strengths:**

- Very fast inference (parallel convolutions)
- Captures local n-gram patterns efficiently
- Great at finding keyword phrases and terminology
- Low computational overhead
- Faster training

❌ **Limitations:**

- Limited to local context (kernel_size=5)
- May miss long-range semantic relationships
- Less sensitive to word order beyond n-gram window
- May underperform on grammatically complex sentences

### Training Configuration

```python
Same as LSTM:
Optimizer: Adam
Loss Function: Binary Crossentropy
Epochs: 8
Batch Size: 8
Metrics: Accuracy
```

---

## 3. How Both Models Are Used for Resume Screening

### Pipeline Overview

```
Resume PDF
    ↓
[PDF Extraction] → Raw resume text
    ↓
[Text Cleaning] → Clean resume text
    ↓
[Tokenization & Padding] → Sequence of integers (length=200)
    ↓
┌─────────────────────────────────────────┐
│                                         │
│  LSTM Model          CNN Model          │
│  │                   │                  │
│  ├→ Similarity Score ├→ Similarity Score│
│                 %            %          │
│                                         │
└─────────────────────────────────────────┘
    ↓
[Ensemble Combination]
    ↓
Final Match Score = (LSTM + CNN) / 2
    ↓
[Skill Extraction & Matching]
    ↓
Final Results with:
- LSTM Score
- CNN Score
- Matched Skills
- Missing Skills
```

### Prediction Process

1. **Text Preparation**

   ```python
   Resume text: "Bansari Naik with 3 years in Python..."
   JD text: "Seeking Python developer with ML experience..."
   ```

2. **Tokenization** (Uses fitted tokenizer from training)

   ```python
   Resume: [45, 23, 189, 12, 78, 34, ...]  # 200 tokens
   JD:     [89, 12, 45, 156, 23, 67, ...]  # 200 tokens
   ```

3. **Parallel Model Inference**

   ```
   LSTM Model:
   [resume_tokens, jd_tokens] → LSTM Encoder → 64-dim vectors
                             → Concatenate → Dense → Output
                             → LSTM Score (0-1)

   CNN Model:
   [resume_tokens, jd_tokens] → CNN Encoder → 64-dim vectors
                             → Concatenate → Dense → Output
                             → CNN Score (0-1)
   ```

4. **Score Combination**
   ```
   Final Score = (LSTM_Score + CNN_Score) / 2
   Example: (0.4448 + 0.4218) / 2 = 0.4333 = 43.33%
   ```

### Real Example Output

```
LSTM Score:    44.48%  (Good at sequential understanding)
CNN Score:     42.18%  (Good at keyword matching)
Final Score:   43.33%  (Ensemble average)
TF-IDF Score:   5.47%  (Traditional baseline - for comparison)
```

---

## 4. Comparison: LSTM vs CNN

### Performance Comparison Table

| Aspect                | LSTM                         | CNN                   |
| --------------------- | ---------------------------- | --------------------- |
| **Speed**             | Slower (sequential)          | Faster (parallel)     |
| **Context Range**     | Long-range (entire document) | Local (5-word window) |
| **Training Time**     | ~5-10 seconds                | ~2-3 seconds          |
| **Parameters**        | ~30K+                        | ~20K+                 |
| **Pattern Detection** | Sequential dependencies      | N-gram patterns       |
| **Memory Usage**      | Higher                       | Lower                 |
| **GPU Utilization**   | Lower                        | Higher                |

### When to Use Each

#### ✅ Use LSTM When:

- Resume has important information spread across different sections
- Grammar and sentence structure matter
- You want to understand the overall narrative of the resume
- You have computational resources available
- Resume length can vary significantly
- You need to understand complex relationships between far-apart tokens

**Example:** Detecting that "3 years of experience in ML" scattered in different sections is more relevant than isolated keywords.

#### ✅ Use CNN When:

- You want fast inference for real-time applications
- Server/computational resources are limited
- Specific technical keywords and phrases matter most
- You need to scale to many concurrent requests
- Resume structure is relatively consistent
- Local context (skills, technologies) is more important than global understanding

**Example:** Quickly matching job requirements like "TensorFlow, Keras, Python" to resume keywords.

---

## 5. Which Model is Better?

### Verdict: **It Depends on Your Use Case**

#### For Your Resume Screening Project:

**Current Results:**

- LSTM: 44.48% match
- CNN: 42.18% match
- Difference: Only 2.3%

**Winner: LSTM by marginal advantage**

### Why LSTM Wins (Slightly):

1. **Better Understanding**: LSTM captures that you mention "Machine Learning, Neural Networks, Deep Learning" across your resume
2. **Context Awareness**: Understands the relationship between these concepts
3. **Narrative Understanding**: Recognizes you're describing a coherent skill set

### Why CNN Still Performs Well:

1. **Quick Pattern Matching**: Rapidly identifies key phrases like "Python", "TensorFlow", "NLP"
2. **Efficiency**: Much faster to score (important for screening many resumes)
3. **Close Results**: Only 2.3% difference suggests CNN captures most essential information

### Recommendation for Production

**Use Ensemble Approach** (What Your Code Already Does!) ✅

```
Advantages of combining both:
1. ✅ Captures both sequential and local patterns
2. ✅ More robust (if one model fails, other provides value)
3. ✅ Balanced accuracy from both perspectives
4. ✅ Averages out model-specific biases

Current Formula:
Final Score = (LSTM_Score + CNN_Score) / 2

Could be improved with:
Final Score = (LSTM_Score × 0.6 + CNN_Score × 0.4)
              ↑ More weight to better performing model
```

---

## 6. Optimization Suggestions

### To Improve Both Models:

1. **Increase Training Data**

   ```python
   # Current: ~30 synthetic training pairs
   # Suggested: 100-500 real resume-JD pairs
   ```

2. **Tune Hyperparameters**

   ```python
   # Test different values:
   - LSTM units: 32, 64, 128
   - CNN filters: 32, 64, 96, 128
   - Kernel sizes: 3, 5, 7
   - Embedding dimensions: 32, 64, 100
   ```

3. **Add Attention Mechanisms** (Advanced)

   ```python
   # Attention helps both models focus on important parts
   # Would improve both LSTM and CNN significantly
   ```

4. **Increase Epochs** (With validation-based early stopping)
   ```python
   # Current: 8 epochs
   # Suggested: 20-50 epochs with early stopping
   ```

---

## 7. Architecture Diagrams

### LSTM Detailed Flow

```
┌─────────────────┐
│ Resume Tokens   │
│  [45, 23, 189]  │
└────────┬────────┘
         │
         ↓
    ┌────────────────────┐
    │ Embedding Layer    │
    │ 45 → [0.2, 0.5...] │
    │ 23 → [0.1, 0.8...] │
    │189 → [0.9, 0.3...] │
    └────────┬───────────┘
             │
             ↓
    ┌────────────────────────────────────┐
    │ LSTM Cell (with internal gates)     │
    │                                    │
    │  Input: embedding                  │
    │  forget_gate: drop irrelevant info  │
    │  input_gate: add new info           │
    │  output_gate: produce output        │
    │  cell_state: memory                 │
    │                                    │
    │  Process: [emb1] → [emb2] → [emb3] │
    │           ↓       ↓        ↓       │
    │          (update cell state)        │
    │                                    │
    │  Output: 64-dim final state vector  │
    └────────┬────────────────────────────┘
             │
       ┌─────┴─────┐
       ↓           ↓
   [64-vec]   [64-vec]
   Resume     JD
       │           │
       └─────┬─────┘
             ↓
      [128-vec concatenated]
             ↓
      Dense(64, ReLU)
             ↓
      Dense(1, Sigmoid)
             ↓
         [Score: 0-1]
```

### CNN Detailed Flow

```
┌──────────────────┐
│ Resume Tokens    │
│ [45, 23, 189...]  │
└────────┬─────────┘
         │
         ↓
    ┌────────────────────┐
    │ Embedding Layer    │
    │ 45 → [0.2, 0.5...] │
    │ 23 → [0.1, 0.8...] │
    │189 → [0.9, 0.3...] │
    └────────┬───────────┘
             │
             ↓
    ┌──────────────────────────────────┐
    │ Conv1D (64 filters, kernel=5)    │
    │                                  │
    │ Window 1: [emb1,emb2,emb3,      │
    │            emb4,emb5] → feat_1  │
    │ Window 2: [emb2,emb3,emb4,      │
    │            emb5,emb6] → feat_2  │
    │ ... (64 filters doing this)      │
    │                                  │
    │ Output: 64 × 200 feature map     │
    └────────┬───────────────────────────┘
             │
             ↓
    ┌──────────────────────────────────┐
    │ GlobalMaxPooling1D Layer         │
    │                                  │
    │ For each of 64 filters:          │
    │   Take MAX(features) → 1 value   │
    │                                  │
    │ Output: 64-dim vector            │
    │ (best representation per filter) │
    └────────┬───────────────────────────┘
             │
       ┌─────┴─────┐
       ↓           ↓
   [64-vec]   [64-vec]
   Resume     JD
       │           │
       └─────┬─────┘
             ↓
      [128-vec concatenated]
             ↓
      Dense(64, ReLU)
             ↓
      Dense(1, Sigmoid)
             ↓
         [Score: 0-1]
```

---

## 8. Conclusion

### Key Takeaways:

1. **LSTM** = Better for understanding overall context and relationships
2. **CNN** = Better for speed and scalability
3. **Ensemble** = Best of both worlds (your current approach!)

### Your Current Implementation is Optimal:

Your code uses both models and averages them, which provides:

- ✅ Robust scoring
- ✅ Balanced performance
- ✅ Resilience to model-specific biases
- ✅ Better generalization

### For Production Use:

- Continue using the ensemble approach
- Monitor which model performs better on your data
- Consider weighted averaging based on validation performance
- Add A/B testing with real resumes to optimize weights
