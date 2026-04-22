"""
Model Performance Evaluation Script
Evaluates LSTM and CNN models on training and test datasets
Shows detailed metrics comparison between both models
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import train_test_split

from preprocessing.text_cleaning import clean_text
from preprocessing.tokenizer_pipeline import fit_tokenizer, texts_to_padded_sequences
from utils.synthetic_data import generate_synthetic_training_data
from utils.skill_extraction import ROLE_SKILL_HINTS
from models.lstm_similarity import build_lstm_similarity_model
from models.cnn_text_classification import build_cnn_similarity_model


# Configuration
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 64
TRAINING_EPOCHS = 20  # More epochs for better training data
TRAINING_BATCH_SIZE = 8
NEGATIVE_RATIO = 1.5
TEST_SIZE = 0.2
RANDOM_STATE = 42


def _build_training_texts():
    """Build synthetic training texts based on role profiles"""
    _BASE_ROLE_PROFILES = {
        "web developer": ("html", "css", "javascript", "react", "frontend", "git"),
        "frontend developer": ("html", "css", "javascript", "react", "typescript", "frontend"),
        "backend developer": ("python", "sql", "django", "flask", "backend", "docker"),
        "full stack developer": ("html", "css", "javascript", "react", "node.js", "python", "sql"),
        "data scientist": ("python", "pandas", "numpy", "scikit-learn", "machine learning", "tableau"),
        "machine learning engineer": ("python", "tensorflow", "keras", "pytorch", "machine learning", "docker"),
        "data analyst": ("sql", "excel", "pandas", "tableau", "statistics", "python"),
        "devops engineer": ("docker", "git", "linux", "aws", "spark", "hadoop"),
    }
    
    _ROLE_TEMPLATES = (
        (
            "experienced {role} with hands-on work in {skills}. built products, automated workflows, and collaborated with cross-functional teams.",
            "hiring {role} with strong experience in {skills}. the role needs clean code, reliable delivery, and clear communication.",
        ),
        (
            "{role} professional focused on {skills}. delivered production-ready solutions and improved user experience.",
            "seeking {role} who can apply {skills} to solve real business problems and work closely with product teams.",
        ),
        (
            "results-driven {role} using {skills} to ship software, analyze feedback, and improve reliability.",
            "looking for {role} who brings {skills} and can contribute to architecture, implementation, and testing.",
        ),
    )
    
    resume_texts = []
    jd_texts = []

    profiles = dict(_BASE_ROLE_PROFILES)
    for role, skills in ROLE_SKILL_HINTS.items():
        profiles.setdefault(role, tuple(skills))

    for role, skills in profiles.items():
        skill_phrase = ", ".join(skills)
        for resume_template, jd_template in _ROLE_TEMPLATES:
            resume_texts.append(clean_text(resume_template.format(role=role, skills=skill_phrase)))
            jd_texts.append(clean_text(jd_template.format(role=role, skills=skill_phrase)))

    return resume_texts, jd_texts


def evaluate_models():
    """Evaluate LSTM and CNN models with detailed metrics"""
    
    print("\n" + "=" * 85)
    print("MODEL PERFORMANCE EVALUATION - LSTM vs CNN")
    print("=" * 85)
    
    # ===== DATA PREPARATION =====
    print("\n[1/5] PREPARING DATA...")
    print("-" * 85)
    
    resume_texts, jd_texts = _build_training_texts()
    training_pairs, labels = generate_synthetic_training_data(
        resume_texts,
        jd_texts,
        negative_ratio=NEGATIVE_RATIO,
        random_state=RANDOM_STATE,
    )
    
    # Split into train and test
    train_pairs, test_pairs, train_labels, test_labels = train_test_split(
        training_pairs,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )
    
    print(f"Total samples: {len(training_pairs)}")
    print(f"Training samples: {len(train_pairs)} ({len(train_pairs)/len(training_pairs)*100:.1f}%)")
    print(f"Test samples: {len(test_pairs)} ({len(test_pairs)/len(training_pairs)*100:.1f}%)")
    print(f"Positive samples: {sum(train_labels)} / {len(train_labels)} ({sum(train_labels)/len(train_labels)*100:.1f}%)")
    print(f"Negative samples: {len(train_labels)-sum(train_labels)} / {len(train_labels)} ({(len(train_labels)-sum(train_labels))/len(train_labels)*100:.1f}%)")
    
    # ===== TOKENIZATION =====
    print("\n[2/5] TOKENIZING DATA...")
    print("-" * 85)
    
    tokenizer = fit_tokenizer([text for pair in training_pairs for text in pair])
    vocab_size = max(len(tokenizer.word_index) + 1, 2)
    print(f"Vocabulary size: {vocab_size}")
    
    # Prepare sequences
    def prepare_sequences(pairs, labels):
        resume_texts = [clean_text(resume_text) for resume_text, _ in pairs]
        jd_texts = [clean_text(jd_text) for _, jd_text in pairs]
        
        resume_sequences = texts_to_padded_sequences(tokenizer, resume_texts, max_length=MAX_SEQUENCE_LENGTH)
        jd_sequences = texts_to_padded_sequences(tokenizer, jd_texts, max_length=MAX_SEQUENCE_LENGTH)
        
        return (resume_sequences.astype("int32"), jd_sequences.astype("int32"), 
                np.asarray(labels, dtype="float32"))
    
    train_resume_seq, train_jd_seq, train_targets = prepare_sequences(train_pairs, train_labels)
    test_resume_seq, test_jd_seq, test_targets = prepare_sequences(test_pairs, test_labels)
    
    print(f"Sequences prepared: Resume shape {train_resume_seq.shape}, JD shape {train_jd_seq.shape}")
    
    # ===== MODEL TRAINING =====
    print("\n[3/5] BUILDING AND TRAINING MODELS...")
    print("-" * 85)
    
    # Build models
    print("Building LSTM model...")
    lstm_model = build_lstm_similarity_model(
        vocab_size=vocab_size,
        max_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM
    )
    
    print("Building CNN model...")
    cnn_model = build_cnn_similarity_model(
        vocab_size=vocab_size,
        max_length=MAX_SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM
    )
    
    # Train LSTM
    print(f"\nTraining LSTM model ({TRAINING_EPOCHS} epochs)...")
    lstm_history = lstm_model.fit(
        [train_resume_seq, train_jd_seq],
        train_targets,
        epochs=TRAINING_EPOCHS,
        batch_size=TRAINING_BATCH_SIZE,
        validation_split=0.2,
        verbose=0,
        shuffle=True
    )
    
    # Train CNN
    print(f"Training CNN model ({TRAINING_EPOCHS} epochs)...")
    cnn_history = cnn_model.fit(
        [train_resume_seq, train_jd_seq],
        train_targets,
        epochs=TRAINING_EPOCHS,
        batch_size=TRAINING_BATCH_SIZE,
        validation_split=0.2,
        verbose=0,
        shuffle=True
    )
    
    # ===== MODEL EVALUATION =====
    print("\n[4/5] EVALUATING MODELS...")
    print("-" * 85)
    
    # Get predictions
    print("Getting predictions...")
    
    # Training predictions
    lstm_train_pred = lstm_model.predict([train_resume_seq, train_jd_seq], verbose=0).flatten()
    cnn_train_pred = cnn_model.predict([train_resume_seq, train_jd_seq], verbose=0).flatten()
    
    # Test predictions
    lstm_test_pred = lstm_model.predict([test_resume_seq, test_jd_seq], verbose=0).flatten()
    cnn_test_pred = cnn_model.predict([test_resume_seq, test_jd_seq], verbose=0).flatten()
    
    # Binary predictions
    lstm_train_binary = (lstm_train_pred > 0.5).astype(int)
    lstm_test_binary = (lstm_test_pred > 0.5).astype(int)
    cnn_train_binary = (cnn_train_pred > 0.5).astype(int)
    cnn_test_binary = (cnn_test_pred > 0.5).astype(int)
    
    # ===== METRICS CALCULATION =====
    print("\n[5/5] CALCULATING METRICS...")
    print("-" * 85)
    
    def calculate_metrics(y_true, y_pred, y_pred_proba, model_name, dataset_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except:
            roc_auc = 0.0
        
        return {
            "Model": model_name,
            "Dataset": dataset_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        }
    
    # Calculate metrics for both models and both datasets
    metrics = []
    
    metrics.append(calculate_metrics(train_targets, lstm_train_binary, lstm_train_pred, "LSTM", "Train"))
    metrics.append(calculate_metrics(test_targets, lstm_test_binary, lstm_test_pred, "LSTM", "Test"))
    metrics.append(calculate_metrics(train_targets, cnn_train_binary, cnn_train_pred, "CNN", "Train"))
    metrics.append(calculate_metrics(test_targets, cnn_test_binary, cnn_test_pred, "CNN", "Test"))
    
    # ===== RESULTS DISPLAY =====
    print("\n" + "=" * 85)
    print("PERFORMANCE METRICS")
    print("=" * 85)
    
    for metric in metrics:
        print(f"\n[{metric['Model']}] {metric['Dataset']} Dataset:")
        print(f"  Accuracy:   {metric['Accuracy']:.4f} ({metric['Accuracy']*100:.2f}%)")
        print(f"  Precision:  {metric['Precision']:.4f}")
        print(f"  Recall:     {metric['Recall']:.4f}")
        print(f"  F1-Score:   {metric['F1-Score']:.4f}")
        print(f"  ROC-AUC:    {metric['ROC-AUC']:.4f}")
    
    # ===== MODEL COMPARISON =====
    print("\n" + "=" * 85)
    print("MODEL COMPARISON")
    print("=" * 85)
    
    lstm_train_acc = metrics[0]["Accuracy"]
    lstm_test_acc = metrics[1]["Accuracy"]
    cnn_train_acc = metrics[2]["Accuracy"]
    cnn_test_acc = metrics[3]["Accuracy"]
    
    lstm_f1 = metrics[1]["F1-Score"]
    cnn_f1 = metrics[3]["F1-Score"]
    
    print(f"\nTraining Accuracy:")
    print(f"  LSTM: {lstm_train_acc:.4f} ({lstm_train_acc*100:.2f}%)")
    print(f"  CNN:  {cnn_train_acc:.4f} ({cnn_train_acc*100:.2f}%) [{'+' if cnn_train_acc > lstm_train_acc else ''}{(cnn_train_acc-lstm_train_acc)*100:.2f}%]")
    
    print(f"\nTest Accuracy:")
    print(f"  LSTM: {lstm_test_acc:.4f} ({lstm_test_acc*100:.2f}%)")
    print(f"  CNN:  {cnn_test_acc:.4f} ({cnn_test_acc*100:.2f}%) [{'+' if cnn_test_acc > lstm_test_acc else ''}{(cnn_test_acc-lstm_test_acc)*100:.2f}%]")
    
    print(f"\nGeneralization (Train - Test Accuracy):")
    lstm_gen = lstm_train_acc - lstm_test_acc
    cnn_gen = cnn_train_acc - cnn_test_acc
    print(f"  LSTM: {lstm_gen:.4f} ({lstm_gen*100:.2f}%) {'[Overfitting]' if lstm_gen > 0.1 else '[Good]'}")
    print(f"  CNN:  {cnn_gen:.4f} ({cnn_gen*100:.2f}%) {'[Overfitting]' if cnn_gen > 0.1 else '[Good]'}")
    
    print(f"\nTest F1-Score (Best metric for imbalanced data):")
    print(f"  LSTM: {lstm_f1:.4f}")
    print(f"  CNN:  {cnn_f1:.4f} [{'+' if cnn_f1 > lstm_f1 else ''}{(cnn_f1-lstm_f1):.4f}]")
    
    # ===== CONFUSION MATRIX =====
    print("\n" + "=" * 85)
    print("CONFUSION MATRIX (Test Data)")
    print("=" * 85)
    
    def print_confusion_matrix(y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        print(f"\n[{model_name}] Confusion Matrix:")
        print(f"        Predicted")
        print(f"       Neg  Pos")
        print(f"Actual Neg  {cm[0,0]:3d}  {cm[0,1]:3d}")
        print(f"       Pos  {cm[1,0]:3d}  {cm[1,1]:3d}")
        print(f"\nInterpretation:")
        print(f"  True Negatives (TN):  {cm[0,0]}")
        print(f"  False Positives (FP): {cm[0,1]}")
        print(f"  False Negatives (FN): {cm[1,0]}")
        print(f"  True Positives (TP):  {cm[1,1]}")
    
    print_confusion_matrix(test_targets, lstm_test_binary, "LSTM")
    print_confusion_matrix(test_targets, cnn_test_binary, "CNN")
    
    # ===== TRAINING CURVES =====
    print("\n" + "=" * 85)
    print("TRAINING HISTORY")
    print("=" * 85)
    
    print(f"\nLSTM Model:")
    print(f"  Initial Loss:  {lstm_history.history['loss'][0]:.4f}")
    print(f"  Final Loss:    {lstm_history.history['loss'][-1]:.4f}")
    print(f"  Initial Acc:   {lstm_history.history['accuracy'][0]:.4f}")
    print(f"  Final Acc:     {lstm_history.history['accuracy'][-1]:.4f}")
    print(f"  Val Loss:      {lstm_history.history['val_loss'][-1]:.4f}")
    print(f"  Val Acc:       {lstm_history.history['val_accuracy'][-1]:.4f}")
    
    print(f"\nCNN Model:")
    print(f"  Initial Loss:  {cnn_history.history['loss'][0]:.4f}")
    print(f"  Final Loss:    {cnn_history.history['loss'][-1]:.4f}")
    print(f"  Initial Acc:   {cnn_history.history['accuracy'][0]:.4f}")
    print(f"  Final Acc:     {cnn_history.history['accuracy'][-1]:.4f}")
    print(f"  Val Loss:      {cnn_history.history['val_loss'][-1]:.4f}")
    print(f"  Val Acc:       {cnn_history.history['val_accuracy'][-1]:.4f}")
    
    # ===== RECOMMENDATIONS =====
    print("\n" + "=" * 85)
    print("RECOMMENDATIONS")
    print("=" * 85)
    
    best_test_acc = max(lstm_test_acc, cnn_test_acc)
    best_model = "LSTM" if lstm_test_acc > cnn_test_acc else "CNN"
    
    print(f"\nBest Performing Model: {best_model}")
    print(f"  Test Accuracy: {best_test_acc*100:.2f}%")
    print(f"  Margin: {abs(lstm_test_acc - cnn_test_acc)*100:.2f}%")
    
    if abs(lstm_test_acc - cnn_test_acc) < 0.02:
        print("\nModels have similar performance. Ensemble approach is recommended.")
    else:
        print(f"\n{best_model} model is significantly better. Consider using it for production.")
    
    if lstm_gen > 0.15 or cnn_gen > 0.15:
        print("\nWarning: Significant overfitting detected. Consider:")
        print("  - Increasing training data")
        print("  - Adding regularization")
        print("  - Reducing model complexity")
    else:
        print("\nModel generalization is good. Both train and test metrics are balanced.")
    
    print("\n" + "=" * 85 + "\n")


if __name__ == "__main__":
    evaluate_models()
