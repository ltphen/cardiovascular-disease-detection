import os
import numpy as np
from src.data_processing import load_ptbxl_data
from src.feature_extraction import extract_features
from src.models import train_cnn_model, train_cnn_lstm_model, train_cnn_lstm_attention_model, train_classical_model, evaluate_model
from sklearn.metrics import roc_auc_score

def main():
    # 1. Load and process PTB-XL data
    print("Loading PTB-XL data...")
    X_train, y_train, X_val, y_val, X_test, y_test, label_names = load_ptbxl_data(
        root_dir="data/raw/",  # <--- CHANGE THIS PATH to your PTB-XL data location
        sampling_frequency=100
    )
    num_classes = y_train.shape[1]

    # 2. Train CNN model
    print("\nTraining CNN model...")
    cnn_model, cnn_history = train_cnn_model(
        X_train, y_train,
        X_val, y_val,
        num_classes=num_classes,
        batch_size=32,
        epochs=50,
        patience=10  # Increased patience for larger dataset
    )

    # 2b. Train CNN + LSTM hybrid model
    print("\nTraining CNN-LSTM hybrid model...")
    hybrid_model, hybrid_history = train_cnn_lstm_model(
        X_train, y_train,
        X_val, y_val,
        num_classes=num_classes,
        batch_size=32,
        epochs=50,
        patience=10,
        lstm_units=64
    )

    # 2c. Train CNN + LSTM + Attention model
    print("\nTraining CNN-LSTM-Attention model...")
    attn_model, attn_history = train_cnn_lstm_attention_model(
        X_train, y_train,
        X_val, y_val,
        num_classes=num_classes,
        batch_size=32,
        epochs=50,
        patience=10,
        lstm_units=64
    )

    # 3. Train classical ML model
    print("\nExtracting features for classical ML...")
    X_train_feat = extract_features(X_train, fs=100)
    X_val_feat = extract_features(X_val, fs=100)
    X_test_feat = extract_features(X_test, fs=100)
    
    print("\nTraining Random Forest model...")
    rf_model, scaler = train_classical_model(
        X_train_feat, y_train,
        X_val_feat, y_val,
        model_type="rf",
        n_estimators=200
    )
    
    # 4. Evaluate both models
    print("\n--- Final Evaluation ---")
    
    # Evaluate CNN
    print("Evaluating CNN model...")
    y_pred_cnn = cnn_model.predict(X_test)
    cnn_auc = roc_auc_score(y_test, y_pred_cnn, average='macro')
    print(f"CNN Test Macro AUC: {cnn_auc:.4f}")

    # Evaluate Hybrid
    print("\nEvaluating CNN-LSTM hybrid model...")
    y_pred_hybrid = hybrid_model.predict(X_test)
    hybrid_auc = roc_auc_score(y_test, y_pred_hybrid, average='macro')
    print(f"Hybrid Test Macro AUC: {hybrid_auc:.4f}")

    # Evaluate Attention model
    print("\nEvaluating CNN-LSTM-Attention model...")
    y_pred_attn = attn_model.predict(X_test)
    attn_auc = roc_auc_score(y_test, y_pred_attn, average='macro')
    print(f"Attention Test Macro AUC: {attn_auc:.4f}")

    # Evaluate Random Forest
    print("\nEvaluating Random Forest model...")
    X_test_feat_scaled = scaler.transform(X_test_feat)
    y_pred_rf = rf_model.predict_proba(X_test_feat_scaled)
    # predict_proba returns a list of [n_samples, 2] arrays. We need the prob of the positive class.
    y_pred_rf_formatted = np.array([pred[:, 1] for pred in y_pred_rf]).T
    rf_auc = roc_auc_score(y_test, y_pred_rf_formatted, average='macro')
    print(f"RF Test Macro AUC:  {rf_auc:.4f}")

    print("\n--- Detailed Per-Class AUC ---")
    print(f"{'Class':<10} | {'CNN AUC':<10} | {'Hybrid AUC':<12} | {'Attn AUC':<10} | {'RF AUC':<10}")
    print("-" * 70)
    for i, label in enumerate(label_names):
        cnn_class_auc = roc_auc_score(y_test[:, i], y_pred_cnn[:, i])
        hybrid_class_auc = roc_auc_score(y_test[:, i], y_pred_hybrid[:, i])
        attn_class_auc = roc_auc_score(y_test[:, i], y_pred_attn[:, i])
        rf_class_auc = roc_auc_score(y_test[:, i], y_pred_rf_formatted[:, i])
        print(f"{label:<10} | {cnn_class_auc:<10.4f} | {hybrid_class_auc:<12.4f} | {attn_class_auc:<10.4f} | {rf_class_auc:<10.4f}")

if __name__ == "__main__":
    main() 