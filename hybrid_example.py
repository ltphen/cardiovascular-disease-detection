import os
import numpy as np
from src.data_processing import load_ptbxl_data
from src.feature_extraction import extract_features
from src.models import (
    train_cnn_model, 
    train_classical_model, 
    train_hybrid_model,
    evaluate_model
)
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(histories, model_names):
    """Plot training history for multiple models."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (history, name) in enumerate(zip(histories, model_names)):
        # Plot loss
        axes[0, 0].plot(history.history['loss'], label=f'{name} Train')
        axes[0, 0].plot(history.history['val_loss'], label=f'{name} Val')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # Plot accuracy
        axes[0, 1].plot(history.history['binary_accuracy'], label=f'{name} Train')
        axes[0, 1].plot(history.history['val_binary_accuracy'], label=f'{name} Val')
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].legend()
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        
        # Plot AUC
        axes[1, 0].plot(history.history['auc'], label=f'{name} Train')
        axes[1, 0].plot(history.history['val_auc'], label=f'{name} Val')
        axes[1, 0].set_title('Training AUC')
        axes[1, 0].legend()
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_auc_comparison(auc_scores, model_names, label_names):
    """Plot AUC comparison across models and classes."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Bar plot of macro AUC scores
    axes[0].bar(model_names, [auc_scores[name]['macro'] for name in model_names])
    axes[0].set_title('Macro AUC Comparison')
    axes[0].set_ylabel('Macro AUC')
    axes[0].set_ylim(0.5, 1.0)
    
    # Heatmap of per-class AUC scores
    class_aucs = np.array([[auc_scores[name]['per_class'][i] for name in model_names] 
                           for i in range(len(label_names))])
    
    sns.heatmap(class_aucs, 
                xticklabels=model_names, 
                yticklabels=label_names,
                annot=True, 
                fmt='.3f',
                cmap='RdYlBu_r',
                ax=axes[1])
    axes[1].set_title('Per-Class AUC Scores')
    
    plt.tight_layout()
    plt.savefig('auc_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 1. Load and process PTB-XL data
    print("Loading PTB-XL data...")
    X_train, y_train, X_val, y_val, X_test, y_test, label_names = load_ptbxl_data(
        root_dir="data/raw/",  # <--- CHANGE THIS PATH to your PTB-XL data location
        sampling_frequency=100
    )
    num_classes = y_train.shape[1]

    print(f"\nDataset Info:")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_names}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # 2. Train CNN model (baseline)
    print("\n" + "="*50)
    print("Training CNN model (baseline)...")
    print("="*50)
    cnn_model, cnn_history = train_cnn_model(
        X_train, y_train,
        X_val, y_val,
        num_classes=num_classes,
        batch_size=32,
        epochs=50,
        patience=10
    )
    
    # 3. Train Hybrid CNN+LSTM model
    print("\n" + "="*50)
    print("Training Hybrid CNN+LSTM model...")
    print("="*50)
    hybrid_model, hybrid_history = train_hybrid_model(
        X_train, y_train,
        X_val, y_val,
        num_classes=num_classes,
        model_type="hybrid",
        batch_size=32,
        epochs=50,
        patience=10,
        lstm_units=128,
        attention_size=128
    )
    
    # 4. Train Advanced Hybrid model
    print("\n" + "="*50)
    print("Training Advanced Hybrid model...")
    print("="*50)
    advanced_model, advanced_history = train_hybrid_model(
        X_train, y_train,
        X_val, y_val,
        num_classes=num_classes,
        model_type="advanced",
        batch_size=32,
        epochs=50,
        patience=10
    )
    
    # 5. Train classical ML model
    print("\n" + "="*50)
    print("Training Random Forest model...")
    print("="*50)
    X_train_feat = extract_features(X_train, fs=100)
    X_val_feat = extract_features(X_val, fs=100)
    X_test_feat = extract_features(X_test, fs=100)
    
    rf_model, scaler = train_classical_model(
        X_train_feat, y_train,
        X_val_feat, y_val,
        model_type="rf",
        n_estimators=200
    )
    
    # 6. Evaluate all models
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    # Get predictions from all models
    y_pred_cnn = cnn_model.predict(X_test)
    y_pred_hybrid = hybrid_model.predict(X_test)
    y_pred_advanced = advanced_model.predict(X_test)
    
    # Random Forest predictions
    X_test_feat_scaled = scaler.transform(X_test_feat)
    y_pred_rf_proba = rf_model.predict_proba(X_test_feat_scaled)
    y_pred_rf = np.array([pred[:, 1] for pred in y_pred_rf_proba]).T
    
    # Calculate AUC scores
    models = {
        'CNN': y_pred_cnn,
        'Hybrid CNN+LSTM': y_pred_hybrid,
        'Advanced Hybrid': y_pred_advanced,
        'Random Forest': y_pred_rf
    }
    
    auc_scores = {}
    print(f"\n{'Model':<20} | {'Macro AUC':<10}")
    print("-" * 35)
    
    for name, predictions in models.items():
        macro_auc = roc_auc_score(y_test, predictions, average='macro')
        per_class_auc = [roc_auc_score(y_test[:, i], predictions[:, i]) for i in range(num_classes)]
        
        auc_scores[name] = {
            'macro': macro_auc,
            'per_class': per_class_auc
        }
        
        print(f"{name:<20} | {macro_auc:<10.4f}")
    
    # 7. Detailed per-class analysis
    print(f"\n{'Class':<10} | {'CNN':<10} | {'Hybrid':<10} | {'Advanced':<10} | {'RF':<10}")
    print("-" * 60)
    for i, label in enumerate(label_names):
        cnn_auc = auc_scores['CNN']['per_class'][i]
        hybrid_auc = auc_scores['Hybrid CNN+LSTM']['per_class'][i]
        advanced_auc = auc_scores['Advanced Hybrid']['per_class'][i]
        rf_auc = auc_scores['Random Forest']['per_class'][i]
        
        print(f"{label:<10} | {cnn_auc:<10.4f} | {hybrid_auc:<10.4f} | {advanced_auc:<10.4f} | {rf_auc:<10.4f}")
    
    # 8. Model comparison analysis
    print(f"\n" + "="*50)
    print("Model Comparison Analysis")
    print("="*50)
    
    # Find best model
    best_model = max(auc_scores.keys(), key=lambda x: auc_scores[x]['macro'])
    print(f"Best performing model: {best_model} (Macro AUC: {auc_scores[best_model]['macro']:.4f})")
    
    # Improvement analysis
    baseline_auc = auc_scores['CNN']['macro']
    hybrid_improvement = (auc_scores['Hybrid CNN+LSTM']['macro'] - baseline_auc) / baseline_auc * 100
    advanced_improvement = (auc_scores['Advanced Hybrid']['macro'] - baseline_auc) / baseline_auc * 100
    
    print(f"Hybrid CNN+LSTM improvement over CNN: {hybrid_improvement:+.2f}%")
    print(f"Advanced Hybrid improvement over CNN: {advanced_improvement:+.2f}%")
    
    # 9. Plot results
    print(f"\nGenerating plots...")
    model_names = ['CNN', 'Hybrid CNN+LSTM', 'Advanced Hybrid']
    histories = [cnn_history, hybrid_history, advanced_history]
    
    plot_training_history(histories, model_names)
    plot_auc_comparison(auc_scores, list(auc_scores.keys()), label_names)
    
    print(f"\nPlots saved as 'training_history.png' and 'auc_comparison.png'")
    
    # 10. Save results
    results = {
        'auc_scores': auc_scores,
        'label_names': label_names,
        'model_comparison': {
            'best_model': best_model,
            'hybrid_improvement': hybrid_improvement,
            'advanced_improvement': advanced_improvement
        }
    }
    
    print(f"\nResults summary:")
    print(f"- Best model: {best_model}")
    print(f"- Hybrid CNN+LSTM improvement: {hybrid_improvement:+.2f}%")
    print(f"- Advanced Hybrid improvement: {advanced_improvement:+.2f}%")
    
    return results

if __name__ == "__main__":
    main()