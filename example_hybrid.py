import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report
import tensorflow as tf

# Import original modules
from src.data_processing import load_ptbxl_data
from src.feature_extraction import extract_features
from src.models import train_cnn_model, train_classical_model

# Import new hybrid modules
from src.hybrid_models import train_hybrid_model
from src.data_augmentation import create_augmented_dataset, preprocess_ecg_advanced

def plot_training_history(histories, model_names):
    """Plot training history for comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (history, name) in enumerate(zip(histories, model_names)):
        if history is None:
            continue
            
        # Plot training & validation loss
        axes[0, 0].plot(history.history['loss'], label=f'{name} - Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label=f'{name} - Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Plot training & validation AUC
        if 'auc' in history.history:
            axes[0, 1].plot(history.history['auc'], label=f'{name} - Train AUC')
            axes[0, 1].plot(history.history['val_auc'], label=f'{name} - Val AUC')
            axes[0, 1].set_title('Model AUC')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('AUC')
            axes[0, 1].legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def evaluate_all_models(models, model_names, X_test, y_test, label_names, scaler=None):
    """Evaluate all models and return comprehensive metrics"""
    results = {}
    
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    for model, name in zip(models, model_names):
        if model is None:
            continue
            
        print(f"\n--- {name} ---")
        
        # Prepare test data
        X_test_eval = X_test
        if name == 'Random Forest' and scaler is not None:
            X_test_eval = scaler.transform(X_test_eval)
        
        # Get predictions
        if name == 'Random Forest':
            y_pred = model.predict_proba(X_test_eval)
            y_pred = np.array([pred[:, 1] for pred in y_pred]).T
        else:
            y_pred = model.predict(X_test_eval)
        
        # Calculate metrics
        macro_auc = roc_auc_score(y_test, y_pred, average='macro')
        micro_auc = roc_auc_score(y_test, y_pred, average='micro')
        
        results[name] = {
            'macro_auc': macro_auc,
            'micro_auc': micro_auc,
            'per_class_auc': []
        }
        
        print(f"Macro AUC: {macro_auc:.4f}")
        print(f"Micro AUC: {micro_auc:.4f}")
        print("\nPer-class AUC:")
        
        for i, label in enumerate(label_names):
            class_auc = roc_auc_score(y_test[:, i], y_pred[:, i])
            results[name]['per_class_auc'].append(class_auc)
            print(f"  {label}: {class_auc:.4f}")
    
    return results

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("Loading PTB-XL data...")
    X_train, y_train, X_val, y_val, X_test, y_test, label_names = load_ptbxl_data(
        root_dir="data/raw/",  # Change this to your PTB-XL data location
        sampling_frequency=100
    )
    num_classes = y_train.shape[1]
    
    print(f"Original dataset sizes:")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Apply advanced preprocessing
    print("\nApplying advanced preprocessing...")
    for i in range(X_train.shape[0]):
        X_train[i, :, 0] = preprocess_ecg_advanced(X_train[i, :, 0], sampling_rate=100)
    for i in range(X_val.shape[0]):
        X_val[i, :, 0] = preprocess_ecg_advanced(X_val[i, :, 0], sampling_rate=100)
    for i in range(X_test.shape[0]):
        X_test[i, :, 0] = preprocess_ecg_advanced(X_test[i, :, 0], sampling_rate=100)
    
    # Create augmented training dataset
    print("\nCreating augmented training dataset...")
    X_train_aug, y_train_aug = create_augmented_dataset(
        X_train, y_train, 
        augmentation_factor=3,  # Triple the training data
        sampling_rate=100,
        augmentation_list=['noise', 'time_shift', 'amplitude_scaling', 'baseline_wander', 'power_line']
    )
    
    print(f"Augmented training set size: {X_train_aug.shape[0]}")
    
    # Initialize results storage
    models = []
    model_names = []
    histories = []
    
    # Train original CNN model
    print("\n" + "="*50)
    print("TRAINING ORIGINAL CNN MODEL")
    print("="*50)
    
    cnn_model, cnn_history = train_cnn_model(
        X_train_aug, y_train_aug,
        X_val, y_val,
        num_classes=num_classes,
        batch_size=32,
        epochs=50,
        patience=10
    )
    
    models.append(cnn_model)
    model_names.append('Original CNN')
    histories.append(cnn_history)
    
    # Train CNN-LSTM hybrid model
    print("\n" + "="*50)
    print("TRAINING CNN-LSTM HYBRID MODEL")
    print("="*50)
    
    cnn_lstm_model, cnn_lstm_history = train_hybrid_model(
        model_type='cnn_lstm',
        X_train=X_train_aug,
        y_train=y_train_aug,
        X_val=X_val,
        y_val=y_val,
        num_classes=num_classes,
        batch_size=32,
        epochs=50,
        patience=10,
        lstm_units=128,
        dropout_rate=0.3
    )
    
    models.append(cnn_lstm_model)
    model_names.append('CNN-LSTM')
    histories.append(cnn_lstm_history)
    
    # Train CNN-Transformer hybrid model
    print("\n" + "="*50)
    print("TRAINING CNN-TRANSFORMER HYBRID MODEL")
    print("="*50)
    
    cnn_transformer_model, cnn_transformer_history = train_hybrid_model(
        model_type='cnn_transformer',
        X_train=X_train_aug,
        y_train=y_train_aug,
        X_val=X_val,
        y_val=y_val,
        num_classes=num_classes,
        batch_size=32,
        epochs=50,
        patience=10,
        d_model=128,
        num_heads=8,
        num_transformer_layers=4,
        dropout_rate=0.3
    )
    
    models.append(cnn_transformer_model)
    model_names.append('CNN-Transformer')
    histories.append(cnn_transformer_history)
    
    # Train Attention CNN model
    print("\n" + "="*50)
    print("TRAINING ATTENTION CNN MODEL")
    print("="*50)
    
    attention_cnn_model, attention_cnn_history = train_hybrid_model(
        model_type='attention_cnn',
        X_train=X_train_aug,
        y_train=y_train_aug,
        X_val=X_val,
        y_val=y_val,
        num_classes=num_classes,
        batch_size=32,
        epochs=50,
        patience=10,
        dropout_rate=0.3
    )
    
    models.append(attention_cnn_model)
    model_names.append('Attention CNN')
    histories.append(attention_cnn_history)
    
    # Train Random Forest for comparison
    print("\n" + "="*50)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*50)
    
    print("Extracting features for classical ML...")
    X_train_feat = extract_features(X_train, fs=100)
    X_val_feat = extract_features(X_val, fs=100)
    X_test_feat = extract_features(X_test, fs=100)
    
    rf_model, scaler = train_classical_model(
        X_train_feat, y_train,
        X_val_feat, y_val,
        model_type="rf",
        n_estimators=300
    )
    
    models.append(rf_model)
    model_names.append('Random Forest')
    histories.append(None)
    
    # Plot training histories
    print("\nPlotting training histories...")
    plot_training_history(histories[:-1], model_names[:-1])  # Exclude RF
    
    # Evaluate all models
    results = evaluate_all_models(
        models, model_names, X_test, y_test, label_names, scaler
    )
    
    # Create summary table
    print("\n" + "="*80)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*80)
    
    print(f"{'Model':<20} | {'Macro AUC':<10} | {'Micro AUC':<10} | {'Best Class':<15}")
    print("-" * 70)
    
    for name in model_names:
        if name in results:
            macro_auc = results[name]['macro_auc']
            micro_auc = results[name]['micro_auc']
            best_class_idx = np.argmax(results[name]['per_class_auc'])
            best_class = label_names[best_class_idx]
            best_auc = results[name]['per_class_auc'][best_class_idx]
            
            print(f"{name:<20} | {macro_auc:<10.4f} | {micro_auc:<10.4f} | {best_class:<7}({best_auc:.4f})")
    
    # Find best performing model
    best_model_name = max(results.keys(), key=lambda x: results[x]['macro_auc'])
    best_auc = results[best_model_name]['macro_auc']
    
    print(f"\n🏆 Best Model: {best_model_name} with Macro AUC: {best_auc:.4f}")
    
    # Calculate improvement over baseline
    baseline_auc = results['Original CNN']['macro_auc']
    improvement = ((best_auc - baseline_auc) / baseline_auc) * 100
    
    print(f"📈 Improvement over baseline CNN: {improvement:.2f}%")
    
    # Save detailed results
    print("\nSaving detailed results...")
    with open('model_evaluation_results.txt', 'w') as f:
        f.write("ECG Classification Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        for name in model_names:
            if name in results:
                f.write(f"{name}:\n")
                f.write(f"  Macro AUC: {results[name]['macro_auc']:.4f}\n")
                f.write(f"  Micro AUC: {results[name]['micro_auc']:.4f}\n")
                f.write(f"  Per-class AUC:\n")
                for i, label in enumerate(label_names):
                    f.write(f"    {label}: {results[name]['per_class_auc'][i]:.4f}\n")
                f.write("\n")
        
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Improvement: {improvement:.2f}%\n")
    
    print("✅ Evaluation complete! Results saved to 'model_evaluation_results.txt'")
    print("📊 Training history plots saved to 'training_history.png'")

if __name__ == "__main__":
    main()