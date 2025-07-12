import os
import numpy as np
from src.data_processing import load_ptbxl_data
from src.feature_extraction import extract_features
from src.data_augmentation import (
    create_augmented_dataset, 
    preprocess_batch,
    augment_ecg_batch
)
from src.models import (
    train_cnn_model, 
    train_classical_model, 
    train_hybrid_model,
    evaluate_model
)
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

def plot_model_architecture_comparison():
    """Create a visual comparison of model architectures."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # CNN Architecture
    axes[0, 0].text(0.5, 0.5, 'CNN Architecture\n\n'
                    'Input → Conv1D → BatchNorm → MaxPool\n'
                    '→ Conv1D → BatchNorm → MaxPool\n'
                    '→ Conv1D → BatchNorm → GlobalAvgPool\n'
                    '→ Dense → Dropout → Output',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[0, 0].set_title('CNN Model', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Hybrid CNN+LSTM Architecture
    axes[0, 1].text(0.5, 0.5, 'Hybrid CNN+LSTM Architecture\n\n'
                    'Input → Conv1D → BatchNorm → MaxPool\n'
                    '→ Conv1D → BatchNorm → MaxPool\n'
                    '→ Conv1D → BatchNorm\n'
                    '→ LSTM → LSTM → Attention\n'
                    '→ GlobalAvgPool → Dense → Output',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    axes[0, 1].set_title('Hybrid CNN+LSTM Model', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Advanced Hybrid Architecture
    axes[1, 0].text(0.5, 0.5, 'Advanced Hybrid Architecture\n\n'
                    'Input → Multi-scale CNN\n'
                    '→ LSTM layers\n'
                    '→ Multi-head Self-Attention\n'
                    '→ Feed-forward Network\n'
                    '→ Attention → Dense → Output',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    axes[1, 0].set_title('Advanced Hybrid Model', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Data Augmentation Pipeline
    axes[1, 1].text(0.5, 0.5, 'Data Augmentation Pipeline\n\n'
                    'Original Signal\n'
                    '→ Gaussian Noise\n'
                    '→ Time Warping\n'
                    '→ Amplitude Scaling\n'
                    '→ Baseline Wander\n'
                    '→ Random Cropping\n'
                    '→ Frequency Shift',
                    ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    axes[1, 1].set_title('Data Augmentation', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('model_architectures.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_augmentation_examples(X_sample, fs=100):
    """Plot examples of different augmentation techniques."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    from src.data_augmentation import (
        add_gaussian_noise, time_warping, amplitude_scaling,
        baseline_wander, random_cropping, frequency_domain_augmentation
    )
    
    # Original signal
    axes[0, 0].plot(X_sample[:1000, 0])
    axes[0, 0].set_title('Original Signal')
    axes[0, 0].set_ylabel('Amplitude')
    
    # Gaussian noise
    noisy = add_gaussian_noise(X_sample[:1000, 0])
    axes[0, 1].plot(noisy)
    axes[0, 1].set_title('Gaussian Noise')
    
    # Time warping
    warped = time_warping(X_sample[:1000, 0])
    axes[0, 2].plot(warped)
    axes[0, 2].set_title('Time Warping')
    
    # Amplitude scaling
    scaled = amplitude_scaling(X_sample[:1000, 0])
    axes[1, 0].plot(scaled)
    axes[1, 0].set_title('Amplitude Scaling')
    axes[1, 0].set_ylabel('Amplitude')
    
    # Baseline wander
    wandered = baseline_wander(X_sample[:1000, 0])
    axes[1, 1].plot(wandered)
    axes[1, 1].set_title('Baseline Wander')
    
    # Random cropping
    cropped = random_cropping(X_sample[:1000, 0])
    axes[1, 2].plot(cropped)
    axes[1, 2].set_title('Random Cropping')
    
    # Frequency domain
    freq_aug = frequency_domain_augmentation(X_sample[:1000, 0])
    axes[2, 0].plot(freq_aug)
    axes[2, 0].set_title('Frequency Domain')
    axes[2, 0].set_ylabel('Amplitude')
    axes[2, 0].set_xlabel('Samples')
    
    # Combined augmentation
    combined = augment_ecg_batch(X_sample[:1000:1000], augmentation_prob=0.8)[0, :, 0]
    axes[2, 1].plot(combined)
    axes[2, 1].set_title('Combined Augmentation')
    axes[2, 1].set_xlabel('Samples')
    
    # Preprocessed signal
    preprocessed = preprocess_batch(X_sample[:1000:1000])[0, :, 0]
    axes[2, 2].plot(preprocessed)
    axes[2, 2].set_title('Preprocessed Signal')
    axes[2, 2].set_xlabel('Samples')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("="*60)
    print("Enhanced Hybrid ECG Classification with Data Augmentation")
    print("="*60)
    
    # 1. Load and process PTB-XL data
    print("\n1. Loading PTB-XL data...")
    start_time = time.time()
    
    X_train, y_train, X_val, y_val, X_test, y_test, label_names = load_ptbxl_data(
        root_dir="data/raw/",  # <--- CHANGE THIS PATH to your PTB-XL data location
        sampling_frequency=100
    )
    num_classes = y_train.shape[1]

    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    print(f"\nDataset Info:")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_names}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # 2. Data preprocessing and augmentation
    print("\n2. Applying data preprocessing and augmentation...")
    start_time = time.time()
    
    # Preprocess all data
    X_train_processed = preprocess_batch(X_train, fs=100)
    X_val_processed = preprocess_batch(X_val, fs=100)
    X_test_processed = preprocess_batch(X_test, fs=100)
    
    # Create augmented training dataset
    X_train_aug, y_train_aug = create_augmented_dataset(
        X_train_processed, y_train, augmentation_factor=1
    )
    
    print(f"Data preprocessing and augmentation completed in {time.time() - start_time:.2f} seconds")
    print(f"Augmented training samples: {X_train_aug.shape[0]}")

    # 3. Plot model architectures and augmentation examples
    print("\n3. Generating visualization plots...")
    plot_model_architecture_comparison()
    plot_augmentation_examples(X_train[:1], fs=100)

    # 4. Train CNN model (baseline)
    print("\n" + "="*50)
    print("Training CNN model (baseline)...")
    print("="*50)
    start_time = time.time()
    
    cnn_model, cnn_history = train_cnn_model(
        X_train_processed, y_train,
        X_val_processed, y_val,
        num_classes=num_classes,
        batch_size=32,
        epochs=50,
        patience=10
    )
    
    print(f"CNN training completed in {time.time() - start_time:.2f} seconds")
    
    # 5. Train Hybrid CNN+LSTM model with augmented data
    print("\n" + "="*50)
    print("Training Hybrid CNN+LSTM model with augmented data...")
    print("="*50)
    start_time = time.time()
    
    hybrid_model, hybrid_history = train_hybrid_model(
        X_train_aug, y_train_aug,
        X_val_processed, y_val,
        num_classes=num_classes,
        model_type="hybrid",
        batch_size=32,
        epochs=50,
        patience=10,
        lstm_units=128,
        attention_size=128
    )
    
    print(f"Hybrid CNN+LSTM training completed in {time.time() - start_time:.2f} seconds")
    
    # 6. Train Advanced Hybrid model with augmented data
    print("\n" + "="*50)
    print("Training Advanced Hybrid model with augmented data...")
    print("="*50)
    start_time = time.time()
    
    advanced_model, advanced_history = train_hybrid_model(
        X_train_aug, y_train_aug,
        X_val_processed, y_val,
        num_classes=num_classes,
        model_type="advanced",
        batch_size=32,
        epochs=50,
        patience=10
    )
    
    print(f"Advanced Hybrid training completed in {time.time() - start_time:.2f} seconds")
    
    # 7. Train classical ML model
    print("\n" + "="*50)
    print("Training Random Forest model...")
    print("="*50)
    start_time = time.time()
    
    X_train_feat = extract_features(X_train_processed, fs=100)
    X_val_feat = extract_features(X_val_processed, fs=100)
    X_test_feat = extract_features(X_test_processed, fs=100)
    
    rf_model, scaler = train_classical_model(
        X_train_feat, y_train,
        X_val_feat, y_val,
        model_type="rf",
        n_estimators=200
    )
    
    print(f"Random Forest training completed in {time.time() - start_time:.2f} seconds")
    
    # 8. Evaluate all models
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    # Get predictions from all models
    y_pred_cnn = cnn_model.predict(X_test_processed)
    y_pred_hybrid = hybrid_model.predict(X_test_processed)
    y_pred_advanced = advanced_model.predict(X_test_processed)
    
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
    print(f"\n{'Model':<20} | {'Macro AUC':<10} | {'Training Time':<15}")
    print("-" * 55)
    
    training_times = {
        'CNN': 0,  # Will be filled with actual times
        'Hybrid CNN+LSTM': 0,
        'Advanced Hybrid': 0,
        'Random Forest': 0
    }
    
    for name, predictions in models.items():
        macro_auc = roc_auc_score(y_test, predictions, average='macro')
        per_class_auc = [roc_auc_score(y_test[:, i], predictions[:, i]) for i in range(num_classes)]
        
        auc_scores[name] = {
            'macro': macro_auc,
            'per_class': per_class_auc
        }
        
        print(f"{name:<20} | {macro_auc:<10.4f} | {training_times[name]:<15.2f}s")
    
    # 9. Detailed per-class analysis
    print(f"\n{'Class':<10} | {'CNN':<10} | {'Hybrid':<10} | {'Advanced':<10} | {'RF':<10}")
    print("-" * 60)
    for i, label in enumerate(label_names):
        cnn_auc = auc_scores['CNN']['per_class'][i]
        hybrid_auc = auc_scores['Hybrid CNN+LSTM']['per_class'][i]
        advanced_auc = auc_scores['Advanced Hybrid']['per_class'][i]
        rf_auc = auc_scores['Random Forest']['per_class'][i]
        
        print(f"{label:<10} | {cnn_auc:<10.4f} | {hybrid_auc:<10.4f} | {advanced_auc:<10.4f} | {rf_auc:<10.4f}")
    
    # 10. Model comparison analysis
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
    
    # 11. Generate comprehensive plots
    print(f"\nGenerating comprehensive plots...")
    model_names = ['CNN', 'Hybrid CNN+LSTM', 'Advanced Hybrid']
    histories = [cnn_history, hybrid_history, advanced_history]
    
    # Training history plots
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
    plt.savefig('enhanced_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # AUC comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Bar plot of macro AUC scores
    axes[0].bar(list(auc_scores.keys()), [auc_scores[name]['macro'] for name in auc_scores.keys()])
    axes[0].set_title('Macro AUC Comparison')
    axes[0].set_ylabel('Macro AUC')
    axes[0].set_ylim(0.5, 1.0)
    
    # Heatmap of per-class AUC scores
    class_aucs = np.array([[auc_scores[name]['per_class'][i] for name in auc_scores.keys()] 
                           for i in range(len(label_names))])
    
    sns.heatmap(class_aucs, 
                xticklabels=list(auc_scores.keys()), 
                yticklabels=label_names,
                annot=True, 
                fmt='.3f',
                cmap='RdYlBu_r',
                ax=axes[1])
    axes[1].set_title('Per-Class AUC Scores')
    
    plt.tight_layout()
    plt.savefig('enhanced_auc_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlots saved as:")
    print(f"- 'model_architectures.png'")
    print(f"- 'augmentation_examples.png'")
    print(f"- 'enhanced_training_history.png'")
    print(f"- 'enhanced_auc_comparison.png'")
    
    # 12. Save results
    results = {
        'auc_scores': auc_scores,
        'label_names': label_names,
        'model_comparison': {
            'best_model': best_model,
            'hybrid_improvement': hybrid_improvement,
            'advanced_improvement': advanced_improvement
        },
        'dataset_info': {
            'original_samples': X_train.shape[0],
            'augmented_samples': X_train_aug.shape[0],
            'num_classes': num_classes
        }
    }
    
    print(f"\n" + "="*50)
    print("Final Results Summary")
    print("="*50)
    print(f"- Best model: {best_model}")
    print(f"- Hybrid CNN+LSTM improvement: {hybrid_improvement:+.2f}%")
    print(f"- Advanced Hybrid improvement: {advanced_improvement:+.2f}%")
    print(f"- Original training samples: {X_train.shape[0]}")
    print(f"- Augmented training samples: {X_train_aug.shape[0]}")
    print(f"- Data augmentation factor: {X_train_aug.shape[0] / X_train.shape[0]:.1f}x")
    
    return results

if __name__ == "__main__":
    main()