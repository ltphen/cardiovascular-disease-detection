import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

def build_cnn_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    dropout_rate: float = 0.5
) -> tf.keras.Model:
    """
    Build a 1D CNN model for ECG classification.
    
    Args:
        input_shape: Shape of input windows (window_size, 1)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First conv block
        layers.Conv1D(32, kernel_size=7, activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Second conv block
        layers.Conv1D(64, kernel_size=5, activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        # Third conv block
        layers.Conv1D(128, kernel_size=3, activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        
        # Dense layers
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="sigmoid")
    ])
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "binary_accuracy",
            tf.keras.metrics.AUC(name='auc', multi_label=True)
        ]
    )
    
    return model

def train_cnn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    batch_size: int = 32,
    epochs: int = 50,
    patience: int = 5,
    dropout_rate: float = 0.5
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Train a CNN model on ECG windows.
    
    Args:
        X_train: Training data of shape (n_samples, window_size, 1)
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        num_classes: Number of output classes
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        patience: Patience for early stopping
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Tuple of (trained model, training history)
    """
    # Build model
    model = build_cnn_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def train_classical_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_type: str = "rf",
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Train a classical ML model (Random Forest) on ECG features.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_type: Type of model ("rf" for Random Forest)
        n_estimators: Number of trees for Random Forest
        max_depth: Maximum depth of trees
        random_state: Random seed
        
    Returns:
        Tuple of (trained model, fitted scaler)
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train_scaled, y_train)
    
    # Print validation accuracy
    val_acc = model.score(X_val_scaled, y_val)
    print(f"Validation accuracy: {val_acc:.3f}")
    
    return model, scaler

def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: Optional[StandardScaler] = None
) -> float:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model (either Keras model or sklearn model)
        X_test: Test data
        y_test: Test labels
        scaler: Optional scaler for classical ML models
        
    Returns:
        Test accuracy
    """
    if scaler is not None:
        X_test = scaler.transform(X_test)
        
    if isinstance(model, tf.keras.Model):
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {test_acc:.3f}")
        return test_acc
    else:
        test_acc = model.score(X_test, y_test)
        print(f"Test accuracy: {test_acc:.3f}")
        return test_acc 