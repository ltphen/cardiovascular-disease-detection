import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dropout
from typing import Tuple, Optional

class SelfAttention(layers.Layer):
    """Custom Self-Attention layer for ECG signals"""
    
    def __init__(self, d_model: int, num_heads: int = 8, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=0.1
        )
        self.layernorm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.1)
        
    def call(self, inputs, training=None):
        # Apply multi-head attention
        attn_output = self.mha(
            query=inputs,
            key=inputs,
            value=inputs,
            training=training
        )
        
        # Add residual connection and layer normalization
        out = self.layernorm(inputs + self.dropout(attn_output, training=training))
        return out

class PositionalEncoding(layers.Layer):
    """Positional encoding for transformer-based models"""
    
    def __init__(self, max_len: int = 1000, d_model: int = 128, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        
        # Create positional encoding matrix
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe[np.newaxis, ...], dtype=tf.float32)
        
    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]

def build_cnn_lstm_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    lstm_units: int = 128,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Build a CNN-LSTM hybrid model for ECG classification.
    
    Args:
        input_shape: Shape of input (seq_len, features)
        num_classes: Number of output classes
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # CNN Feature Extraction
    x = layers.Conv1D(64, kernel_size=7, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # LSTM for Temporal Dependencies
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate))(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units // 2, return_sequences=False, dropout=dropout_rate))(x)
    
    # Classification Head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'binary_accuracy',
            tf.keras.metrics.AUC(name='auc', multi_label=True)
        ]
    )
    
    return model

def build_cnn_transformer_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    d_model: int = 128,
    num_heads: int = 8,
    num_transformer_layers: int = 4,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Build a CNN-Transformer hybrid model for ECG classification.
    
    Args:
        input_shape: Shape of input (seq_len, features)
        num_classes: Number of output classes
        d_model: Dimension of the model
        num_heads: Number of attention heads
        num_transformer_layers: Number of transformer layers
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # CNN Feature Extraction
    x = layers.Conv1D(64, kernel_size=7, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    x = layers.Conv1D(d_model, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Positional Encoding
    x = PositionalEncoding(max_len=input_shape[0]//4, d_model=d_model)(x)
    
    # Transformer Layers
    for _ in range(num_transformer_layers):
        x = SelfAttention(d_model=d_model, num_heads=num_heads)(x)
        
        # Feed-forward network
        ff_output = layers.Dense(d_model * 4, activation='relu')(x)
        ff_output = layers.Dense(d_model)(ff_output)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        
        # Residual connection and layer normalization
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification Head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'binary_accuracy',
            tf.keras.metrics.AUC(name='auc', multi_label=True)
        ]
    )
    
    return model

def build_attention_cnn_model(
    input_shape: Tuple[int, int],
    num_classes: int,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Build a CNN model with attention mechanisms for ECG classification.
    
    Args:
        input_shape: Shape of input (seq_len, features)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Multi-scale CNN Feature Extraction
    # Branch 1: Small kernels for detailed features
    branch1 = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    branch1 = layers.BatchNormalization()(branch1)
    
    # Branch 2: Medium kernels for intermediate features
    branch2 = layers.Conv1D(64, kernel_size=7, activation='relu', padding='same')(inputs)
    branch2 = layers.BatchNormalization()(branch2)
    
    # Branch 3: Large kernels for global features
    branch3 = layers.Conv1D(64, kernel_size=15, activation='relu', padding='same')(inputs)
    branch3 = layers.BatchNormalization()(branch3)
    
    # Concatenate multi-scale features
    x = layers.Concatenate()([branch1, branch2, branch3])
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Second CNN layer
    x = layers.Conv1D(256, kernel_size=5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Third CNN layer
    x = layers.Conv1D(512, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Self-Attention Mechanism
    x = SelfAttention(d_model=512, num_heads=8)(x)
    
    # Global Average Pooling with Attention Weights
    attention_weights = layers.Dense(1, activation='softmax')(x)
    x = layers.Multiply()([x, attention_weights])
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification Head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'binary_accuracy',
            tf.keras.metrics.AUC(name='auc', multi_label=True)
        ]
    )
    
    return model

def train_hybrid_model(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    batch_size: int = 32,
    epochs: int = 100,
    patience: int = 15,
    **model_kwargs
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Train a hybrid model on ECG data.
    
    Args:
        model_type: Type of hybrid model ('cnn_lstm', 'cnn_transformer', 'attention_cnn')
        X_train: Training data of shape (n_samples, seq_len, features)
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        num_classes: Number of output classes
        batch_size: Batch size for training
        epochs: Maximum number of epochs
        patience: Patience for early stopping
        **model_kwargs: Additional model parameters
        
    Returns:
        Tuple of (trained model, training history)
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Build model based on type
    if model_type == 'cnn_lstm':
        model = build_cnn_lstm_model(input_shape, num_classes, **model_kwargs)
    elif model_type == 'cnn_transformer':
        model = build_cnn_transformer_model(input_shape, num_classes, **model_kwargs)
    elif model_type == 'attention_cnn':
        model = build_attention_cnn_model(input_shape, num_classes, **model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f'best_{model_type}_model.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    print(f"Training {model_type} model...")
    print(f"Model summary:")
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history