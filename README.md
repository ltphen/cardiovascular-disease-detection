# ECG Processing Pipeline

This project implements a complete pipeline for processing ECG data from WFDB format files into machine learning-ready datasets. The pipeline supports both classical machine learning approaches and deep learning with CNNs.

## Project Structure

```
.
├── data/                   # Directory for ECG data
│   └── raw/               # Raw WFDB files (Person_01/, Person_02/, etc.)
├── src/                   # Source code
│   ├── data_processing.py # Data loading and preprocessing
│   ├── feature_extraction.py # Feature extraction for classical ML
│   └── models.py         # ML model definitions
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your ECG data in the `data/raw/` directory following the structure:
```
data/raw/
  Person_01/
    rec_1.hea
    rec_1.dat
    rec_1.atr
    rec_2.hea
    ...
  Person_02/
    ...
```

## Usage

1. Data Processing:
```python
from src.data_processing import process_ecg_data

# Process data into windows
X_windows, y_windows = process_ecg_data(
    root_dir="data/raw",
    window_size=1000,  # 2 seconds at 500 Hz
    step=500          # 50% overlap
)
```

2. Feature Extraction (for classical ML):
```python
from src.feature_extraction import extract_features

# Extract features from windows
X_features = extract_features(X_windows)
```

3. Model Training:
```python
from src.models import train_cnn_model, train_classical_model

# Train CNN
cnn_model = train_cnn_model(X_train, y_train, X_val, y_val)

# Or train classical ML model
classical_model = train_classical_model(X_features_train, y_train)
```

## Data Format

The pipeline expects WFDB format files (.hea, .dat, .atr) organized in folders by person:
- Each person folder is named `Person_XX` where XX is a two-digit number
- Each recording consists of three files: rec_N.hea, rec_N.dat, rec_N.atr
- The data is sampled at 500 Hz
- Each recording is 20 seconds long (10000 samples)

## Output

The pipeline produces:
1. Windowed ECG signals (shape: N_windows × window_size × 1)
2. Corresponding labels (shape: N_windows)
3. Optional feature matrix for classical ML (shape: N_windows × n_features)

## License

MIT License 