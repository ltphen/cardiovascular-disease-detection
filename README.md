# ECG Heart Disease Classification Project

This project provides a complete pipeline for training machine learning models to classify cardiac conditions from electrocardiogram (ECG) signals. It includes scripts for data processing, feature extraction, and training two types of models: a classical Random Forest and a deep learning 1D Convolutional Neural Network (CNN).

The primary goal is to take a raw ECG signal and predict the presence or absence of five major heart conditions:

- Normal ECG (NORM)
- Myocardial Infarction (MI)
- ST/T Change (STTC)
- Conduction Disturbance (CD)
- Hypertrophy (HYP)

## Dataset

This project is designed for the PTB-XL ECG Dataset, a large, publicly available dataset of clinical 12-lead ECGs.

Source: https://physionet.org/content/ptb-xl/1.0.3/

Why this dataset? It's an excellent resource because it's large (over 21,000 ECGs), has detailed labels from cardiologists, and most importantly, it comes with predefined data splits. This is crucial for creating scientifically valid and reproducible results.

## Project Structure
```
.
├── data/
│   └── ptb-xl/              <-- Download and place the dataset files here
│       ├── ptbxl_database.csv
│       ├── scp_statements.csv
│       ├── records100/
│       └── records500/
├── src/
│   ├── data_processing.py   <-- Logic for loading and preparing data
│   ├── feature_extraction.py  <-- Logic for calculating manual features
│   └── models.py            <-- Code for building and training models
├── example.py               <-- The main script to run the entire pipeline
└── requirements.txt         <-- List of necessary Python libraries
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-link>
cd <your-repo-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the required libraries:
```bash
pip install -r requirements.txt
```

## How to Run

1. Download the PTB-XL dataset from the link above and place its contents into the `data/ptb-xl/` directory.

2. Open the `example.py` script and update the root_dir variable to point to your dataset location:
```python
# in example.py
root_dir="data/ptb-xl/" # Make sure this path is correct
```

3. Run the main script from your terminal:
```bash
python example.py
```

The script will automatically process the data, train both the CNN and Random Forest models, evaluate them, and print a summary of their performance.

## The Data Processing Pipeline: An In-Depth Explanation

The goal of data processing is to convert the messy, raw dataset into clean, numerical arrays that a machine learning model can understand. This is handled by the `load_ptbxl_data` function in `src/data_processing.py`.

### Step 1: Understanding the Metadata

**What we do:** We load `ptbxl_database.csv` and `scp_statements.csv` using the pandas library.

**Why we do it:** The signal files (.dat) are just numbers. All the critical information—the patient ID, the diagnosis, the data splits—lives in these CSV files. Think of them as the map and legend for our dataset.

### Step 2: Decoding Diagnostic Labels

**What we do:** The diagnoses are stored in a cryptic format called scp_codes. We wrote a function to parse these codes and map them to one of the five high-level "superclasses" (NORM, MI, etc.).

**Why we do it:** This simplifies the problem. Instead of trying to predict one of 71 highly specific conditions, we focus on five broad, clinically relevant categories. This makes the model easier to train and its results easier to interpret.

### Step 3: Creating a Multi-Label Target (y)

**What we do:** An ECG can have multiple conditions at once (e.g., both MI and CD). This is a multi-label problem. We convert the list of diagnoses for each ECG into a binary vector.

**Example:** An ECG with MI and CD becomes the vector `[0,1,0,1,0]` (assuming the order is NORM, MI, STTC, CD, HYP).

**Why we do it:** This is the precise numerical format required by the model. It allows the model to predict the probability of each disease independently.

### Step 4: Loading and Normalizing the ECG Waveforms (X)

**What we do:** We loop through each record, load the raw signal data, and perform Z-score normalization.

**Why Z-Score Normalization is CRITICAL:** This step rescales every ECG signal to have a mean of 0 and a standard deviation of 1. Imagine a race where everyone starts at a different position; it would be unfair. Normalization puts every signal on the same "starting line," ensuring that signals with naturally high or low voltage don't unfairly influence the model. It's essential for helping deep learning models train effectively.

### Step 5: The Golden Rule: Patient-Aware Splitting

**What we do:** We use the strat_fold column provided by the dataset's authors to split our data into training, validation, and test sets.

**Why we do it:** This is the most important step for creating a trustworthy model. A simple random split could accidentally place ECGs from the same patient in both the training and test sets. The model could then get high scores by simply "memorizing" the patient's unique ECG pattern, not by learning the features of the disease. The provided folds guarantee that a patient's data is only in one split, preventing this data leakage and giving us a realistic measure of how the model will perform on new, unseen patients.

## The Models: Two Philosophies

We build and compare two types of models to see which approach works best.

### 1. The Classical Approach: Random Forest

This is a traditional machine learning pipeline that relies on "hand-crafted" features.

**Feature Extraction (feature_extraction.py):** We first manually calculate a set of meaningful features from the ECG signal.

- **Time-Domain Features:** Statistics like mean, standard deviation, max/min values, and skewness. These describe the shape and morphology of the waves.
- **Frequency-Domain Features:** Features derived from a Fourier Transform (FFT). These describe the rhythm and periodic components of the signal.

**The Model (RandomForestClassifier):**

- **What it is:** A powerful model that works like a committee of experts. It builds hundreds of simple "decision trees" and then aggregates their votes to make a final prediction.
- **Why it was chosen:** Random Forests are robust, handle complex data well, and are less prone to overfitting than single decision trees. They are a very strong baseline for any classification task.

### 2. The Deep Learning Approach: 1D Convolutional Neural Network (CNN)

This is an "end-to-end" approach where the model learns the important features by itself, directly from the raw signal. Think of it as an automated microscope that learns to find patterns relevant to disease.

Here is a breakdown of the CNN architecture (`build_multilabel_cnn_model` in `src/models.py`):

**Input Layer:** The "front door" for the data, shaped to accept our 10-second ECG signals.

**Convolutional Layers (Conv1D):** These are the core of the CNN.
- **What they do:** They use a set of "filters" (or kernels) that slide across the signal. Each filter is a small pattern detector. The first layer learns to find simple patterns like slopes and curves. Deeper layers combine these to find more complex patterns, like a full QRS complex or an abnormal T-wave.
- **Our choices:** We use three Conv1D layers with decreasing kernel_size (7, 5, 3). This allows the model to find broad patterns first and then refine them with more detailed filters.

**Batch Normalization (BatchNormalization):**
- **What it does:** It's a "stabilizer" inserted between layers. It rescales the data flowing through the network to keep it in a healthy range.
- **Why it's there:** It dramatically speeds up training and helps prevent the model from getting "stuck," leading to better overall performance.

**Pooling Layers (MaxPooling1D):**
- **What they do:** They downsample the signal by keeping only the most important information from a small region.
- **Why they're there:** They make the model more efficient by reducing the amount of data it has to process. They also make the model more robust by helping it recognize a pattern even if it's shifted slightly in time.

**Global Average Pooling (GlobalAveragePooling1D):**
- **What it does:** After all the feature detection, this layer creates a "grand summary" by averaging each feature map into a single number. This creates a compact, fixed-size vector that represents the entire 10-second ECG.

**Dense Layers:**
- **What they do:** These are the "brain" of the model. They take the summary vector from the pooling layer and learn the complex combinations of features that are associated with each disease.
- **Our choices:** A Dense layer followed by Dropout (which helps prevent overfitting by randomly ignoring some neurons during training) acts as the final reasoning engine.

**Output Layer (Dense(5, activation='sigmoid')):**
- **What it does:** This is the final decision-maker. It has 5 neurons—one for each disease class.
- **Why sigmoid is the crucial choice:** Unlike softmax (which forces a single choice), sigmoid activation treats each neuron independently. It outputs a probability between 0 and 1 for each disease. This allows the model to say, for example, "I'm 95% sure Myocardial Infarction is present, and 80% sure Conduction Disturbance is present," which is exactly what we need for a multi-label problem.

## License

MIT License 