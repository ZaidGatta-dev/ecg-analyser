# ECG Diagnostic Analyser (PTB-XL)

A learning project that trains a neural network to detect five broad
diagnostic categories (Normal, Myocardial Infarction, ST/T changes,
Conduction Disturbance, Hypertrophy) from 12-lead ECG recordings, using
the public PTB-XL dataset.

> ⚠️ **This is an educational project, not a medical device.** It has not
> been clinically validated, is not regulatory-approved (e.g. no FDA/MHRA
> clearance), and must never be used to make real diagnostic or treatment
> decisions. Real diagnostic AI tools go through years of clinical
> validation, regulatory review, and safety monitoring before use on
> actual patients.

## 1. Set up your Python environment

```bash
# Create the environment (only needed once)
python -m venv venv

# Activate it (do this every time you work on the project)
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install everything the project needs
pip install -r requirements.txt
```

## 2. Download PTB-XL

The dataset (~1.7GB) is free.

https://physionet.org/content/ptb-xl/

Download it, unzip it, and place its contents so the folder structure
looks like this:

```
ecg_analyser/
└── data/
    └── ptbxl/
        ├── ptbxl_database.csv
        ├── scp_statements.csv
        ├── records100/
        └── records500/
```

(If you put it somewhere else, update `PTBXL_PATH` in `config.py`.)

## 3. Run the pipeline in this order

```bash
# optional: check that files load correctly
python data_loader.py

# optional: check the model architecture builds without errors
python model.py

# Real work: preprocesses data -> trains model -> saves it to models/
python train.py

# Reports per-class ROC-AUC and a classification report on held-out test data
python evaluate.py

# Try the model on one new recording (must be a PTB-XL-format record)
python predict.py data/ptbxl/records100/00000/00001_lr
```

Training on the full dataset with 100Hz signals on a CPU typically takes
somewhere from 20 minutes to a couple of hours depending on your machine.
If it's too slow, try reducing `epochs` in `train.py`, or get it running 
on a GPU (e.g. via Google Colab, which gives you a free GPU in the browser).

## 4. What "good" looks like

Published PTB-XL benchmarks for this exact 5-class task typically land
in the 0.85-0.93 macro-AUC range for well-tuned deep models. Once it's 
working end-to-end, natural next steps are:

- Adding more convolutional layers or trying a residual (ResNet-style) architecture
- Using the 500Hz signals for finer detail
- Class-weighting the loss function to handle imbalance (there are many more NORM examples than HYP)
- Trying the finer-grained "subclass" or "form" labels instead of the 5 superclasses
- Building a small web interface (e.g. with Streamlit) to upload an ECG and see predictions

## Project structure recap

| File | Job |
|---|---|
| `config.py` | Paths and constants |
| `data_loader.py` | Reads raw files off disk into DataFrames/arrays |
| `preprocess.py` | Filters, normalizes, encodes labels, splits data |
| `model.py` | Defines the CNN architecture |
| `train.py` | Runs the full training pipeline |
| `evaluate.py` | Measures performance on held-out test data |
| `predict.py` | Runs the trained model on one new ECG |
