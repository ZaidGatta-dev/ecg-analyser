"""Central paths and constants for the ECG classification pipeline."""

PTBXL_PATH = "data/ptbxl/"
MODEL_DIR = "models/"
PROCESSED_DIR = "processed/"

SAMPLING_RATE = 100  # Hz; PTB-XL also provides 500Hz recordings

VALIDATION_FOLD = 9
TEST_FOLD = 10  # folds are PTB-XL's built-in, patient-stratified split

NUM_LEADS = 12
DIAGNOSTIC_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
