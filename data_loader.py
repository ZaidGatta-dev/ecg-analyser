"""Loads PTB-XL metadata and raw waveform signals from disk."""

import ast
import numpy as np
import pandas as pd
import wfdb

import config


def load_metadata(ptbxl_path: str = config.PTBXL_PATH) -> pd.DataFrame:
    metadata = pd.read_csv(ptbxl_path + "ptbxl_database.csv", index_col="ecg_id")
    metadata.scp_codes = metadata.scp_codes.apply(ast.literal_eval)  # stored as stringified dict
    return metadata


def load_diagnostic_mapping(ptbxl_path: str = config.PTBXL_PATH) -> pd.DataFrame:
    """Maps fine-grained SCP codes (e.g. 'IMI') to broad diagnostic superclasses (e.g. 'MI')."""
    agg_df = pd.read_csv(ptbxl_path + "scp_statements.csv", index_col=0)
    return agg_df[agg_df.diagnostic == 1]


def scp_codes_to_superclass(scp_codes: dict, agg_df: pd.DataFrame) -> list:
    superclasses = [agg_df.loc[code].diagnostic_class
                     for code in scp_codes if code in agg_df.index]
    return list(set(superclasses))


def load_raw_signals(metadata: pd.DataFrame, ptbxl_path: str = config.PTBXL_PATH,
                      sampling_rate: int = config.SAMPLING_RATE) -> np.ndarray:
    """Returns array of shape (n_recordings, n_timesteps, n_leads)."""
    filename_column = "filename_lr" if sampling_rate == 100 else "filename_hr"

    signals = []
    for i, filepath in enumerate(metadata[filename_column]):
        signal, _ = wfdb.rdsamp(ptbxl_path + filepath)
        signals.append(signal)
        if (i + 1) % 2000 == 0:
            print(f"  loaded {i + 1}/{len(metadata)} recordings...")

    return np.array(signals)


if __name__ == "__main__":
    meta = load_metadata()
    diag_map = load_diagnostic_mapping()
    sample_signals = load_raw_signals(meta.iloc[:5])
    print(f"Metadata: {len(meta)} rows | Sample signal shape: {sample_signals.shape}")
