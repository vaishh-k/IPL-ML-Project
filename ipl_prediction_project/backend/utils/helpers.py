# ==============================================================================
# IPL ML LAB - SYSTEM FILE
# ==============================================================================
# File: helpers.py
# Path: backend/utils/helpers.py
# Description: Common backend helper utilities for path resolving, logging setups, and status messages.
# ==============================================================================

import os
import shutil
import pandas as pd
import numpy as np

TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp_data")

def get_session_dir(session_id: str) -> str:
    """Gets or creates the directory for the given session_id."""
    session_dir = os.path.join(TEMP_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

def get_stage_filepath(session_id: str, stage: str) -> str:
    """Returns the filepath for a specific stage of the dataframe in a session."""
    return os.path.join(get_session_dir(session_id), f"{stage}.pkl")

def save_dataframe(df: pd.DataFrame, session_id: str, stage: str) -> None:
    """Saves the dataframe to disk for the given session and stage."""
    filepath = get_stage_filepath(session_id, stage)
    # Reset index to avoid any multiindex serialization issues, but preserve rows
    df.to_pickle(filepath)

def load_dataframe(session_id: str, stage: str) -> pd.DataFrame:
    """Loads the dataframe from disk for the given session and stage."""
    filepath = get_stage_filepath(session_id, stage)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataframe for stage '{stage}' does not exist in session '{session_id}'.")
    return pd.read_pickle(filepath)

def get_latest_dataframe(session_id: str) -> tuple[pd.DataFrame, str]:
    """
    Attempts to load the most advanced stage dataframe available for the session.
    Returns the dataframe and the name of the stage.
    """
    stages = ["pca", "scaled", "encoded", "cleaned", "original"]
    for stage in stages:
        filepath = get_stage_filepath(session_id, stage)
        if os.path.exists(filepath):
            return load_dataframe(session_id, stage), stage
    raise FileNotFoundError(f"No dataframe found for session '{session_id}'.")

def has_stage_dataframe(session_id: str, stage: str) -> bool:
    """Checks if a dataframe for a stage exists."""
    return os.path.exists(get_stage_filepath(session_id, stage))

def clean_session_data(session_id: str) -> None:
    """Deletes all files associated with a session."""
    session_dir = os.path.join(TEMP_DIR, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)

def clean_old_sessions() -> None:
    """Helper to clear temp directory if it exists, used on startup."""
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception:
            pass
    os.makedirs(TEMP_DIR, exist_ok=True)

def sanitize_nans(obj):
    """
    Recursively replaces NaN, Inf, -Inf with None (null in JSON),
    and converts numpy arrays and scalars to native Python types.
    """
    if isinstance(obj, np.ndarray):
        return [sanitize_nans(x) for x in obj.tolist()]
        
    if isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
        return int(obj)
        
    if isinstance(obj, np.floating):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
        
    if isinstance(obj, np.bool_):
        return bool(obj)
        
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_nans(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_nans(x) for x in obj]
    return obj
